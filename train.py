import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from model import build_unet, Patches, PatchEncoder, transformer_encoder
import pandas as pd
import json

# Constants
H = 256
W = 256
NUM_CLASSES = 4


class TrainingStateManager:
    """Manages the training state across sessions"""

    def __init__(self, model_dir):
        self.state_file = os.path.join(model_dir, "training_state.json")
        self.current_fold = 1
        self.current_epoch = 0
        self.fold_metrics = []
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.current_fold = state.get('current_fold', 1)
                self.current_epoch = state.get('current_epoch', 0)
                self.fold_metrics = state.get('fold_metrics', [])
                print(f"Resumed from fold {self.current_fold}, epoch {self.current_epoch}")
        else:
            print("Starting new training session")

    def save_state(self, fold, epoch, metrics=None):
        state = {
            'current_fold': fold,
            'current_epoch': epoch,
            'fold_metrics': self.fold_metrics if metrics is None else metrics
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)

    def update_metrics(self, fold_metrics):
        self.fold_metrics = fold_metrics
        self.save_state(self.current_fold, self.current_epoch, self.fold_metrics)


class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    """Custom ReduceLROnPlateau that saves and loads its state"""

    def __init__(self, monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1, mode='min',
                 cooldown=0, min_delta=1e-4, state_file='rlrop_state.json'):
        super(CustomReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.mode = mode
        self.cooldown = cooldown
        self.min_delta = min_delta
        self.cooldown_counter = 0
        self.wait = 0
        self.state_file = state_file

        self.mode_dict = {'min': (np.Inf, 'lt'), 'max': (-np.Inf, 'gt')}
        self.best, self.monitor_op = self.mode_dict[mode]

    def on_train_begin(self, logs=None):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.best = state.get('best', self.best)
                self.wait = state.get('wait', self.wait)
                self.cooldown_counter = state.get('cooldown_counter', self.cooldown_counter)
                new_lr = state.get('lr', float(tf.keras.backend.get_value(self.model.optimizer.lr)))
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.mode == 'min':
            is_better = current < self.best - self.min_delta
        else:
            is_better = current > self.best + self.min_delta

        if is_better:
            self.best = current
            self.wait = 0
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            elif self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0
                self.cooldown_counter = self.cooldown
            else:
                self.wait += 1

        self._save_state()

    def _save_state(self):
        state = {
            'best': float(self.best),
            'wait': self.wait,
            'lr': float(tf.keras.backend.get_value(self.model.optimizer.lr)),
            'cooldown_counter': self.cooldown_counter
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint that saves epoch number"""

    def __init__(self, state_manager, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)
        self.state_manager = state_manager

    def on_epoch_end(self, epoch, logs=None):
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.state_manager.current_epoch = epoch + 1
        self.state_manager.save_state(self.state_manager.current_fold, self.state_manager.current_epoch)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    x = sorted(glob(os.path.join(path, "images", "*.png")))
    y = sorted(glob(os.path.join(path, "masks", "*.png")))
    return x, y


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    return x.astype(np.float32)


def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (H, W))
    x[(x >= 0) & (x <= 42)] = 0
    x[(x >= 43) & (x <= 127)] = 1
    x[(x >= 128) & (x <= 212)] = 2
    x[(x >= 213) & (x <= 255)] = 3
    return x.astype(np.int32)


def tf_dataset(x, y, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image = read_image(x)
        mask = read_mask(y)
        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, NUM_CLASSES, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, NUM_CLASSES])
    return image, mask


def train_fold(fold_num, train_x, train_y, valid_x, valid_y, base_dir, state_manager,
               batch_size=8, num_epochs=100, initial_epoch=0):
    fold_dir = os.path.join(base_dir, f"fold_{fold_num}")
    create_dir(fold_dir)

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size)

    train_steps = (len(train_x) + batch_size - 1) // batch_size
    valid_steps = (len(valid_x) + batch_size - 1) // batch_size

    model_path = os.path.join(fold_dir, "model.h5")
    
    # Define custom objects dictionary
    custom_objects = {
        'build_unet': build_unet,
        'Patches': Patches,
        'PatchEncoder': PatchEncoder,
        'transformer_encoder': transformer_encoder
    }

    if os.path.exists(model_path) and initial_epoch > 0:
        try:
            # Load the complete model with custom objects
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Loaded complete model from epoch {initial_epoch}")
        except Exception as e:
            print(f"Error loading complete model: {e}")
            print("Attempting to load weights only...")
            # If loading complete model fails, create new model and load weights
            model = build_unet((H, W, 3), NUM_CLASSES)
            model.load_weights(model_path)
            print(f"Loaded weights from epoch {initial_epoch}")
    else:
        model = build_unet((H, W, 3), NUM_CLASSES)

    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        CustomModelCheckpoint(
            state_manager=state_manager,
            filepath=model_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch'
        ),
        CustomReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode='min',
            state_file=os.path.join(fold_dir, "rlrop_state.json")
        ),
        CSVLogger(os.path.join(fold_dir, "log.csv"), append=True),
        TensorBoard(log_dir=os.path.join(fold_dir, "logs")),
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=1,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_steps,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

    val_metrics = pd.DataFrame(history.history)
    best_epoch = val_metrics['val_loss'].idxmin()
    
    return {
        'accuracy': val_metrics.loc[best_epoch, 'val_accuracy'],
        'loss': val_metrics.loc[best_epoch, 'val_loss']
    }


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset_path = os.path.join("datasets", "2 CH")
    model_dir = os.path.join("2_CH", "Kfold_training")
    create_dir(model_dir)
    state_manager = TrainingStateManager(model_dir)

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")

    x_train, y_train = load_data(train_path)
    x_val, y_val = load_data(val_path)
    x_all = x_train + x_val
    y_all = y_train + y_val
    x_all, y_all = shuffling(x_all, y_all)

    print(f"Total samples: {len(x_all)}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = state_manager.fold_metrics

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_all), 1):
        if fold < state_manager.current_fold:
            continue

        print(f"\nTraining Fold {fold}")
        print("-" * 50)

        fold_train_x = [x_all[i] for i in train_idx]
        fold_train_y = [y_all[i] for i in train_idx]
        fold_val_x = [x_all[i] for i in val_idx]
        fold_val_y = [y_all[i] for i in val_idx]

        metrics = train_fold(
            fold_num=fold,
            train_x=fold_train_x,
            train_y=fold_train_y,
            valid_x=fold_val_x,
            valid_y=fold_val_y,
            base_dir=model_dir,
            state_manager=state_manager,
            batch_size=8,
            num_epochs=100,
            initial_epoch=state_manager.current_epoch if fold == state_manager.current_fold else 0
        )

        fold_metrics.append(metrics)
        state_manager.update_metrics(fold_metrics)

        print(f"Fold {fold} - Accuracy: {metrics['accuracy']:.3f}, Loss: {metrics['loss']:.3f}")

        state_manager.current_fold = fold + 1
        state_manager.current_epoch = 0
        state_manager.save_state(state_manager.current_fold, state_manager.current_epoch)

    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    final_metrics = {
        'accuracy': f"{mean_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}",
        'loss': f"{mean_metrics['loss']:.3f} ± {std_metrics['loss']:.3f}"
    }

    metrics_path = os.path.join(model_dir, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print("\nFinal Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric.capitalize()}: {value}")


if __name__ == "__main__":
    main()
