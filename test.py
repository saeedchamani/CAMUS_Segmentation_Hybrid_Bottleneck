import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import (
    f1_score, fbeta_score, jaccard_score, precision_score, recall_score
)
from scipy.spatial.distance import directed_hausdorff
from model import build_unet, Patches, PatchEncoder, transformer_encoder
import json

# Constants
H = 256
W = 256
NUM_CLASSES = 4


def create_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    """Load images and masks from the specified path"""
    x = sorted(glob(os.path.join(path, "images", "*.png")))
    y = sorted(glob(os.path.join(path, "masks", "*.png")))
    return x, y


def read_image(path):
    """Read and preprocess image"""
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    """Read and preprocess mask"""
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (H, W))
    x[(x >= 0) & (x <= 42)] = 0
    x[(x >= 43) & (x <= 127)] = 1
    x[(x >= 128) & (x <= 212)] = 2
    x[(x >= 213) & (x <= 255)] = 3
    x = x.astype(np.int32)
    return x


class MetricsCalculator:
    @staticmethod
    def calculate_specificity(y_true, y_pred):
        """Calculate specificity metric"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat != 0))
        return tn / (tn + fp + 1e-7)

    @staticmethod
    def calculate_dice_coefficient(y_true, y_pred):
        """Calculate Dice coefficient"""
        smooth = 1e-7
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    @staticmethod
    def calculate_hausdorff_distance(y_true, y_pred):
        """Calculate Hausdorff distance"""
        return directed_hausdorff(y_true, y_pred)[0]

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate all metrics for binary case"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        return {
            'f1': f1_score(y_true_flat, y_pred_flat, average='binary'),
            'f2': fbeta_score(y_true_flat, y_pred_flat, beta=2, average='binary'),
            'jaccard': jaccard_score(y_true_flat, y_pred_flat, average='binary'),
            'precision': precision_score(y_true_flat, y_pred_flat, average='binary'),
            'recall': recall_score(y_true_flat, y_pred_flat, average='binary'),
            'specificity': MetricsCalculator.calculate_specificity(y_true, y_pred),
            'hausdorff': MetricsCalculator.calculate_hausdorff_distance(y_true, y_pred),
            'dice': MetricsCalculator.calculate_dice_coefficient(y_true, y_pred)
        }


def test_fold(fold_num, test_x, test_y, model_dir):
    """Test a single fold and return metrics"""
    # Load model
    model_path = os.path.join(model_dir, f"fold_{fold_num}", "model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={'build_unet': build_unet, 'Patches': Patches, 'PatchEncoder': PatchEncoder, 'transformer_encoder': transformer_encoder})
    # Create results directory for this fold
    results_dir = os.path.join(model_dir, f"fold_{fold_num}", "test_results")
    create_dir(results_dir)

    # Initialize metrics storage
    metrics_per_class = {i: [] for i in range(NUM_CLASSES)}
    overall_metrics = []

    # Process each test image
    for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x_path).split('.')[0]

        # Read and preprocess
        x = read_image(x_path)
        y_true = read_mask(y_path)

        # Predict
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = np.argmax(y_pred, axis=-1)

        # Calculate metrics for each class
        for class_id in range(NUM_CLASSES):
            y_true_class = (y_true == class_id).astype(np.int32)
            y_pred_class = (y_pred == class_id).astype(np.int32)

            metrics = MetricsCalculator.calculate_metrics(y_true_class, y_pred_class)
            metrics['image_name'] = name
            metrics['class_id'] = class_id
            metrics_per_class[class_id].append(metrics)

        # Save visualization
        save_visualization(x, y_true, y_pred, name, results_dir)

    return metrics_per_class


def save_visualization(x, y_true, y_pred, name, save_dir):
    """Save visualization of predictions"""
    # Convert to visualization format
    y_true_viz = np.expand_dims(y_true, axis=-1) * (255 / NUM_CLASSES)
    y_pred_viz = np.expand_dims(y_pred, axis=-1) * (255 / NUM_CLASSES)

    # Convert to RGB
    y_true_viz = np.concatenate([y_true_viz] * 3, axis=-1).astype(np.uint8)
    y_pred_viz = np.concatenate([y_pred_viz] * 3, axis=-1).astype(np.uint8)
    x_viz = (x * 255).astype(np.uint8)

    # Create separator
    h, w, _ = x_viz.shape
    separator = np.ones((h, 10, 3), dtype=np.uint8) * 255

    # Concatenate images
    final_image = np.concatenate([x_viz, separator, y_true_viz, separator, y_pred_viz], axis=1)

    # Save
    cv2.imwrite(os.path.join(save_dir, f"{name}.png"), final_image)


def save_metrics(metrics_per_class, fold_num, model_dir):
    """Save metrics for a fold"""
    results_dir = os.path.join(model_dir, f"fold_{fold_num}", "test_results")

    # Save per-class metrics
    for class_id, metrics_list in metrics_per_class.items():
        df = pd.DataFrame(metrics_list)
        df.to_csv(os.path.join(results_dir, f"metrics_class_{class_id}.csv"), index=False)

        # Print average metrics for this class
        print(f"\nClass {class_id} Average Metrics:")
        for metric in ['f1', 'f2', 'jaccard', 'precision', 'recall', 'specificity', 'hausdorff', 'dice']:
            avg_value = df[metric].mean()
            print(f"{metric.capitalize()}: {avg_value:.4f}")


def main():
    """Main function to run the testing"""
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directories
    dataset_path = os.path.join("datasets", "2 CH")
    model_dir = os.path.join("2_CH", "Kfold_training")

    # Load test data
    test_path = os.path.join(dataset_path, "test")
    test_x, test_y = load_data(test_path)

    # Test each fold
    all_folds_metrics = []
    for fold in range(1, 6):  # 5-fold cross validation
        print(f"\nTesting Fold {fold}")
        print("-" * 50)

        metrics_per_class = test_fold(fold, test_x, test_y, model_dir)
        save_metrics(metrics_per_class, fold, model_dir)

        # Store fold metrics for averaging
        fold_metrics = {
            'fold': fold,
            'metrics_per_class': metrics_per_class
        }
        all_folds_metrics.append(fold_metrics)

    # Calculate and save average metrics across all folds
    print("\nAverage Metrics Across All Folds:")
    metrics_summary = calculate_average_metrics(all_folds_metrics)
    save_summary_metrics(metrics_summary, model_dir)


def calculate_average_metrics(all_folds_metrics):
    """Calculate average metrics across all folds"""
    metrics_keys = ['f1', 'f2', 'jaccard', 'precision', 'recall', 'specificity', 'hausdorff', 'dice']
    summary = {class_id: {metric: [] for metric in metrics_keys} for class_id in range(NUM_CLASSES)}

    # Collect metrics from all folds
    for fold_data in all_folds_metrics:
        metrics_per_class = fold_data['metrics_per_class']
        for class_id in range(NUM_CLASSES):
            class_metrics = metrics_per_class[class_id]
            for metric in metrics_keys:
                avg_metric = np.mean([m[metric] for m in class_metrics])
                summary[class_id][metric].append(avg_metric)

    # Calculate mean and std for each metric
    final_summary = {}
    for class_id in range(NUM_CLASSES):
        final_summary[class_id] = {}
        for metric in metrics_keys:
            values = summary[class_id][metric]
            final_summary[class_id][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    return final_summary


def save_summary_metrics(metrics_summary, model_dir):
    """Save summary metrics to JSON and display them"""
    summary_path = os.path.join(model_dir, "test_metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    # Display summary
    for class_id in range(NUM_CLASSES):
        print(f"\nClass {class_id} Summary:")
        for metric, values in metrics_summary[class_id].items():
            print(f"{metric.capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")


if __name__ == "__main__":
    main()