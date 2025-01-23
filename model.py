import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D
from keras import backend as K


def SE(inputs, ratio=8):
    channel_axis = -1
    num_filters = inputs.shape[channel_axis]
    se_shape = (1, 1, num_filters)

    x = L.GlobalAveragePooling2D()(inputs)
    x = L.Reshape(se_shape)(x)
    x = L.Dense(num_filters // ratio, activation="relu", use_bias=False)(x)
    x = L.Dense(num_filters, activation="sigmoid", use_bias=False)(x)

    x = L.Multiply()([inputs, x])
    return x


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = L.Dense(channel // ratio,
                               activation='relu',
                               kernel_initializer='he_normal',
                               use_bias=True,
                               bias_initializer='zeros')
    shared_layer_two = L.Dense(channel,
                               kernel_initializer='he_normal',
                               use_bias=True,
                               bias_initializer='zeros')

    avg_pool = L.GlobalAveragePooling2D()(input_feature)
    avg_pool = L.Reshape((1, 1, channel))(avg_pool)

    assert avg_pool.shape[1:] == (1, 1, channel), f"Average pool shape {avg_pool.shape[1:]} != (1, 1, {channel})"

    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (
    1, 1, channel // ratio), f"Average pool shape after dense1 {avg_pool.shape[1:]} != (1, 1, {channel // ratio})"

    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (
    1, 1, channel), f"Average pool shape after dense2 {avg_pool.shape[1:]} != (1, 1, {channel})"

    max_pool = L.GlobalMaxPooling2D()(input_feature)
    max_pool = L.Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel), f"Max pool shape {max_pool.shape[1:]} != (1, 1, {channel})"

    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (
    1, 1, channel // ratio), f"Max pool shape after dense1 {max_pool.shape[1:]} != (1, 1, {channel // ratio})"

    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (
    1, 1, channel), f"Max pool shape after dense2 {max_pool.shape[1:]} != (1, 1, {channel})"

    cbam_feature = L.Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = L.Permute((3, 1, 2))(cbam_feature)

    return L.multiply([input_feature, cbam_feature])




def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = L.Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    # Average pooling across channels
    avg_pool = L.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1, f"Average pool last dimension {avg_pool.shape[-1]} != 1"

    # Max pooling across channels
    max_pool = L.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1, f"Max pool last dimension {max_pool.shape[-1]} != 1"

    # Concatenate pooled features
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2, f"Concatenated feature last dimension {concat.shape[-1]} != 2"

    # Apply spatial attention with convolution
    cbam_feature = Conv2D(filters=1,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same',
                         activation='sigmoid',
                         kernel_initializer='he_normal',
                         use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1, f"CBAM feature last dimension {cbam_feature.shape[-1]} != 1"

    # Handle channels_first format if needed
    if K.image_data_format() == "channels_first":
        cbam_feature = L.Permute((3, 1, 2))(cbam_feature)

    return L.multiply([input_feature, cbam_feature])



def cbam_block(cbam_feature, ratio=8):

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SE(x,ratio=8)

    y = Conv2D(num_filters, 1, padding="same")(inputs)

    x = L.Add()([x, y])
    x = Conv2D(num_filters, 1, padding="same")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p



def attention_gate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv2D(num_filters, 1, padding="same")(out)
    out = L.Activation("sigmoid")(out)

    return out * s

def decoder_block(x, s, num_filters):
    x = L.UpSampling2D(interpolation="bilinear")(x)
    s = attention_gate(x, s, num_filters)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# Define the PatchEncoder layer to encode patches with positional embeddings
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.projection = layers.Dense(embedding_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embedding_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        embedded_patches = self.projection(patches)
        encoded = embedded_patches + self.position_embedding(positions)
        return encoded


# Define the Transformer encoder block
def transformer_encoder(inputs, num_heads, mlp_dim, dropout_rate):
    # Layer Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    # Multi-Head Attention
    key_dim = inputs.shape[-1] // num_heads
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x, x)
    # Skip Connection
    x = layers.Add()([x, inputs])
    # Layer Normalization
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    # MLP
    y = layers.Dense(mlp_dim, activation='gelu')(y)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    # Skip Connection
    outputs = layers.Add()([x, y])
    return outputs



def ASPP(inputs):

    """ 1x1 conv """
    y1 = Conv2D(512, 1, padding="same", use_bias=False)(inputs)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)

    """ 3x3 conv """
    y2 = Conv2D(512, 3, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    """ 5x5 conv """
    y3 = Conv2D(512, 5, padding="same", use_bias=False)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    """ 7x7 conv """
    y4 = Conv2D(512, 7, padding="same", use_bias=False)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y = Concatenate()([y1, y2, y3, y4])

    shape = y.shape
    x1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(y)
    x1 = Conv2D(512, 1, padding="same", use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(x1)

    """ 1x1 conv """
    x2 = Conv2D(512, 1, padding="same", use_bias=False)(y)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    """ 3x3 conv rate=6 """
    x3 = Conv2D(512, 3, padding="same", use_bias=False, dilation_rate=6)(y)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    """ 3x3 conv rate=9 """
    x4 = Conv2D(512, 3, padding="same", use_bias=False, dilation_rate=9)(y)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    """ 3x3 conv rate=12 """
    x5 = Conv2D(512, 3, padding="same", use_bias=False, dilation_rate=12)(y)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)

    x = Concatenate()([x1, x2, x3, x4, x5])
    x = Conv2D(512, 1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def aspp_decoder(inputs, num_filters):
    x1 = L.Conv2D(num_filters, 3, dilation_rate=6, padding='same')(inputs)
    x1 = L.BatchNormalization()(x1)

    x2 = L.Conv2D(num_filters, 3, dilation_rate=12, padding='same')(inputs)
    x2 = L.BatchNormalization()(x2)

    x3 = L.Conv2D(num_filters, 3, dilation_rate=18, padding='same')(inputs)
    x3 = L.BatchNormalization()(x3)

    x4 = L.Conv2D(num_filters, (3,3) , dilation_rate=6, padding='same')(inputs)
    x4 = L.BatchNormalization()(x4)

    y = L.Add()([x1, x2, x3, x4])
    y = L.Conv2D(num_filters, 1, padding='same')(y)

    return y


def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s1 = cbam_block(s1, ratio=8)

    s2, p2 = encoder_block(p1, 128)
    s2 = cbam_block(s2, ratio=8)

    s3, p3 = encoder_block(p2, 256)
    s3 = cbam_block(s3, ratio=8)

    s4, p4 = encoder_block(p3, 512)
    s4 = cbam_block(s4, ratio=8)

    """ViT in the bridge"""
    # Bottleneck with Vision Transformer (ViT)
    # ViT Parameters
    patch_size = 2
    num_patches = (p4.shape[1] // patch_size) * (p4.shape[2] // patch_size)
    embedding_dim = 512
    num_heads = 12
    mlp_dim = 2048
    num_layers = 12
    dropout_rate = 0.1

    # Extract patches from the feature map
    patches = Patches(patch_size)(p4)

    # Encode patches with positional embeddings
    encoded_patches = PatchEncoder(num_patches, embedding_dim)(patches)

    # Apply Transformer Encoder layers
    for _ in range(num_layers):
        encoded_patches = transformer_encoder(encoded_patches, num_heads, mlp_dim, dropout_rate)

    # Reshape the encoded patches back to feature map
    x = layers.Dense(embedding_dim)(encoded_patches)
    patch_dims = p4.shape[1] // patch_size
    x = layers.Reshape((patch_dims, patch_dims, embedding_dim))(x)   ##(8, 8, 512)

    # Upsample to match the size before bottleneck
    x = layers.UpSampling2D(size=(2, 2))(x)          # Shape: (16, 16, 512)

    w = ASPP(p4)                                     # Shape: (16, 16, 512)

    x = Concatenate()([x, w])
    x = cbam_block(x, ratio=8)

    d1 = decoder_block(x, s4, 512)
    d1 = cbam_block(d1, ratio=8)

    d2 = decoder_block(d1, s3, 256)
    d2 = cbam_block(d2, ratio=8)

    d3 = decoder_block(d2, s2, 128)
    d3 = cbam_block(d3, ratio=8)

    d4 = decoder_block(d3, s1, 64)
    d4 = cbam_block(d4, ratio=8)


    y1 = aspp_decoder(d4, 1)
    y1 = L.Activation("sigmoid")(y1)

    y2 = aspp_decoder(d3, 1)
    y2 = L.UpSampling2D(size=(2, 2), interpolation='bilinear')(y2)
    y2 = L.Activation("sigmoid")(y2)

    y3 = aspp_decoder(d2, 1)
    y3 = L.UpSampling2D(size=(4, 4), interpolation='bilinear')(y3)
    y3 = L.Activation("sigmoid")(y3)

    y4 = aspp_decoder(d1, 1)
    y4 = L.UpSampling2D(size=(8, 8), interpolation='bilinear')(y4)
    y4 = L.Activation("sigmoid")(y4)

    y5 = aspp_decoder(x, 1)
    y5 = L.UpSampling2D(size=(16, 16), interpolation='bilinear')(y5)
    y5 = L.Activation("sigmoid")(y5)


    y0 = L.Concatenate()([y1, y2, y3, y4, y5])
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(y0)

    model = Model(inputs, outputs, name="UNET")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_unet(input_shape, 4)
    model.summary()