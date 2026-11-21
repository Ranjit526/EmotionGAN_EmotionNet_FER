"""TensorFlow/Keras architectures matching paper Figures 2–3 and Table 3.

Imports tensorflow lazily, so dataset inspection and preparation work without it.
"""
from __future__ import annotations
from . import NUM_CLASSES


def build_generator(latent_dim=100):
    import tensorflow as tf
    L = tf.keras.layers
    z = L.Input(shape=(int(latent_dim),), name="noise")
    x = L.Dense(6 * 6 * 256, use_bias=False, name="dense_expand")(z)
    x = L.BatchNormalization(name="gen_bn0")(x)
    x = L.LeakyReLU(negative_slope=0.2)(x)
    x = L.Reshape((6, 6, 256), name="reshape_6x6")(x)
    x = L.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False, name="transpose1")(x)
    x = L.BatchNormalization(name="gen_bn1")(x)
    x = L.LeakyReLU(negative_slope=0.2)(x)
    x = L.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False, name="transpose2")(x)
    x = L.BatchNormalization(name="gen_bn2")(x)
    x = L.LeakyReLU(negative_slope=0.2)(x)
    image = L.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh",
                              name="transpose3_tanh")(x)
    model = tf.keras.Model(z, image, name="EmotionGAN_Generator")
    if model.output_shape != (None, 48, 48, 1):
        raise RuntimeError(f"Incorrect generator shape {model.output_shape}")
    return model


def build_discriminator():
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input(shape=(48, 48, 1), name="face_real_or_fake")
    x = L.Conv2D(64, 4, strides=2, padding="same", name="disc_conv1")(inp)
    x = L.LeakyReLU(negative_slope=0.2)(x)
    x = L.Dropout(0.3)(x)
    x = L.Conv2D(128, 4, strides=2, padding="same", name="disc_conv2")(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(negative_slope=0.2)(x)
    x = L.Dropout(0.3)(x)
    x = L.Flatten()(x)
    # Binary logits, equivalent to a sigmoid followed by binary CE, but numerically stable.
    logits = L.Dense(1, name="realness_logits")(x)
    return tf.keras.Model(inp, logits, name="EmotionGAN_Discriminator")


def build_emotionnet(dropout=0.4, variant="emotionnet"):
    """Return classifier. The 'emotionnet' variant is exactly Table 3 (146759 trainable).

    Other variants deliberately change a single factor for controlled comparisons:
      basic: remove identity shortcut; residual_max: global max instead of average;
      residual_nodrop: use 0 dropout for auxiliary studies.
    """
    import tensorflow as tf
    L = tf.keras.layers
    if variant not in ("emotionnet", "basic", "residual_max", "residual_nodrop"):
        raise ValueError(f"Unknown classifier variant {variant}")
    inp = L.Input((48, 48, 1), name="grayscale_face")
    x = L.Conv2D(64, 3, padding="same", name="conv_64_1")(inp)
    x = L.BatchNormalization(name="bn_1")(x)
    x = L.Activation("relu", name="relu_1")(x)
    skip = x
    x = L.Conv2D(64, 3, padding="same", name="conv_64_2")(x)
    x = L.BatchNormalization(name="bn_2")(x)
    if variant != "basic":
        x = L.Add(name="identity_shortcut")([skip, x])
    x = L.Activation("relu", name="residual_relu")(x)
    x = L.Conv2D(128, 3, padding="same", name="conv_128")(x)
    x = L.BatchNormalization(name="bn_3")(x)
    x = L.Activation("relu", name="relu_3")(x)
    if variant == "residual_max":
        x = L.GlobalMaxPooling2D(name="global_max_pool")(x)
    else:
        x = L.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = L.Dense(256, activation="relu", name="dense_256")(x)
    x = L.Dropout(0.0 if variant == "residual_nodrop" else float(dropout), name="dropout")(x)
    output = L.Dense(NUM_CLASSES, activation="softmax", name="emotion_softmax")(x)
    model = tf.keras.Model(inp, output, name="EmotionNet_CNN" if variant == "emotionnet" else variant)
    if variant == "emotionnet":
        trainable = sum(int(__import__("math").prod(w.shape)) for w in model.trainable_weights)
        if trainable != 146759:
            raise RuntimeError(f"Paper Table 3 mismatch: expected 146759, obtained {trainable}")
    return model
