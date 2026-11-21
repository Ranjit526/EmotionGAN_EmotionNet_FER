"""Train-only geometric transforms using TensorFlow's built-in projective op.

One independent affine matrix per image, backwards mapping from output to input.
Angles and displacement ranges are explicitly configurable. Evaluation is unaugmented.
"""
from __future__ import annotations


def augment_batch(images, config):
    import tensorflow as tf
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    c = tf.cast(tf.shape(x)[0], tf.int32)
    h = tf.cast(tf.shape(x)[1], tf.float32)
    w = tf.cast(tf.shape(x)[2], tf.float32)
    pi = tf.constant(3.141592653589793, tf.float32)
    def unif(low, high):
        return tf.random.uniform((c,), float(low), float(high), dtype=tf.float32)
    a = unif(-config["max_rotation_deg"], config["max_rotation_deg"]) * pi / 180.0
    shear = unif(-config["max_shear_deg"], config["max_shear_deg"]) * pi / 180.0
    zoom = unif(config["min_zoom"], config["max_zoom"])
    flip = tf.where(unif(0., 1.) < float(config["flip_probability"]), -1., 1.)
    cos, sin = tf.cos(a), tf.sin(a)
    # Forward transform F = rotation @ shear @ (zoom * horizontal_flip)
    # A = [[cos*z*flip, (cos*tan(shear)-sin)*z],
    #      [sin*z*flip, (sin*tan(shear)+cos)*z]]
    f00 = cos * zoom * flip
    f01 = (cos * tf.tan(shear) - sin) * zoom
    f10 = sin * zoom * flip
    f11 = (sin * tf.tan(shear) + cos) * zoom
    det = f00 * f11 - f01 * f10
    inv00, inv01 = f11 / det, -f01 / det
    inv10, inv11 = -f10 / det, f00 / det
    tx = unif(-config["max_translate_fraction"], config["max_translate_fraction"]) * w
    ty = unif(-config["max_translate_fraction"], config["max_translate_fraction"]) * h
    cx, cy = (w - 1.) / 2., (h - 1.) / 2.
    # x_input = A^-1 @ (x_output - center - shift) + center
    b0 = cx - inv00 * (cx + tx) - inv01 * (cy + ty)
    b1 = cy - inv10 * (cx + tx) - inv11 * (cy + ty)
    zeros = tf.zeros_like(b0)
    transforms = tf.stack([inv00, inv01, b0, inv10, inv11, b1, zeros, zeros], axis=1)
    augmented = tf.raw_ops.ImageProjectiveTransformV3(
        images=x, transforms=transforms, output_shape=tf.shape(x)[1:3],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)
    augmented.set_shape(x.shape)
    return augmented
