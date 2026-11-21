import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="Install TensorFlow to verify neural network forward/backward passes")
from emotionfer.models import build_generator, build_discriminator, build_emotionnet
from emotionfer.augment import augment_batch


def test_model_shapes_and_parameter_count():
    g = build_generator()
    d = build_discriminator()
    c = build_emotionnet()
    assert g.output_shape == (None,48,48,1)
    assert d.output_shape == (None,1)
    assert c.output_shape == (None,7)
    assert sum(int(np.prod(w.shape)) for w in c.trainable_weights) == 146759
    assert sum(int(np.prod(w.shape)) for w in c.non_trainable_weights) == 512
    z = tf.random.normal((3,100))
    images = g(z, training=False)
    assert images.shape == (3,48,48,1)
    assert tf.reduce_max(tf.abs(images)).numpy() <= 1.
    probs = c(images, training=False).numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-5)


def test_identity_augmentation():
    x = tf.random.uniform((4,48,48,1), -1., 1.)
    cfg = {"max_rotation_deg": 0., "max_translate_fraction": 0., "max_shear_deg": 0.,
           "min_zoom": 1., "max_zoom": 1., "flip_probability": 0.}
    y = augment_batch(x, cfg)
    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-5)


def test_gradients_exist_for_generator_and_discriminator():
    g, d = build_generator(), build_discriminator()
    z = tf.random.normal((2,100))
    with tf.GradientTape() as tape:
        out = d(g(z, training=True), training=False)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(out), logits=out))
    gradients = tape.gradient(loss, g.trainable_variables)
    assert any(x is not None and np.any(np.abs(x.numpy()) > 0) for x in gradients)
