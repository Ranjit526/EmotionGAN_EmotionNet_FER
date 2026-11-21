"""CNN training. Validation never contains GAN images or contributes to weights."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from . import CLASSES
from .augment import augment_batch
from .data import load_prepared
from .models import build_emotionnet
from .utils import set_seed
from .config import write_json


def _load_examples(run_dir, use_synthetic=False, oversample=False, seed=42):
    data = load_prepared(run_dir)
    train_x = (data["train_images"].astype(np.float32) / 127.5 - 1.).astype(np.float16)
    train_y = data["train_labels"].astype(np.int64)
    if use_synthetic:
        path = Path(run_dir) / "synthetic.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing synthetic images: {path}. Run synthesize first.")
        with np.load(path, allow_pickle=False) as pack:
            sx, sy = pack["images"], pack["labels"]
        train_x = np.concatenate([train_x, sx.astype(np.float16)], axis=0)
        train_y = np.concatenate([train_y, sy.astype(np.int64)], axis=0)
    if oversample:
        rng = np.random.default_rng(seed)
        counts = np.bincount(train_y, minlength=7)
        max_count = counts.max()
        choices = np.concatenate([rng.choice(np.flatnonzero(train_y == c), size=max_count,
                                             replace=True) for c in range(7)])
        train_x = train_x[choices]
        train_y = train_y[choices]
    return data, train_x, train_y


def _tf_data(images, labels, batch_size, training, augmentation=None,
             weights=None, seed=42, repeat=False):
    import tensorflow as tf
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels, weights))
    if training:
        ds = ds.shuffle(min(10000, len(labels)), seed=int(seed), reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(int(batch_size))
    def transform(x, y, w):
        x = tf.cast(x, tf.float32)
        if training and augmentation is not None:
            x = augment_batch(x, augmentation)
        return x, tf.one_hot(tf.cast(y, tf.int32), 7), tf.cast(w, tf.float32)
    ds = ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_classifier(run_dir, config, *, name="proposed", variant="emotionnet",
                     use_synthetic=True, augmentation=True, weighting=False,
                     oversample=False, update_matched=False, output_dir=None):
    import tensorflow as tf
    run_dir = Path(run_dir)
    out = Path(output_dir) if output_dir is not None else run_dir / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed, deterministic=bool(config.get("deterministic", False)))
    tf.keras.backend.clear_session()
    data, train_x, train_y = _load_examples(run_dir, use_synthetic=use_synthetic,
                                            oversample=oversample, seed=seed)
    train_counts = np.bincount(train_y, minlength=7)
    if np.any(train_counts == 0):
        raise ValueError("All seven training classes are mandatory")
    weights = None
    if weighting:
        # Balanced loss without GAN: same total per-class weight, mean sample weight ~1.
        weights = (len(train_y) / (7. * train_counts[train_y])).astype(np.float32)
    cnn_config = config["cnn"]
    batch_size = int(cnn_config["batch_size"])
    steps_per_epoch = None
    if update_matched:
        real_count = len(data["train_labels"])
        balanced_count = 7 * max(np.bincount(data["train_labels"], minlength=7))
        # For real-only comparisons, replay augmented real examples to match updates.
        steps_per_epoch = int(np.ceil(balanced_count / batch_size))
    train_ds = _tf_data(train_x, train_y, batch_size, True,
                        augmentation=cnn_config["augmentation"] if augmentation else None,
                        weights=weights, seed=seed, repeat=update_matched)
    val_x = (data["validation_images"].astype(np.float32) / 127.5 - 1.).astype(np.float16)
    val_y = data["validation_labels"].astype(np.int64)
    val_ds = _tf_data(val_x, val_y, batch_size, False)
    model = build_emotionnet(dropout=cnn_config["dropout"], variant=variant)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_config["learning_rate"],
                                                    beta_1=cnn_config.get("beta_1", 0.9),
                                                    beta_2=cnn_config.get("beta_2", 0.999)),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=float(cnn_config["lr_factor"]),
            patience=int(cnn_config["lr_patience"]), min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out / "cnn_history.csv"), append=False),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best_validation.keras"),
                                           monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=int(cnn_config["epochs"]),
              steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)
    final_path = out / "classifier_final.keras"
    model.save(final_path)
    write_json(out / "training_protocol.json", {
        "experiment": name, "variant": variant, "train_real": len(data["train_labels"]),
        "train_total": len(train_y), "train_per_class": dict(zip(CLASSES, train_counts.tolist())),
        "validation_real": len(val_y), "test_used_during_training": False,
        "gan_augmentation": bool(use_synthetic), "conventional_augmentation": bool(augmentation),
        "class_weighted_loss": bool(weighting), "random_oversampling": bool(oversample),
        "update_matched": bool(update_matched), "steps_per_epoch": steps_per_epoch,
        "classifier_trainable_params": int(sum(__import__("math").prod(w.shape)
                                                for w in model.trainable_weights)),
        "epoch_evaluation_policy": "final epoch"})
    from .reporting import plot_cnn_history
    plot_cnn_history(out / "cnn_history.csv", out / "training_curves.png", config)
    tf.keras.backend.clear_session()
    return final_path
