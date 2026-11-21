"""Seven separate GAN instances; train partition is the only available input."""
from __future__ import annotations
import csv
import shutil
import warnings
from pathlib import Path
import numpy as np
from . import CLASSES
from .config import read_json, write_json
from .data import load_prepared
from .models import build_generator, build_discriminator
from .utils import set_seed


def _train_one_class(images_uint8, cls, run_dir, config, resume=True):
    import tensorflow as tf
    params = config["gan"]
    seed = int(config["seed"]) + 100 * (cls + 1)
    set_seed(seed, deterministic=bool(config.get("deterministic", False)))
    class_folder = Path(run_dir) / "gans" / CLASSES[cls].lower()
    class_folder.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    if not resume and (class_folder / "checkpoints").exists():
        shutil.rmtree(class_folder / "checkpoints")
    g = build_generator(params["latent_dim"])
    d = build_discriminator()
    g_opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"],
                                     beta_1=params["beta_1"], beta_2=params["beta_2"])
    d_opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"],
                                     beta_1=params["beta_1"], beta_2=params["beta_2"])
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    completed = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epochs")
    ckpt = tf.train.Checkpoint(generator=g, discriminator=d,
                               generator_optimizer=g_opt, discriminator_optimizer=d_opt,
                               completed_epochs=completed)
    manager = tf.train.CheckpointManager(ckpt, str(class_folder / "checkpoints"), max_to_keep=2)
    if resume and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"[{CLASSES[cls]}] Resuming checkpoint at epoch {int(completed.numpy())}", flush=True)
    elif not resume and int(completed.numpy()) == 0:
        # A fresh restart overwrites the final weights and appends a new history file.
        pass
    real = ((images_uint8.astype(np.float32) / 255.) * 2. - 1.).astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices(real).shuffle(
        min(len(real), 4096), seed=seed, reshuffle_each_iteration=True
    ).batch(int(params["batch_size"]), drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    log_path = class_folder / "gan_history.csv"
    if not resume or not log_path.exists() or int(completed.numpy()) == 0:
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow(["epoch", "generator_loss", "discriminator_loss"])
    for epoch in range(int(completed.numpy()), int(params["epochs"])):
        g_losses, d_losses = [], []
        for real_batch in train_ds:
            n = tf.shape(real_batch)[0]
            z = tf.random.normal([n, int(params["latent_dim"])])
            with tf.GradientTape() as tape:
                fake = tf.stop_gradient(g(z, training=True))
                real_logit = d(real_batch, training=True)
                fake_logit = d(fake, training=True)
                d_loss = (cross_entropy(tf.ones_like(real_logit), real_logit) +
                          cross_entropy(tf.zeros_like(fake_logit), fake_logit))
            d_grad = tape.gradient(d_loss, d.trainable_variables)
            d_opt.apply_gradients(zip(d_grad, d.trainable_variables))
            with tf.GradientTape() as tape:
                fresh_fake = g(tf.random.normal([n, int(params["latent_dim"])]), training=True)
                # Freeze discriminator batch-normalization moving stats on the G step.
                generated_logit = d(fresh_fake, training=False)
                g_loss = cross_entropy(tf.ones_like(generated_logit), generated_logit)
            g_grad = tape.gradient(g_loss, g.trainable_variables)
            g_opt.apply_gradients(zip(g_grad, g.trainable_variables))
            d_losses.append(float(d_loss.numpy()))
            g_losses.append(float(g_loss.numpy()))
        g_avg, d_avg = float(np.mean(g_losses)), float(np.mean(d_losses))
        completed.assign(epoch + 1)
        with log_path.open("a", newline="") as file:
            csv.writer(file).writerow([epoch + 1, g_avg, d_avg])
        print(f"[{CLASSES[cls]}] epoch {epoch + 1}/{params['epochs']}: "
              f"G={g_avg:.4f} D={d_avg:.4f}", flush=True)
        if int(completed.numpy()) % int(params["checkpoint_every"]) == 0:
            manager.save(checkpoint_number=int(completed.numpy()))
    g.save_weights(class_folder / "generator.weights.h5")
    manager.save(checkpoint_number=int(completed.numpy()))
    write_json(class_folder / "gan_summary.json", {
        "emotion": CLASSES[cls], "real_train_count": len(real),
        "epochs_completed": int(completed.numpy()), "generator_params": g.count_params(),
        "discriminator_params": d.count_params(), "generator_weights": "generator.weights.h5"})
    tf.keras.backend.clear_session()


def train_gans(run_dir, config, resume=True):
    data = load_prepared(run_dir)
    tr_images, tr_labels = data["train_images"], data["train_labels"]
    for cls in range(7):
        _train_one_class(tr_images[tr_labels == cls], cls, run_dir, config, resume)
    from .reporting import plot_gan_losses
    plot_gan_losses(run_dir, config)


def synthetic_budget(labels):
    counts = np.bincount(labels, minlength=7).astype(int)
    return np.maximum(0, counts.max() - counts).astype(int)


def _assess_generated(images, source_images, limit, seed):
    """Unsupervised diagnostics, *not* proof of correct synthetic expression labels."""
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(seed)
    def sample(x):
        ids = rng.choice(len(x), size=min(int(limit), len(x)), replace=False)
        return np.asarray(x[ids], dtype=np.float32).reshape(len(ids), -1)
    synth = sample(images)
    real = sample(source_images.astype(np.float32) / 127.5 - 1.)
    dist = pairwise_distances(synth, real, metric="euclidean") / np.sqrt(48 * 48)
    dist2 = pairwise_distances(synth, synth, metric="euclidean") / np.sqrt(48 * 48)
    np.fill_diagonal(dist2, np.inf)
    return {
        "synthetic_pixel_std": float(np.std(images)),
        "mean_closest_real_pixel_distance": float(np.mean(np.min(dist, axis=1))),
        "mean_closest_synthetic_pixel_distance": float(np.mean(np.min(dist2, axis=1))) if len(synth)>1 else None,
        "sampled_for_diagnostics": int(len(synth)),
        "possible_mode_collapse_warning": bool(len(synth) > 1 and np.mean(np.min(dist2, axis=1)) < 0.025),
        "note": "Diagnostics detect low variation/near matches, not expression fidelity or identity memorization conclusively."
    }


def synthesize(run_dir, config):
    import tensorflow as tf
    run_dir = Path(run_dir)
    data = load_prepared(run_dir)
    tr_images, tr_labels = data["train_images"], data["train_labels"]
    budgets = synthetic_budget(tr_labels)
    synth_images, synth_labels = [], []
    diag = {}
    for cls, count in enumerate(budgets):
        print(f"[{CLASSES[cls]}] target synthetic count: {count}", flush=True)
        weights = run_dir / "gans" / CLASSES[cls].lower() / "generator.weights.h5"
        if not weights.exists():
            raise FileNotFoundError(f"Missing {weights}. Run train-gans first.")
        if count == 0:
            diag[CLASSES[cls]] = {"real": int(np.sum(tr_labels == cls)), "synthetic": 0}
            continue
        set_seed(int(config["seed"]) + 10000 + cls)
        tf.keras.backend.clear_session()
        model = build_generator(config["gan"]["latent_dim"])
        model.load_weights(weights)
        batches = []
        for start in range(0, int(count), int(config["gan"]["generation_batch_size"])):
            batch_size = min(int(config["gan"]["generation_batch_size"]), count - start)
            z = tf.random.normal((batch_size, int(config["gan"]["latent_dim"])))
            batch = model(z, training=False).numpy()
            if not np.isfinite(batch).all():
                raise RuntimeError(f"Non-finite generated samples for {CLASSES[cls]}")
            batches.append(np.asarray(batch, dtype=np.float16))
        generated = np.concatenate(batches, axis=0)
        if generated.shape != (int(count), 48, 48, 1):
            raise RuntimeError(f"Unexpected generator output shape {generated.shape}")
        if float(np.max(np.abs(generated))) > 1.001:
            raise RuntimeError("Generator samples not in [-1,1]")
        synth_images.append(generated)
        synth_labels.append(np.repeat(cls, int(count)).astype(np.int64))
        diag[CLASSES[cls]] = {
            "real": int(np.sum(tr_labels == cls)), "synthetic": int(count),
            **_assess_generated(generated, tr_images[tr_labels == cls],
                                config["reports"]["image_diagnostics_limit"], int(config["seed"]) + cls)}
        if diag[CLASSES[cls]]["possible_mode_collapse_warning"]:
            warnings.warn(f"Possible low-diversity GAN samples for {CLASSES[cls]}; "
                          "inspect synthetic_diagnostics.json before scientific interpretation", stacklevel=2)
    if synth_images:
        images = np.concatenate(synth_images)
        labels = np.concatenate(synth_labels)
    else:
        images = np.empty((0, 48, 48, 1), dtype=np.float16)
        labels = np.empty((0,), dtype=np.int64)
    np.savez_compressed(run_dir / "synthetic.npz", images=images, labels=labels)
    write_json(run_dir / "synthetic_diagnostics.json", {
        "per_class": diag, "counts": {CLASSES[i]: int(x) for i,x in enumerate(budgets)},
        "training_only": True, "synthetic_labels": "Inherited from class-wise generator; not independently verified"})
    from .reporting import plot_class_distributions
    plot_class_distributions(run_dir, tr_labels, config, synthetic_labels=labels)
    print(f"Created {len(images)} generated training images", flush=True)
    return {CLASSES[i]: int(v) for i, v in enumerate(budgets)}
