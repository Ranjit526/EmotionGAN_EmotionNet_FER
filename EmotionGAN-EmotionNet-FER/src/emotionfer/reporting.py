"""Publication-oriented graphics generated only from actual recorded data.
No sample faces are exported by default; all figures are non-identifying statistics.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from . import CLASSES


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path, cfg):
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg["reports"].get("dpi", 600)), bbox_inches="tight")
    plt.close(fig)


def plot_class_distributions(run_dir, train_labels, cfg, synthetic_labels=None):
    plt = _plt()
    a = np.bincount(train_labels, minlength=7)
    b = np.bincount(synthetic_labels, minlength=7) if synthetic_labels is not None else np.zeros(7, int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(7)
    ax.bar(x, a, label="Real training", alpha=0.85)
    if synthetic_labels is not None:
        ax.bar(x, b, bottom=a, label="GAN synthetic", alpha=0.75)
    ax.set_xticks(x, CLASSES, rotation=25)
    ax.set_xlabel("Emotion class")
    ax.set_ylabel("Number of training images")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _save(fig, Path(run_dir) / "figures" / ("class_balance.png" if synthetic_labels is not None
                                                  else "class_distribution.png"), cfg)


def plot_eda(run_dir, images_uint8, labels, cfg):
    plt = _plt()
    rng = np.random.default_rng(int(cfg["seed"]))
    chosen = np.concatenate([rng.choice(np.flatnonzero(labels == c),
                                        size=min(10, int(np.sum(labels == c))), replace=False)
                             for c in range(7)])
    subs = images_uint8[chosen, :, :, 0].astype(np.float32)
    ys = labels[chosen]
    overall = subs.mean(axis=0)
    brows = slice(11, 22)
    mouth = slice(30, 43)
    crop_cols = slice(8, 40)
    deviations = []
    avg_intensities, contrasts, histograms = [], [], []
    for c in range(7):
        pics = subs[ys == c]
        avg = pics.mean(axis=0)
        delta = np.abs(avg - overall)
        deviations.append([delta[brows, crop_cols].mean(), delta[mouth, crop_cols].mean()])
        avg_intensities.append(pics.mean(axis=(1, 2)).mean())
        contrasts.append(pics.std(axis=(1, 2)).mean())
        histograms.append(np.histogram(pics, bins=16, range=(0, 256), density=True)[0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    a, b, c, d = axes.flatten()
    x = np.arange(7)
    a.bar(x, np.bincount(labels, minlength=7))
    a.set_xticks(x, CLASSES, rotation=25)
    a.set_xlabel("Emotion class")
    a.set_ylabel("Images in training partition")
    deviations = np.asarray(deviations)
    b.bar(x - 0.17, deviations[:, 0], width=0.34, label="Eyebrow region")
    b.bar(x + 0.17, deviations[:, 1], width=0.34, label="Mouth region")
    b.set_xticks(x, CLASSES, rotation=25)
    b.set_xlabel("Emotion class")
    b.set_ylabel("Absolute deviation from overall mean")
    b.legend(fontsize=8, frameon=False)
    c.bar(x - 0.17, avg_intensities, width=0.34, label="Mean intensity")
    c.bar(x + 0.17, contrasts, width=0.34, label="Within-image SD")
    c.set_xticks(x, CLASSES, rotation=25)
    c.set_xlabel("Emotion class")
    c.set_ylabel("Intensity (0–255)")
    c.legend(fontsize=8, frameon=False)
    xs = np.arange(8, 256, 16)
    for i, h in enumerate(histograms):
        d.plot(xs, h, label=CLASSES[i], linewidth=1.2)
    d.set_xlabel("Pixel intensity (0–255)")
    d.set_ylabel("Probability density")
    d.legend(fontsize=7, ncol=2, frameon=False)
    for panel, ax in zip("(a)(b)(c)(d)".replace(")(", ")|(").split("|"), axes.flatten()):
        ax.text(0.01, 0.98, panel, transform=ax.transAxes, va="top", fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    _save(fig, Path(run_dir) / "figures" / "eda_four_panels.png", cfg)


def plot_gan_losses(run_dir, cfg):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    found = False
    for cl in CLASSES:
        p = Path(run_dir) / "gans" / cl.lower() / "gan_history.csv"
        if not p.exists():
            continue
        with p.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        found = True
        epochs = [int(x["epoch"]) for x in rows]
        axs[0].plot(epochs, [float(x["generator_loss"]) for x in rows], label=cl, linewidth=1.1)
        axs[1].plot(epochs, [float(x["discriminator_loss"]) for x in rows], label=cl, linewidth=1.1)
    if found:
        for ax, label in zip(axs, ["Generator loss", "Discriminator loss"]):
            ax.set_xlabel("GAN training epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axs[1].legend(fontsize=8, frameon=False, ncol=2)
        fig.tight_layout()
        _save(fig, Path(run_dir) / "figures" / "gan_losses.png", cfg)
    else:
        plt.close(fig)


def plot_cnn_history(history_csv, output_path, cfg):
    import pandas as pd
    plt = _plt()
    frame = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, ax in (("accuracy", axes[0]), ("loss", axes[1])):
        for prefix in ("", "val_"):
            key = prefix + name
            if key in frame:
                ax.plot(frame["epoch"] + 1, frame[key], label="Validation" if prefix else "Training")
        ax.set_xlabel("CNN epoch")
        ax.set_ylabel(name.capitalize())
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_path, cfg)


def plot_evaluation(output_dir, truth, predictions, probabilities, metrics, cfg):
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    plt = _plt()
    output_dir = Path(output_dir)
    cm = confusion_matrix(truth, predictions, labels=np.arange(7))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(7), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("Actual emotion")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > (cm.max() / 2) else "black")
    fig.tight_layout()
    _save(fig, output_dir / "confusion_matrix.png", cfg)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    for i, (key, name) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))):
        values = [metrics["per_class"][cl][key] for cl in CLASSES]
        ax.bar(x + (i - 1) * 0.24, values, width=0.23, label=name)
    ax.set_xticks(x, CLASSES, rotation=25)
    ax.set_xlabel("Emotion class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir / "per_class_scores.png", cfg)
    fig, ax = plt.subplots(figsize=(7, 6))
    auc_per_class = {}
    for i, name in enumerate(CLASSES):
        y_binary = (truth == i).astype(int)
        if np.unique(y_binary).size < 2:
            auc_per_class[name] = None
            continue
        fpr, tpr, _ = roc_curve(y_binary, probabilities[:, i])
        score = auc(fpr, tpr)
        auc_per_class[name] = float(score)
        ax.plot(fpr, tpr, label=f"{name}: {score:.3f}", linewidth=1.3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir / "roc_one_vs_rest.png", cfg)
    return auc_per_class


def plot_experiment_scores(metrics, path, cfg):
    plt = _plt()
    labels = [row["name"] for row in metrics]
    scores = [row["accuracy"] for row in metrics]
    fig, ax = plt.subplots(figsize=(max(8, len(scores) * 1.2), 4.5))
    ax.bar(range(len(scores)), scores)
    ax.set_xticks(range(len(scores)), labels, rotation=35)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path, cfg)
