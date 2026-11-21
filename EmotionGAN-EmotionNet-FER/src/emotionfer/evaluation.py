"""Held-out final evaluation: no training, callbacks, or test-driven selection."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, log_loss, balanced_accuracy_score)
from . import CLASSES
from .config import write_json
from .data import load_prepared
from .reporting import plot_evaluation


def _safe_prob(prob):
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    return clipped / clipped.sum(axis=1, keepdims=True)


def compute_metrics(truth, probabilities):
    truth = np.asarray(truth, dtype=np.int64)
    prob = np.asarray(probabilities, dtype=np.float64)
    if prob.ndim != 2 or prob.shape != (len(truth), 7) or not np.all(np.isfinite(prob)):
        raise ValueError("Probabilities must be finite [N,7]")
    if (prob < -1e-6).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Softmax probabilities invalid")
    prediction = np.argmax(prob, axis=1)
    p, r, f, support = precision_recall_fscore_support(truth, prediction,
                                                      labels=np.arange(7), zero_division=0)
    mean_p, mean_r, mean_f = map(lambda x: float(np.mean(x)), (p, r, f))
    paper_f = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) else 0.0
    result = {
        "n_test": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision_macro": mean_p, "recall_macro": mean_r,
        "f1_macro": mean_f, "paper_harmonic_macro_f1": paper_f,
        "cross_entropy": float(log_loss(truth, _safe_prob(prob),
                                         labels=np.arange(7))),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(support[i])}
                      for i, name in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(7)).tolist(),
        "f1_definition_note": "Standard macro-F1 and paper's harmonic(mean macro-P, mean macro-R) both reported."
    }
    return result


def evaluate_model(run_dir, config, model_file=None, output_dir=None):
    import tensorflow as tf
    run_dir = Path(run_dir)
    model_file = Path(model_file) if model_file is not None else run_dir / "classifier" / "classifier_final.keras"
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_prepared(run_dir)
    test_x = (data["test_images"].astype(np.float32) / 127.5 - 1.)
    test_y = data["test_labels"]
    model = tf.keras.models.load_model(model_file)
    # A single prediction pass; no test feedback to training or checkpoint selection.
    t0 = time.perf_counter()
    pred = np.asarray(model.predict(test_x, batch_size=int(config["cnn"]["batch_size"]), verbose=0))
    duration = time.perf_counter() - t0
    result = compute_metrics(test_y, pred)
    result["prediction_seconds_total"] = float(duration)
    result["throughput_images_per_second_including_first_call"] = float(len(test_y) / max(duration, 1e-8))
    result["tested_checkpoint"] = str(model_file)
    result["protocol"] = config["dataset"]["protocol"]
    result["note"] = "Metrics originate from actual held-out predictions, not paper claims."
    result["roc_auc_one_vs_rest"] = plot_evaluation(output_dir, test_y, np.argmax(pred, axis=1),
                                                    pred, result, config)
    write_json(output_dir / "metrics.json", result)
    frame = pd.DataFrame({"true_label_id": test_y, "true_label": [CLASSES[i] for i in test_y],
                          "predicted_label_id": np.argmax(pred, axis=1),
                          "predicted_label": [CLASSES[i] for i in np.argmax(pred, axis=1)]})
    for i, label in enumerate(CLASSES):
        frame[f"prob_{label.lower()}"] = pred[:, i]
    frame.to_csv(output_dir / "predictions.csv", index=False)
    return result
