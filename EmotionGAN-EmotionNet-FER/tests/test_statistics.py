import json
from pathlib import Path
import numpy as np
import pandas as pd
from emotionfer.experiments import summarize_runs


def _seed(root, seed, proposed, comparator=None):
    directory = root / f"seed_{seed}"
    (directory / "evaluation").mkdir(parents=True, exist_ok=True)
    (directory / "evaluation" / "metrics.json").write_text(json.dumps({
        "accuracy": proposed, "f1_macro": proposed*0.99,
        "precision_macro": proposed*0.98, "recall_macro": proposed*0.97}))
    if comparator is not None:
        p = directory / "experiments" / "B3_emotionnet_geom_aug" / "evaluation"
        p.mkdir(parents=True, exist_ok=True)
        (p / "metrics.json").write_text(json.dumps({
            "accuracy": comparator, "f1_macro": comparator*0.99,
            "precision_macro": comparator*0.98, "recall_macro": comparator*0.97}))


def test_one_seed_summary_does_not_write_nan_json(tmp_path):
    _seed(tmp_path, 42, 0.5)
    result = summarize_runs(tmp_path)
    assert len(result) == 1
    assert result[0]["n_seeds"] == 1
    assert result[0]["accuracy_sd"] is None
    assert (tmp_path / "statistical_summary.json").exists()


def test_paired_wilcoxon_and_ci(tmp_path):
    for seed, p in zip([42,123,2024,2025,3407], [.71,.73,.70,.72,.74]):
        _seed(tmp_path, seed, p, p-0.02)
    summary = summarize_runs(tmp_path)
    assert len(summary) == 2
    assert all(row["n_seeds"] == 5 for row in summary)
    assert (tmp_path / "paired_wilcoxon_holm.csv").is_file()
    tests = pd.read_csv(tmp_path / "paired_wilcoxon_holm.csv")
    assert len(tests) == 1
    assert tests.iloc[0]["mean_accuracy_difference"] > 0
