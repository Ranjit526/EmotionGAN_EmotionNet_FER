"""Controlled baselines / ablations, and paired multi-seed summaries."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon, rankdata
from .config import read_json, write_json
from .trainer import train_classifier
from .evaluation import evaluate_model
from .reporting import plot_experiment_scores


BASELINES = {
    "B1_basic_no_aug": {"variant": "basic", "use_synthetic": False, "augmentation": False},
    "B2_basic_geom_aug": {"variant": "basic", "use_synthetic": False, "augmentation": True},
    "B3_emotionnet_geom_aug": {"variant": "emotionnet", "use_synthetic": False, "augmentation": True},
    "B4_emotionnet_class_weight": {"variant": "emotionnet", "use_synthetic": False,
                                    "augmentation": True, "weighting": True},
    "B5_emotionnet_oversample": {"variant": "emotionnet", "use_synthetic": False,
                                  "augmentation": True, "oversample": True},
}

ABLATIONS = {
    "A1_no_residual_no_aug": {"variant": "basic", "use_synthetic": False, "augmentation": False},
    "A2_residual_max_pool": {"variant": "residual_max", "use_synthetic": False,
                              "augmentation": False},
    "A3_residual_GAP": {"variant": "emotionnet", "use_synthetic": False, "augmentation": False},
    "A4_geom_without_GAN": {"variant": "emotionnet", "use_synthetic": False, "augmentation": True},
}


def run_comparisons(run_dir, config, *, suite="all", update_matched=False):
    run_dir = Path(run_dir)
    if suite not in ("all", "baselines", "ablations"):
        raise ValueError("suite must be all, baselines, or ablations")
    if not (run_dir / "classifier" / "classifier_final.keras").exists():
        raise FileNotFoundError("Complete proposed model first: run train")
    if not (run_dir / "evaluation" / "metrics.json").exists():
        evaluate_model(run_dir, config)
    cases = {}
    if suite in ("all", "baselines"):
        cases.update(BASELINES)
    if suite in ("all", "ablations"):
        cases.update(ABLATIONS)
    completed = []
    for name, opts in cases.items():
        out = run_dir / "experiments" / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\nTraining experiment {name}", flush=True)
        model_path = train_classifier(run_dir, config, name=name, output_dir=out,
                                       update_matched=update_matched, **opts)
        result = evaluate_model(run_dir, config, model_file=model_path, output_dir=out / "evaluation")
        completed.append({"name": name, "accuracy": result["accuracy"],
                          "f1_macro": result["f1_macro"],
                          "paper_harmonic_macro_f1": result["paper_harmonic_macro_f1"]})
    reference = read_json(run_dir / "evaluation" / "metrics.json")
    completed.append({"name": "Proposed_EmotionGAN_EmotionNet", "accuracy": reference["accuracy"],
                      "f1_macro": reference["f1_macro"],
                      "paper_harmonic_macro_f1": reference["paper_harmonic_macro_f1"]})
    pd.DataFrame(completed).to_csv(run_dir / "experiments" / f"{suite}_summary.csv", index=False)
    plot_experiment_scores(completed, run_dir / "experiments" / f"{suite}_comparison.png", config)
    write_json(run_dir / "experiments" / f"{suite}_meta.json", {
        "suite": suite, "update_matched": bool(update_matched),
        "important": "Implementation-defined baselines, not paper's unnamed five reproduced methods.",
        "results": completed})
    return completed


def summarize_runs(root):
    """Aggregate seed directories, 95% t CIs, and paired Holm-adjusted Wilcoxon."""
    root = Path(root)
    rows = []
    for directory in sorted(root.glob("seed_*")):
        if not directory.is_dir():
            continue
        seed = int(directory.name.split("_", 1)[1])
        paths = [("Proposed", directory / "evaluation" / "metrics.json")]
        paths += [(name, p) for name in list(BASELINES) + list(ABLATIONS)
                  if (p := directory / "experiments" / name / "evaluation" / "metrics.json").exists()]
        for experiment, path in paths:
            if not path.exists():
                continue
            metric = read_json(path)
            rows.append({"seed": seed, "model": experiment,
                         "accuracy": metric["accuracy"], "f1_macro": metric["f1_macro"],
                         "precision_macro": metric["precision_macro"],
                         "recall_macro": metric["recall_macro"]})
    if not rows:
        raise FileNotFoundError("No seed_*/evaluation/metrics.json found")
    raw = pd.DataFrame(rows)
    raw.to_csv(root / "per_seed_metrics.csv", index=False)
    summary_rows = []
    for model, group in raw.groupby("model", sort=True):
        row = {"model": model, "n_seeds": len(group)}
        for metric in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            values = group[metric].to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values)
            sd = np.std(values, ddof=1) if n > 1 else float("nan")
            half_width = t.ppf(0.975, n-1)*sd/np.sqrt(n) if n > 1 else float("nan")
            row[metric + "_mean"] = float(mean)
            row[metric + "_sd"] = float(sd) if n > 1 else None
            row[metric + "_ci95_low"] = float(mean-half_width) if n > 1 else None
            row[metric + "_ci95_high"] = float(mean+half_width) if n > 1 else None
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(root / "summary_mean_sd_ci.csv", index=False)
    tests = []
    proposed = raw[raw["model"] == "Proposed"].set_index("seed")
    for model in sorted(set(raw["model"]) - {"Proposed"}):
        base = raw[raw["model"] == model].set_index("seed")
        seeds = sorted(set(base.index) & set(proposed.index))
        if len(seeds) < 5:
            continue
        diff = (proposed.loc[seeds, "accuracy"] - base.loc[seeds, "accuracy"]).to_numpy()
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
        nz = diff[~np.isclose(diff, 0)]
        if len(nz):
            ranks = rankdata(abs(nz), method="average")
            rank_biserial = float((sum(ranks[nz > 0]) - sum(ranks[nz < 0])) / sum(ranks))
        else:
            rank_biserial = 0.0
        sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        cohens_dz = float(np.mean(diff) / sd_diff) if sd_diff > 0 else None
        tests.append({"baseline": model, "n_paired_seeds": len(seeds),
                      "mean_accuracy_difference": float(diff.mean()),
                      "median_accuracy_difference": float(np.median(diff)),
                      "paired_cohens_dz": cohens_dz,
                      "rank_biserial_correlation": rank_biserial,
                      "wilcoxon_p": p})
    # Standard Holm step-down adjustment (family of all comparisons).
    ordered = sorted(range(len(tests)), key=lambda i: tests[i]["wilcoxon_p"])
    prior = 0.0
    for rank, i in enumerate(ordered):
        corrected = min(1., tests[i]["wilcoxon_p"] * (len(tests) - rank))
        prior = max(prior, corrected)
        tests[i]["holm_adjusted_p"] = float(prior)
    if tests:
        pd.DataFrame(tests).to_csv(root / "paired_wilcoxon_holm.csv", index=False)
    write_json(root / "statistical_summary.json", {
        "runs_found": sorted(raw["seed"].unique().tolist()),
        "summary": summary_rows, "paired_tests": tests,
        "ci": "Two-sided Student-t 95% CI across independent random-seed runs; n>1 required",
        "effect_size": "Paired mean/median difference, paired Cohen dz (when defined), and Wilcoxon rank-biserial correlation",
        "test_caveat": "Tests are exploratory for n=5 and cannot validate manuscript results without matching data/protocol."})
    return summary_rows
