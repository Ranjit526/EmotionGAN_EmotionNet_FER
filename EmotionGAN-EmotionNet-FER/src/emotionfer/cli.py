"""Command-line interface and end-to-end orchestration."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from .config import load_config, save_config, read_json, write_json
from .data import demo_csv, save_prepared
from .utils import versions


def _config(args):
    if getattr(args, "config", None):
        return load_config(args.config)
    locked = Path(args.run_dir) / "config.lock.yaml"
    if not locked.exists():
        raise FileNotFoundError(f"Run configuration unavailable: {locked}. Specify --config.")
    return load_config(locked)


def _prepare(args, config, *, overwrite=False):
    run_dir = Path(args.run_dir)
    out = run_dir / "partitions.npz"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists. Use --overwrite to regenerate partitions, "
                              "or continue with train-gans.")
    if out.exists() and overwrite and any((run_dir / name).exists() for name in ("gans", "synthetic.npz", "classifier", "evaluation", "experiments")):
        raise RuntimeError("Refusing to replace splits of an existing trained run. Use a fresh --run-dir.")
    saved = save_prepared(args.data, run_dir, config)
    save_config(config, run_dir / "config.lock.yaml")
    print(f"Prepared partitions: train={saved['partitions']['train']['count']}, "
          f"val={saved['partitions']['validation']['count']}, "
          f"test={saved['partitions']['test']['count']}")
    return saved


def _run_stages(args, config):
    from .gan import train_gans, synthesize
    from .trainer import train_classifier
    from .evaluation import evaluate_model
    if not (Path(args.run_dir) / "partitions.npz").exists():
        _prepare(args, config)
    elif not (Path(args.run_dir) / "config.lock.yaml").exists():
        raise RuntimeError("Existing partitions without locked configuration; refuse unsafe continuation")
    else:
        existing = load_config(Path(args.run_dir) / "config.lock.yaml")
        if existing != config:
            raise RuntimeError("Requested config differs from locked run config; use a fresh --run-dir")
    train_gans(args.run_dir, config, resume=True)
    synthesize(args.run_dir, config)
    model = train_classifier(args.run_dir, config)
    result = evaluate_model(args.run_dir, config, model_file=model)
    print("FINAL HELD-OUT METRICS:", {key: round(result[key], 4)
                                      for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")})
    if getattr(args, "comparisons", None) is not None:
        from .experiments import run_comparisons
        run_comparisons(args.run_dir, config, suite=args.comparisons,
                        update_matched=getattr(args, "update_matched", False))
    return result


def make_parser():
    parser = argparse.ArgumentParser(prog="emotionfer",
                                     description="Leakage-controlled EmotionGAN + EmotionNet-CNN FER-2013")
    sub = parser.add_subparsers(dest="command", required=True)
    def config_run(p):
        p.add_argument("--run-dir", required=True, help="Run directory containing prepared splits")
        p.add_argument("--config", default=None, help="YAML (default: run-dir/config.lock.yaml)")
    demo = sub.add_parser("create-demo", help="Random synthetic CSV for CODE smoke tests, not research")
    demo.add_argument("--out", default="data/demo_fer2013.csv")
    demo.add_argument("--per-class", type=int, default=32)
    prep = sub.add_parser("prepare", help="Load dataset; lock train/val/test partition")
    config_run(prep)
    prep.add_argument("--data", required=True, help="FER-2013 CSV or labeled image directory")
    prep.add_argument("--overwrite", action="store_true")
    for cmd in ("train-gans", "synthesize", "train", "evaluate"):
        p = sub.add_parser(cmd)
        config_run(p)
        if cmd == "train-gans":
            p.add_argument("--fresh", action="store_true", help="Do not resume existing GAN checkpoints")
    all_cmd = sub.add_parser("run", help="End-to-end prepare, GAN, synthetic, CNN, evaluation")
    config_run(all_cmd)
    all_cmd.add_argument("--data", required=True)
    all_cmd.add_argument("--comparisons", choices=["all", "baselines", "ablations"], default=None)
    all_cmd.add_argument("--update-matched", action="store_true")
    cmp = sub.add_parser("compare", help="Controlled baseline and/or ablation suite")
    config_run(cmp)
    cmp.add_argument("--suite", choices=["all", "baselines", "ablations"], default="all")
    cmp.add_argument("--update-matched", action="store_true")
    pred = sub.add_parser("predict", help="Predict from already cropped face image")
    pred.add_argument("--model", required=True)
    pred.add_argument("--image", required=True)
    pred.add_argument("--out", default=None, help="Optional JSON output path")
    multi = sub.add_parser("multi-seed", help="Run several seeds sequentially, generate statistics")
    multi.add_argument("--data", required=True)
    multi.add_argument("--config", required=True)
    multi.add_argument("--root", required=True)
    multi.add_argument("--seeds", default="42,123,2024,2025,3407")
    multi.add_argument("--comparisons", choices=["all", "baselines", "ablations"], default=None)
    summary = sub.add_parser("summarize", help="Aggregate seed_* outputs and paired statistics")
    summary.add_argument("--root", required=True)
    sub.add_parser("environment", help="Show versions and GPU availability")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "create-demo":
        print(f"Created non-FER random code-test CSV at {demo_csv(args.out, args.per_class)}")
        return 0
    if args.command == "environment":
        print(versions())
        return 0
    if args.command == "predict":
        from .inference import predict_face
        result = predict_face(args.model, args.image)
        print(result)
        if args.out:
            write_json(args.out, result)
        return 0
    if args.command == "summarize":
        from .experiments import summarize_runs
        print(summarize_runs(args.root))
        return 0
    if args.command == "multi-seed":
        from .experiments import summarize_runs
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("Provide unique comma-separated numeric seeds")
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            conf = deepcopy(load_config(args.config))
            conf["seed"] = seed
            run_dir = root / f"seed_{seed}"
            subargs = argparse.Namespace(data=args.data, run_dir=str(run_dir), comparisons=args.comparisons)
            print(f"\n========== SEED {seed} ==========", flush=True)
            _run_stages(subargs, conf)
        summarize_runs(root)
        return 0
    config = _config(args)
    if args.command == "prepare":
        _prepare(args, config, overwrite=args.overwrite)
    elif args.command == "train-gans":
        from .gan import train_gans
        train_gans(args.run_dir, config, resume=not args.fresh)
    elif args.command == "synthesize":
        from .gan import synthesize
        synthesize(args.run_dir, config)
    elif args.command == "train":
        from .trainer import train_classifier
        print(f"Saved final checkpoint: {train_classifier(args.run_dir, config)}")
    elif args.command == "evaluate":
        from .evaluation import evaluate_model
        print(evaluate_model(args.run_dir, config))
    elif args.command == "compare":
        from .experiments import run_comparisons
        print(run_comparisons(args.run_dir, config, suite=args.suite,
                              update_matched=args.update_matched))
    elif args.command == "run":
        _run_stages(args, config)
    else:
        raise ValueError(f"Unknown command {args.command}")
    return 0
