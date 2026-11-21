"""Versioned YAML configurations. Pure Python: TensorFlow is imported lazily."""
from __future__ import annotations
import json
from pathlib import Path
import yaml
from . import CLASSES


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    validate_config(config)
    return config


def validate_config(config):
    ds = config.get("dataset", {})
    if ds.get("protocol") not in ("subset", "official"):
        raise ValueError("dataset.protocol must be subset or official")
    if ds.get("protocol") == "subset":
        quota = ds.get("subset_counts", {})
        if set(quota) != set(CLASSES) or any(not isinstance(v, int) or v < 4 for v in quota.values()):
            raise ValueError(f"subset_counts needs positive integer quota >=4 for {CLASSES}")
        if not 0 < float(ds.get("test_fraction", 0.20)) < 1:
            raise ValueError("Invalid test fraction")
        if not 0 < float(ds.get("validation_fraction_of_train", 0.1)) < 1:
            raise ValueError("Invalid validation fraction")
    for stage in ("gan", "cnn"):
        conf = config.get(stage, {})
        for key in ("epochs", "batch_size"):
            if int(conf.get(key, 0)) <= 0:
                raise ValueError(f"{stage}.{key} must be positive")
        if float(conf.get("learning_rate", 0)) <= 0:
            raise ValueError(f"{stage}.learning_rate must be positive")
    if int(config.get("gan", {}).get("latent_dim", 0)) < 1:
        raise ValueError("gan.latent_dim must be positive")
    return True


def save_config(config, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def write_json(path, value):
    from .utils import to_serializable
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(value), file, indent=2, allow_nan=False)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
