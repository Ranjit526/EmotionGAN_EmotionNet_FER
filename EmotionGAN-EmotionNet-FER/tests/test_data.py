import json
import numpy as np
import pytest
from pathlib import Path
from emotionfer.config import load_config, validate_config
from emotionfer.data import demo_csv, load_dataset, select_splits, save_prepared, load_prepared
from emotionfer import CLASSES
from emotionfer.utils import image_hash


def config_smoke():
    return load_config(Path(__file__).parents[1] / "configs/smoke.yaml")


def test_emotion_label_order():
    assert CLASSES == ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


def test_demo_loading_and_partition_sizes(tmp_path):
    filename = demo_csv(tmp_path / "demo.csv", per_class=32)
    ds = load_dataset(filename)
    assert ds.images.shape == (224, 48, 48, 1)
    assert ds.images.dtype == np.uint8
    assert np.bincount(ds.labels, minlength=7).tolist() == [32]*7
    splits, duplicate_counts, _ = select_splits(ds, config_smoke())
    assert set(map(int, np.concatenate(list(splits.values())))) <= set(range(224))
    assert len(set(np.concatenate(list(splits.values())).tolist())) == sum(map(len, splits.values()))
    assert all(n == 0 for n in duplicate_counts.values())
    expected = sum(config_smoke()["dataset"]["subset_counts"].values())
    assert sum(map(len, splits.values())) == expected
    assert len(splits["test"]) in (24, 25)


def test_save_prepared_and_manifest(tmp_path):
    filename = demo_csv(tmp_path / "demo.csv", per_class=32)
    manifest = save_prepared(filename, tmp_path / "run", config_smoke())
    loaded = load_prepared(tmp_path / "run")
    assert loaded["train_images"].shape[1:] == (48, 48, 1)
    assert loaded["test_images"].shape[1:] == (48, 48, 1)
    assert len(manifest["partitions"]["train"]["pixel_sha256"]) == len(loaded["train_images"])
    assert (tmp_path / "run" / "figures" / "eda_four_panels.png").is_file()
    assert "tensorflow" in manifest["environment"]


def test_official_split_from_usage(tmp_path):
    filename = demo_csv(tmp_path / "demo.csv", per_class=32)
    ds = load_dataset(filename)
    conf = config_smoke()
    conf["dataset"]["protocol"] = "official"
    splits, overlaps, note = select_splits(ds, conf)
    assert len(splits["train"]) > len(splits["validation"])
    assert len(splits["test"]) > 0
    assert all(v == 0 for v in overlaps.values())


def test_invalid_quotas_fail():
    conf = config_smoke()
    conf["dataset"]["subset_counts"]["Disgust"] = 0
    with pytest.raises(ValueError):
        validate_config(conf)


def test_hash_consistent():
    x = np.arange(48*48, dtype=np.uint8).reshape(48,48,1)
    assert image_hash(x) == image_hash(x.copy())
    assert image_hash(x) != image_hash(np.zeros_like(x))
