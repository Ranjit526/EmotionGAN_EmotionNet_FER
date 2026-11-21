"""FER-2013 CSV / labeled image-folder ingestion and stable split manifests."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from . import CLASSES
from .config import write_json
from .utils import image_hash, file_hash, versions


@dataclass
class Dataset:
    images: np.ndarray
    labels: np.ndarray
    usages: np.ndarray
    sources: list[str]


def _parse_pixels(text, index):
    arr = np.fromstring(str(text), sep=" ", dtype=np.int32)
    if arr.size != 48 * 48 or np.any((arr < 0) | (arr > 255)):
        raise ValueError(f"Invalid 48x48 8-bit image in row {index}: {arr.size} pixels")
    return arr.astype(np.uint8).reshape(48, 48, 1)


def _read_csv(path):
    # CSV supports conventional FER-2013 labels 0..6, 'pixels', 'Usage'.
    frames = pd.read_csv(path)
    columns = {str(c).strip().lower(): c for c in frames.columns}
    if "emotion" not in columns or "pixels" not in columns:
        raise ValueError("CSV must contain columns 'emotion' and 'pixels'")
    labels = pd.to_numeric(frames[columns["emotion"]], errors="raise").to_numpy(dtype=np.int64)
    if not np.all((labels >= 0) & (labels < 7)):
        raise ValueError("FER-2013 numeric emotion labels must be 0..6")
    image_array = np.empty((len(frames), 48, 48, 1), dtype=np.uint8)
    for index, value in enumerate(frames[columns["pixels"]]):
        image_array[index] = _parse_pixels(value, index)
    usage_col = columns.get("usage")
    usages = (frames[usage_col].fillna("").astype(str).str.strip().to_numpy()
              if usage_col is not None else np.repeat("", len(labels)))
    return Dataset(image_array, labels, usages,
                   [f"{Path(path).name}:row:{i}" for i in range(len(frames))])


def _read_folder(path):
    # Directory layout: root/{angry,disgust,...}/files or
    # root/{Training,PublicTest,PrivateTest}/{angry,disgust,...}/files
    aliases = {c.lower(): i for i, c in enumerate(CLASSES)}
    aliases.update({"disgusted": 1, "surprised": 5, "happiness": 3, "sadness": 4})
    usages, imgs, labels, sources = [], [], [], []
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = sorted(x for x in path.rglob("*") if x.is_file() and x.suffix.lower() in supported)
    for file in image_files:
        parts = [p.lower() for p in file.relative_to(path).parts[:-1]]
        class_hits = [aliases[x] for x in parts if x in aliases]
        if len(class_hits) != 1:
            continue
        usage = next((x for x in parts if x in {"training", "publictest", "privatetest"}), "")
        with Image.open(file) as image:
            gray = image.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
            imgs.append(np.asarray(gray, dtype=np.uint8)[:, :, None])
        labels.append(class_hits[0])
        usages.append(usage)
        sources.append(str(file.relative_to(path)))
    if not imgs:
        raise ValueError("No images found in recognized class-labeled folders")
    return Dataset(np.stack(imgs), np.array(labels, dtype=np.int64),
                   np.array(usages), sources)


def load_dataset(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if len(csvs) == 1:
            return _read_csv(csvs[0])
        return _read_folder(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return _read_csv(path)
    raise FileNotFoundError(f"Provide FER-2013 CSV or class-labeled image directory: {path}")


def _filter_duplicate_images(images):
    # Exact pixel-level duplicate removal for reconstructed subset; avoids hash overlap.
    seen = set()
    kept = []
    for i, image in enumerate(images):
        digest = image_hash(image)
        if digest not in seen:
            seen.add(digest)
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def _split_overlap(images, tr, va, te):
    hashed = []
    for ids in (tr, va, te):
        hashed.append({image_hash(images[i]) for i in ids})
    return {"train_val": len(hashed[0] & hashed[1]),
            "train_test": len(hashed[0] & hashed[2]),
            "val_test": len(hashed[1] & hashed[2])}


def select_splits(dataset, config):
    dsconf = config["dataset"]
    rng_seed = int(config["seed"])
    all_ids = np.arange(len(dataset.labels), dtype=np.int64)
    if dsconf["protocol"] == "subset":
        if dsconf.get("deduplicate_subset", True):
            all_ids = _filter_duplicate_images(dataset.images)
        rng = np.random.default_rng(rng_seed)
        picks = []
        for c, name in enumerate(CLASSES):
            options = all_ids[dataset.labels[all_ids] == c]
            quota = int(dsconf["subset_counts"][name])
            if len(options) < quota:
                raise ValueError(f"Insufficient unique {name} samples: need {quota}, found {len(options)}. "
                                 "Set a smaller subset quota or use the full dataset.")
            picks.extend(rng.choice(options, size=quota, replace=False).tolist())
        picks = np.asarray(sorted(picks), dtype=np.int64)
        tr_outer, te = train_test_split(picks, test_size=float(dsconf["test_fraction"]),
                                        random_state=rng_seed, stratify=dataset.labels[picks])
        tr, va = train_test_split(tr_outer, test_size=float(dsconf["validation_fraction_of_train"]),
                                  random_state=rng_seed + 1, stratify=dataset.labels[tr_outer])
        protocol_note = "reconstructed class-quotas: not identical original manuscript row IDs"
    else:
        normalized = np.asarray([str(x).strip().lower() for x in dataset.usages])
        allowed = {"training", "publictest", "privatetest"}
        if not set(normalized).issubset(allowed):
            raise ValueError("Official protocol needs FER-2013 Usage=Training/PublicTest/PrivateTest")
        tr = np.flatnonzero(normalized == "training")
        va = np.flatnonzero(normalized == "publictest")
        te = np.flatnonzero(normalized == "privatetest")
        if min(map(len, (tr, va, te))) == 0:
            raise ValueError("The official protocol requires all three partitions")
        protocol_note = "official FER-2013; exact duplicate overlaps measured but not removed"
    if min(map(len, (tr, va, te))) == 0:
        raise ValueError("Empty partition")
    for partition, ids in (("train", tr), ("validation", va), ("test", te)):
        if set(dataset.labels[ids]) != set(range(7)):
            raise ValueError(f"{partition} missing at least one of the seven classes")
    overlap = _split_overlap(dataset.images, tr, va, te)
    if any(overlap.values()):
        warnings.warn(f"Exact duplicate image overlap across partitions: {overlap}. "
                      "Do not interpret this evaluation as fully leakage-free.", stacklevel=2)
    return {"train": np.asarray(tr, dtype=np.int64), "validation": np.asarray(va, dtype=np.int64),
            "test": np.asarray(te, dtype=np.int64)}, overlap, protocol_note


def save_prepared(data_path, run_dir, config):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(data_path)
    splits, overlap, note = select_splits(ds, config)
    arrays = {}
    manifest = {"config_seed": int(config["seed"]), "protocol": config["dataset"]["protocol"],
                "note": note, "image_shape": [48, 48, 1],
                "class_index_to_name": dict(enumerate(CLASSES)),
                "image_hash_overlaps": overlap, "partitions": {},
                "dataset_path": str(Path(data_path).resolve()), "environment": versions()}
    if Path(data_path).is_file():
        manifest["dataset_file_sha256"] = file_hash(data_path)
    for part, ids in splits.items():
        arrays[f"{part}_images"] = ds.images[ids].astype(np.uint8)
        arrays[f"{part}_labels"] = ds.labels[ids].astype(np.int64)
        counts = np.bincount(ds.labels[ids], minlength=7)
        manifest["partitions"][part] = {
            "count": len(ids), "per_class": {c: int(counts[i]) for i, c in enumerate(CLASSES)},
            "sources": [ds.sources[i] for i in ids],
            "pixel_sha256": [image_hash(ds.images[i]) for i in ids]}
    np.savez_compressed(run_dir / "partitions.npz", **arrays)
    write_json(run_dir / "split_manifest.json", manifest)
    from .reporting import plot_class_distributions, plot_eda
    plot_class_distributions(run_dir, arrays["train_labels"], config)
    plot_eda(run_dir, arrays["train_images"], arrays["train_labels"], config)
    return manifest


def load_prepared(run_dir):
    path = Path(run_dir) / "partitions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run prepare first")
    with np.load(path, allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}


def demo_csv(out, per_class=32, seed=42):
    """Random patterns for code tests ONLY, not real facial data / valid FER evidence."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["emotion", "pixels", "Usage"])
        for cls in range(7):
            for i in range(per_class):
                arr = np.clip(rng.normal(110 + 5 * cls, 35, (48, 48)), 0, 255).astype(np.uint8)
                arr[(3 + cls * 3):(8 + cls * 3), 12:36] = 30 + cls * 20
                usage = "Training" if i % 10 < 8 else "PublicTest" if i % 10 == 8 else "PrivateTest"
                writer.writerow([cls, " ".join(map(str, arr.flat)), usage])
    return out
