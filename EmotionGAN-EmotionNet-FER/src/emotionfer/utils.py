"""General helpers and provenance, no eager heavy-framework dependency."""
from __future__ import annotations
import hashlib
import os
import random
import platform
from pathlib import Path
import numpy as np


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    return value


def set_seed(seed, deterministic=False):
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except ModuleNotFoundError:
        pass


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while (chunk := file.read(1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()


def image_hash(image):
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def versions():
    import pandas, sklearn, scipy, PIL, matplotlib
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "pillow": PIL.__version__, "matplotlib": matplotlib.__version__}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [x.name for x in tf.config.list_physical_devices("GPU")]
    except ModuleNotFoundError:
        info["tensorflow"] = "NOT INSTALLED"
        info["gpus"] = []
    return info
