"""Inference on an already cropped facial image. No face detector is bundled."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from . import CLASSES


def predict_face(model_path, image_path):
    import tensorflow as tf
    with Image.open(image_path) as f:
        gray = f.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)
    arr = (arr / 127.5 - 1.)[None, :, :, None]
    model = tf.keras.models.load_model(model_path)
    probabilities = np.asarray(model.predict(arr, verbose=0)[0], dtype=np.float64)
    idx = int(np.argmax(probabilities))
    return {"image": str(Path(image_path)), "model": str(Path(model_path)),
            "predicted_expression": CLASSES[idx], "confidence": float(probabilities[idx]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
            "important": "Predicted facial expression, not an objective measurement of internal emotional state."}
