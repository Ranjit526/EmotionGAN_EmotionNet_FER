#!/usr/bin/env bash
set -euo pipefail
# Requires pip install -e . && TensorFlow. Accuracy on the fake data has no research meaning.
python -m emotionfer create-demo --out data/demo_fer2013.csv --per-class 32
python -m emotionfer run --data data/demo_fer2013.csv --run-dir runs/smoke --config configs/smoke.yaml
python -m pytest -q tests
