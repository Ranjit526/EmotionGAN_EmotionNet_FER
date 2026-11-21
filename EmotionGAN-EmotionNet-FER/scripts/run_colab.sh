#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_colab.sh /content/fer2013.csv /content/emotionfer_runs/paper
DATA=${1:?Provide path to fer2013.csv}
RUN=${2:-runs/paper}
python -m emotionfer run --data "$DATA" --run-dir "$RUN" --config configs/paper.yaml
