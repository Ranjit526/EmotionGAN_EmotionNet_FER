# Running the project (quick reference)

## Option A — Google Colab / GPU

1. Upload or unzip this project in `/content/EmotionGAN-EmotionNet-FER`.
2. Enable **Runtime → Change runtime type → T4 GPU**, if available.
3. Upload your legitimately obtained FER-2013 CSV to `/content/fer2013.csv`.
4. Run the included `notebooks/EmotionGAN_EmotionNet_Colab.ipynb`, or enter:

```python
%cd /content/EmotionGAN-EmotionNet-FER
%pip install -q -e .
!python -m emotionfer environment
!python -m emotionfer run --data /content/fer2013.csv --run-dir /content/fer_runs/paper --config configs/paper.yaml
```

Use `configs/official.yaml` and a separate output folder for the official FER-2013 partitions.

## Option B — Linux / Windows terminal

```bash
python -m pip install -e .
python -m emotionfer run --data "/path/to/fer2013.csv" --run-dir runs/paper --config configs/paper.yaml
```

On Windows, run each stage as a separate single-line command instead of using Unix `\` line continuation.

## Code-only test (no face data)

```bash
python -m emotionfer create-demo --out data/demo_fer2013.csv --per-class 32
python -m emotionfer prepare --data data/demo_fer2013.csv --run-dir runs/demo_prep --config configs/smoke.yaml
python -m pytest -q tests
```

Full neural-network smoke training additionally requires TensorFlow. Never interpret demo accuracy as a FER benchmark.

## Common errors

- **No module named `emotionfer`:** install with `pip install -e .` from the project root, or export `PYTHONPATH=src`.
- **No module named `tensorflow`:** activate the environment in which `pip install -e .` completed, and check `python -m emotionfer environment`.
- **Missing `Usage` values:** use `configs/paper.yaml` for subset CSVs. Official protocol requires all three FER-2013 `Usage` partitions.
- **Not enough samples for a class:** obtain full FER-2013 source, or reduce the seven class counts in a new YAML configuration.
- **Out of GPU memory:** reduce `gan.batch_size`, `gan.generation_batch_size`, or `cnn.batch_size` in a new run configuration, retaining the other settings where possible; report changes in the paper.
- **Interrupted GAN job:** rerun `train-gans --run-dir RUN`; latest saved epoch is restored.
- **Cannot change a run:** use a new `--run-dir`; manifests and configs are locked to preserve provenance.
- **Accuracy far from 97.89%:** different splits, independent validation, class imbalance, GAN quality and missing paper hyperparameters can explain discrepancies. Do not fabricate or tune test results.
