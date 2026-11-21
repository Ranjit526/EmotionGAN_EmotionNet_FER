# Implementation audit, paper alignment and methodological choices

## Source studied

The supplied 25-page manuscript includes: abstract, methodological subsections 3.1–3.8, Figures 1–5, Tables 1–5, and Algorithm 1. The uploaded file terminates in the early experimental section (4.1); the stated later detailed baseline/ablation results are not present. Accordingly, this project cannot recreate unspecified baseline papers or prove the reported scores.

## Paper component → source

| Paper component | Implementation | Notes |
|---|---|---|
| §3.2 FER-2013 and EDA | `data.py`, `reporting.py` | Uses real training images for descriptive EDA; no faces exported by default |
| §3.2 stratified 80:20 | `data.select_splits` | Adds strictly separate inner validation: 72:8:20 overall |
| Fig. 2 / §3.3 GAN generator | `models.build_generator` | 100 latent dimensions, 3 stride-2 transpose convolutions, Tanh |
| Fig. 2 / §3.3 discriminator | `models.build_discriminator` | 2 conv layers, LeakyReLU, BN, dropout, 1 realness logit |
| §3.3 class-wise training | `gan.train_gans` | One independent GAN per class; never sees validation/test |
| §3.3 synthetic sample budget | `gan.synthetic_budget`, `gan.synthesize` | Largest real train class count minus the class count |
| Fig. 3 / Table 3 / §3.4 CNN | `models.build_emotionnet` | 146,759 trainable parameters, one residual Add and GAP |
| §3.5 training strategy | `trainer.train_classifier` | Adam 1e-4; 20 epochs; validation-based ReduceLROnPlateau |
| §3.2 geometric transforms | `augment.augment_batch` | Per-image random affine with rotation, translation, shear, zoom, flip |
| Algorithm 1 | `cli.run`, `cli.prepare`, `train-gans`, `synthesize`, `train`, `evaluate` | Split locked before any GAN weights or augmentation |
| §3.7 classification metrics | `evaluation.compute_metrics` | Standard macro-F1 *and* harmonic macro-P/R F1 |
| §4 baseline/ablation claims | `experiments.py` | Implementation-defined B1–B5 and A1–A4; never mislabel as exact original baselines |

## Reproducibility fixes relative to the described manuscript

**Test-set usage.** Manuscript §3.6 and §4.0 state that a ReduceLROnPlateau scheduler monitors a held-out loss, despite the abstract describing a leakage-controlled test. That feedback changes future optimizer updates, so it constitutes test-driven learning-rate selection. In this code, the 80% outer training portion is split internally into 90% train and 10% validation. All adaptive decisions use validation; only final trained weights are evaluated on the held-out 20% test. This **changes the synthetic budget** and should be reported as a methodological refinement rather than a numerical reproduction.

**Official vs custom subsets.** The source uses ~7,070 images with per-class illustrative quotas and does not disclose the original row IDs or seed. The code chooses a documented, reproducible random subset. The official split is also supported as a separate protocol. Never equate the two.

**Image duplicates.** In reconstructed-subset mode the loader deduplicates identical pixel arrays globally *before sampling*, so image hashes are disjoint across train, validation and test. Original FER-2013 official partitions are preserved; potential identical-image overlap is warned and quantified, since removing official test rows would change the benchmark. Subject independence is not testable because person identities are not supplied.

**FER label IDs.** Numeric FER-2013 label IDs are `0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral`. The source manuscript's diagram verbally places Neutral before Sad/Surprise. The code consistently follows the native IDs and records mappings.

**Metrics.** Standard seven-class macro-F1 is the mean of class F1 scores. The manuscript describes a harmonic mean of macro-precision and macro-recall; those quantities differ in general. Both are output, without conflation.

**Numerical claim.** The paper's 97.89% accuracy with an illustrative 7,070-image 80:20 split of 1,414 test images cannot represent the exact fraction of correct predictions for that fixed test size when rounded to two decimals; its tabulated class counts are approximate. This is one reason to report exact denominators and a confusion matrix from each actual run.

## Explicit design choices not numerically pinned down in source

- Generator expands a 100-dimensional normal input to 6×6×256; transposed convolutions use 4×4 stride-2 same padding and channels 128/64/1 to obtain 48×48×1.
- Discriminator has channels 64 and 128 with 4×4 stride-2 kernels and dropout 0.3. Dense output logits are mathematically equivalent to a sigmoid with BCE, with improved numerical stability.
- GAN Adam uses β₁=0.5 and β₂=0.999; CNN Adam uses β₁=0.9, β₂=0.999. Both use lr=0.0001.
- Augmentation magnitudes are ±10° rotation, ±8% translation, ±8° shear, zoom 0.90–1.10 and horizontal flip probability 0.5; only training images are transformed.
- Training uses final-epoch classifier as primary model; best-validation model is archived separately, not silently substituted.
- Preprocessing is fixed `2*(image/255)-1` in the entire pipeline, consistent with Tanh GAN outputs.
- No synthetic samples are inserted into validation/test. No generator images are shared outside a local run by default.

## Expected runtime and GPU considerations

Seven independent GANs are substantial. The official FER-2013 benchmark runs seven 100-epoch GAN training jobs plus the classifier. The saved checkpoints support recovery. Inference requires only the ~0.15M-parameter classifier, with the caveat that TensorFlow framework overhead may dominate for single 48×48 images.

## Quality controls

- Ingestion rejects invalid class IDs, non-integer pixel values, malformed shapes and insufficient subset quotas.
- Split manifests record row/path origins, per-image hashes, class counts, input file SHA-256 and software versions.
- GAN sampler verifies output count, finite outputs, 48×48 shape and Tanh range. Numerical diagnostics compare generated images with generated/real examples in the same class; warnings are **not independent proof** of class fidelity or privacy.
- CNN model enforces Table 3's exact trainable parameter count.
- Evaluation uses held-out real images once in a forward-only model, saving every predicted probability and confusion-matrix count.
- Paired statistical reporting is optional; no paper percentages are hardcoded as measured results.

## Known restrictions

- No real FER dataset or trained model is included due to redistribution restrictions and because training data were not provided.
- A cropped grayscale face is required for inference. Automatic detection, face tracking, webcam support and temporal sequence modelling are outside the paper's implemented scope.
- GAN quality and category labels should be independently audited before asserting semantic or demographic validity.
- Model scores across seeds are not deterministic to machine precision on every GPU/CUDA installation.

## Test commands

```bash
python -m pytest -q tests
python -m emotionfer create-demo --out data/demo_fer2013.csv --per-class 32
python -m emotionfer run --data data/demo_fer2013.csv --run-dir runs/smoke --config configs/smoke.yaml
```

The first command includes TensorFlow-specific shape, identity-affine and gradient tests when TensorFlow is installed. The latter trains a one-epoch smoke configuration on random non-face patterns.
