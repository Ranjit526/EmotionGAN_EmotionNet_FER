[github.com/Ranjit526/EmotionGAN_EmotionNet_FER/blob/main/README.md](https://github.com/Ranjit526/EmotionGAN_EmotionNet_FER/blob/main/README.md)
[![DOI](https://zenodo.org/badge/1466246574.svg)](https://doi.org/10.5281/zenodo.24412958)

# EmotionGAN + EmotionNet-CNN: Facial Expression Recognition

A deep learning framework for facial expression recognition using class-specific Generative Adversarial Networks (GANs) and a residual Convolutional Neural Network (CNN).

The framework uses GAN-based data augmentation to improve class balance, followed by EmotionNet-CNN for facial expression classification using 48 × 48 grayscale facial images from the FER-2013 dataset.

## 1. Project Overview

The project consists of two main stages:

- **Stage 1 – EmotionGAN:** Trains separate GAN models for seven facial emotion categories to improve class balance.
- **Stage 2 – EmotionNet-CNN:** Uses a residual CNN architecture to classify facial expressions into seven emotion categories.

### Supported Emotion Classes

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

## 2. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Generative Adversarial Networks (GAN)
- Convolutional Neural Networks (CNN)

## 3. Dataset

**Dataset:** FER-2013 (Facial Expression Recognition 2013)

The FER-2013 dataset contains grayscale facial images categorized into seven emotion classes.

### Dataset Specifications

- Image size: 48 × 48 pixels
- Image format: Grayscale
- Number of classes: 7
- Input format: CSV or class-labeled image directories

### FER-2013 CSV Format

The dataset must contain the following columns:

- `emotion`: Numerical emotion label.
- `pixels`: Space-separated pixel values.
- `Usage`: Dataset partition (Training, PublicTest, PrivateTest).

### Emotion Label Mapping

| Label | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |



## 4. Requirements

- Python 3.10–3.12
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- NVIDIA GPU (recommended)
- CUDA-compatible environment (if using GPU)

CPU execution is also supported for smaller experiments.

## 5. Installation

### Step 1: Extract the Project

```bash
unzip EmotionGAN-EmotionNet-FER.zip
cd EmotionGAN-EmotionNet-FER
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### Step 5: Verify the Environment

```bash
python -m emotionfer environment
```

## 6. Project Execution

### Run the Complete Framework

```bash
python -m emotionfer run \
  --data /path/to/fer2013.csv \
  --run-dir runs/paper_subset \
  --config configs/paper.yaml
```

The complete execution includes:

1. Dataset loading and preprocessing.
2. Exploratory Data Analysis (EDA).
3. Data splitting into training, validation, and testing sets.
4. Training seven class-specific GAN models.
5. Class balancing through GAN-based augmentation.
6. EmotionNet-CNN model training.
7. Model evaluation and performance analysis.
8. Saving trained models, predictions, and visualization results.

## 7. Individual Execution Modules

Each stage can also be executed independently.

### Step 1: Dataset Preparation

```bash
python -m emotionfer prepare --data /path/to/fer2013.csv --run-dir runs/paper_subset --config configs/paper.yaml
```

### Step 2: Train GAN Models

```bash
python -m emotionfer train-gans --run-dir runs/paper_subset
```

### Step 3: Generate Additional Training Images

```bash
python -m emotionfer synthesize --run-dir runs/paper_subset
```

### Step 4: Train EmotionNet-CNN

```bash
python -m emotionfer train --run-dir runs/paper_subset
```

### Step 5: Evaluate the Model

```bash
python -m emotionfer evaluate --run-dir runs/paper_subset
```

## 8. Official FER-2013 Evaluation

The framework also supports the official FER-2013 dataset partition.

- Training: Model training.
- PublicTest: Validation.
- PrivateTest: Final testing.

Run the official evaluation:

```bash
python -m emotionfer run \
  --data /path/to/fer2013.csv \
  --run-dir runs/official \
  --config configs/official.yaml
```

## 9. Model Architecture

### EmotionGAN Architecture

Each emotion category has an individual GAN consisting of:

**Generator:**
- Random noise input (100 dimensions).
- Fully connected layer.
- Feature reshaping.
- Transposed convolution layers.
- Tanh activation.
- Output image size: 48 × 48 × 1.

**Discriminator:**
- Convolutional layers.
- LeakyReLU activation.
- Batch Normalization.
- Dropout.
- Real/Fake classification.

### EmotionNet-CNN Architecture

The classifier consists of:

1. Convolutional layer (64 filters).
2. Batch Normalization.
3. ReLU activation.
4. Convolutional layer (64 filters).
5. Residual identity connection.
6. Convolutional layer (128 filters).
7. Global Average Pooling.
8. Fully Connected Layer (256 units).
9. Dropout (0.4).
10. Softmax output layer (7 classes).

### Training Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 48 × 48 × 1 |
| Number of Classes | 7 |
| GAN Models | 7 |
| GAN Training Epochs | 100 |
| CNN Training Epochs | 20 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |
| Output Activation | Softmax |

## 10. Model Evaluation Metrics

The framework evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score
- Class-wise Performance

### Evaluation Visualizations

The framework generates:

- Training and validation accuracy curves.
- Training and validation loss curves.
- Confusion matrix.
- Per-class performance charts.
- ROC curves.
- Class distribution plots.
- GAN training loss plots.

## 11. Baseline Models and Ablation Studies

The project supports baseline comparisons and ablation experiments.

### Baseline Models

```bash
python -m emotionfer compare --run-dir runs/paper_subset --suite baselines
```

### Ablation Studies

```bash
python -m emotionfer compare --run-dir runs/paper_subset --suite ablations
```

### Complete Comparison

```bash
python -m emotionfer compare --run-dir runs/paper_subset --suite all --update-matched
```

## 12. Multi-Seed Evaluation

The framework supports evaluation using multiple random seeds.

```bash
python -m emotionfer multi-seed \
  --data /path/to/fer2013.csv \
  --config configs/paper.yaml \
  --root runs/five_seed \
  --seeds 42,123,2024,2025,3407
```

Generate result summaries:

```bash
python -m emotionfer summarize --root runs/five_seed
```

Statistical summaries include:

- Mean and standard deviation.
- Student-t 95% confidence intervals.
- Wilcoxon signed-rank test (when applicable).
- Holm correction.
- Paired mean and median differences.

## 13. Facial Expression Prediction

The trained EmotionNet-CNN model can predict the emotion from an individual cropped facial image.

```bash
python -m emotionfer predict \
  --model runs/paper_subset/classifier/classifier_final.keras \
  --image /path/to/cropped_face.png \
  --out runs/paper_subset/single_prediction.json
```

### Prediction Output

The model provides:

- Predicted emotion category.
- Class probabilities for all seven emotions.
- Prediction results in JSON format.

**Note:** The input image must already contain a cropped face.

## 14. Project Directory Structure

```text
EmotionGAN-EmotionNet-FER/
│
├── configs/
│   ├── paper.yaml
│   ├── official.yaml
│   └── smoke.yaml
│
├── src/
│   └── emotionfer/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── __main__.py
│
├── tests/
│
├── scripts/
│   └── smoke_test.sh
│
└── requirements.txt
```

### Generated Results Directory

```text
runs/
└── paper_subset/
    │
    ├── config.lock.yaml
    ├── split_manifest.json
    ├── partitions.npz
    │
    ├── figures/
    │   ├── class_distribution.png
    │   ├── class_balance.png
    │   ├── eda_four_panels.png
    │   └── gan_losses.png
    │
    ├── gans/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    │
    ├── classifier/
    │   ├── best_validation.keras
    │   ├── classifier_final.keras
    │   ├── cnn_history.csv
    │   └── training_curves.png
    │
    ├── evaluation/
    │   ├── metrics.json
    │   ├── predictions.csv
    │   ├── confusion_matrix.png
    │   ├── per_class_scores.png
    │   └── roc_one_vs_rest.png
    │
    └── experiments/
```

## 15. Testing

Install pytest:

```bash
python -m pip install pytest
```

Run tests:

```bash
python -m pytest -q tests
```

Run the smoke test:

```bash
bash scripts/smoke_test.sh
```

## 16. Key Features

- Seven-class facial expression recognition.
- Class-specific GAN architecture.
- GAN-based training data augmentation.
- Residual CNN classification.
- Automated dataset preprocessing.
- Exploratory Data Analysis.
- Model training and evaluation.
- Confusion matrix and ROC visualization.
- Baseline model comparisons.
- Ablation studies.
- Multi-seed statistical evaluation.
- Single-image emotion prediction.
- Model checkpoint saving.

## 17. License

This project source code is licensed under the MIT License.

