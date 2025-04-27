# BixECG

<img src="https://github.com/CellularSyntax/BixECG/blob/main/img/logo.png?raw=true" width="1000"/>

---

## About

This repository provides the official code for the paper **"XYZ"**, introducing **BixECG**, a highly efficient ECG sequence labeling model built on the **xLSTM** architecture [cite xLSTM paper here].

**Key features of BixECG**:
- **5–10× smaller model size** compared to previous ECG architectures.
- **Inference speeds in the nanosecond range per heartbeat**, enabling **real-time operation on edge devices**.
- **Real-world deployment**: BixECG was integrated into an Android application and successfully applied to live ECG data acquired from a **uECG** wearable patch device.

**Why Edge Deployment Matters**:  
Running ECG models directly on smartphones or wearables removes the need for continuous cloud communication, reduces latency, preserves patient privacy, and dramatically lowers energy consumption — critical for mobile health and implantable medical technologies.

---

## Repository Structure

| Folder/File | Description |
|:------------|:------------|
| `scripts/` | Training and evaluation scripts |
| `configs/` | Configuration files for training and evaluation |
| `datasets/` | Preprocessed QTDB, LUDB, and Rabbit ECG datasets (.npz format) |
| `checkpoints/` | Pretrained model weights and logs |
| `android_app/` | Source code for the real-time Android deployment |
| `img/` | Logo and visualization assets |
| `README.md` | This file |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/CellularSyntax/BixECG.git
cd BixECG
```

---

### 2. Installation

Install the Python dependencies either for **CUDA-enabled (GPU)** or **CPU-only** training and inference.

#### Option A: CUDA-enabled (Recommended for training)

```bash
pip install -r requirements_cuda.txt
```

#### Option B: CPU-only

```bash
pip install -r requirements.txt
```

---

## Training

The main training script is:

```bash
scripts/train_bixecg.py
```

To launch training:

```bash
python scripts/train_bixecg.py --config configs/train_config.yaml
```

You can customize the model configuration, training hyperparameters, and dataset choice by modifying `configs/train_config.yaml`.

---

## Evaluation

The main evaluation script is:

```bash
scripts/evaluate_bixecg.py
```

To run evaluation:

```bash
python scripts/evaluate_bixecg.py --model_path checkpoints/bixecg_ludb.pt --dataset ludb
```

Evaluation supports:
- **Strict labeling metrics**
- **Tolerance-aware evaluation**
- **Uncertainty quantification (e.g., MC Dropout Entropy Analysis)**
- **Calibration analysis (ECE, Reliability Diagrams)**

Results (metrics, figures, explainability plots) will be saved automatically.

---

## Datasets

We provide preprocessed ECG datasets stored as `.npz` files:

| Dataset | Description | Location |
|:--------|:-------------|:---------|
| **QTDB** (MIT-BIH QT Database) | Public benchmark for ECG segmentation | `datasets/qtdb/` |
| **LUDB** (Lobachevsky University Database) | Public ECG delineation dataset | `datasets/ludb/` |
| **Rabbit ECG Dataset** | Experimental ECG data from rabbit studies | `datasets/rabbit/` |

> 📢 **Important:**  
> The **LVAD dataset** (patient data) used in the paper cannot be shared publicly due to data protection regulations.

---

## Pretrained Models

All pretrained BixECG models and corresponding training logs are available under:

```bash
checkpoints/
```

Including:
- BixECG trained on LUDB
- BixECG trained on Rabbit ECG
- Quantized BixECG models for ultra-lightweight deployment

---

## Android Application

The Android application source code for **real-time ECG delineation** using BixECG is available in:

```bash
android_app/
```

The app processes ECG signals acquired from a **uECG** patch in real time, demonstrating true edge inference without external computation resources.

---

## Citation

If you use BixECG in your research or development, please cite:

> **XYZ**  
> [Insert full citation once your paper is officially published.]

---

## Acknowledgments

- **xLSTM Backbone**: BixECG is built on top of the xLSTM architecture. [Please cite the original xLSTM paper.]
- **uECG Project**: Thank you to the uECG team for providing an open-source wearable ECG device.

---

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.

---

# 🚀 Let's Bring Real-Time ECG AI to the Edge!
