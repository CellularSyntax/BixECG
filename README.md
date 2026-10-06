# BixECG

<img src="https://github.com/CellularSyntax/BixECG/blob/master/img/bixecg_logo.png" width="1000"/>

[![CI](https://github.com/CellularSyntax/BixECG/actions/workflows/ci.yml/badge.svg)](https://github.com/CellularSyntax/BixECG/actions/workflows/ci.yml)
[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23194750.svg)](https://doi.org/10.5281/zenodo.23194750)

---

## About

Official code for the paper **"A Compact, Uncertainty-Aware mLSTM Model for Real-Time ECG
Delineation on Edge Devices"** (Lung et al., 2026).

**BixECG** performs **per-sample ECG waveform delineation** — labelling every sample of a
single-lead ECG as **No-Wave (NW), P-wave, QRS-complex, or T-wave** — using a bidirectional
**matrix-memory LSTM (Bi-mLSTM)**, the mLSTM block of the
[xLSTM](https://arxiv.org/abs/2405.04517) architecture.

**Key properties**

- **Compact:** the selected model has only **2,522 parameters** (≈10 kB) — roughly 5–10× smaller
  than comparable ECG delineation networks.
- **Real-time on the edge:** runs on a Cortex-M7 microcontroller (Arduino Portenta H7) at
  **≈129 ms per beat** using **22 kB of on-chip SRAM**, with no cloud dependency.
- **Uncertainty-aware:** provides per-sample softmax confidence, entropy, and calibration
  (ECE, reliability), with an error-detection operating point for selective review.
- **Validated on a live wearable stream:** demonstrated end-to-end on a D-Heart monitor
  streaming over BLE to the edge device.

---

## Repository structure

| Folder / file | Description |
|:--------------|:------------|
| `bixecg/models/` | Model definitions (Bi-mLSTM and the retrained baselines) |
| `bixecg/training/` | Per-model training scripts |
| `bixecg/hpo/` | Optuna-based hyperparameter optimization |
| `bixecg/utils/` | Preprocessing, tolerance-aware evaluation, calibration, plotting |
| `conf/` | JSON configuration files (architecture + training/eval settings) |
| `train_model.py` | Training launcher |
| `hpo_model.py` | Hyperparameter-optimization launcher |
| `validate_model.py` | Evaluation launcher (metrics, calibration, uncertainty, plots) |
| `img/` | Logo and assets |

---

## Installation

BixECG is a Python package (Python ≥ 3.10).

```bash
# as a package (recommended)
pip install git+https://github.com/CellularSyntax/BixECG.git

# or, for development / to run the training & evaluation scripts
git clone https://github.com/CellularSyntax/BixECG.git
cd BixECG
pip install -e .
```

The pinned environment used for the paper is recorded in `requirements_new.txt`
(`pip install -r requirements_new.txt`). Training the Bi-mLSTM requires a working
[`xlstm`](https://pypi.org/project/xlstm/) / `mlstm_kernels` installation (and, for the
fused kernels, a CUDA GPU with `triton`).

---

## Usage

### Training

```bash
python train_model.py --model BiXLSTM      # options: BiXLSTM, Peimankar, Jimenez, Liue
```

### Hyperparameter optimization

```bash
python hpo_model.py --model BiXLSTM        # options: BiXLSTM, Peimankar
```

### Evaluation

```bash
python validate_model.py --model_name <checkpoint> \
    --data_dir ../DATA/ludb --db_name ludb --filter_data --beat_aligned
```

Evaluation produces strict and tolerance-aware metrics, per-class ROC/AUC, confusion
matrices, calibration (ECE, reliability diagrams), uncertainty (entropy) summaries, and
shaded delineation plots. Model architecture, training hyperparameters, and tolerance
windows are set in the `conf/*.json` files.

---

## Pretrained model

The selected **2,522-parameter Bi-mLSTM** weights (the model reported in the paper) are
published on Hugging Face:

**https://huggingface.co/maxhaberbusch/BixECG**

```python
import torch
from bixecg.models.models import get_model

cfg = {"name": "bimlstm", "params": {
    "input_dim": 1, "num_classes": 4, "embedding_dim": 8,
    "num_blocks": 3, "num_heads": 1, "conv1d_kernel_size": 11, "dropout": 0.1}}
model = get_model(cfg)
sd = torch.load("bixecg_mlstm_2522.pt", map_location="cpu")
model.load_state_dict(sd.get("model_state_dict", sd))
model.eval()
```

---

## Datasets

| Dataset | Role | Availability |
|:--------|:-----|:-------------|
| **QTDB** (MIT-BIH QT Database) | Training | Public — [PhysioNet](https://physionet.org/content/qtdb/) |
| **LUDB** (Lobachevsky University Database) | External evaluation | Public — [PhysioNet](https://physionet.org/content/ludb/) |
| **LVAD-DB** (Medical University of Vienna) | External evaluation | **Restricted** — not distributed |
| **D-Heart pilot** (Medical University of Vienna) | On-device pilot | **Restricted** — not distributed |

> ⚠️ **Data protection.** The **LVAD-DB** and **D-Heart pilot** recordings are proprietary
> patient data and **cannot be shared publicly** under data-protection regulations. They are
> **not** included in this repository, the release archive, or the Hugging Face model. They
> are available from the corresponding author on reasonable request. QTDB and LUDB are public
> and are **not** redistributed here — download them from PhysioNet.

---

## Citation

If you use BixECG, please cite the paper and the software (see `CITATION.cff`):

> Lung D, Heute P, Marx M, Schlöglhofer T, Abart T, Moscato F, Riebandt J, Zimpfer D, Haberbusch M.
> *A Compact, Uncertainty-Aware mLSTM Model for Real-Time ECG Delineation on Edge Devices.*
> 2026. (Preprint; journal submission in preparation.)

**Software archive:** [10.5281/zenodo.23194750](https://doi.org/10.5281/zenodo.23194750)
(version DOI; resolves once the Zenodo deposition is published).

---

## Acknowledgments

- **xLSTM / mLSTM backbone** — BixECG builds on the mLSTM block of the xLSTM architecture
  (Beck et al., 2024).
- **D-Heart** — ECG acquisition hardware used for the deployment pilot.

---

## License

© Medical University of Vienna, Center for Medical Physics and Biomedical Engineering,
Cardiovascular Dynamics & Artificial Organs Group.

This project is dual-licensed: **GPLv2** for non-commercial use, or a custom commercial
license from the Medical University of Vienna. See the [`LICENSE`](LICENSE) file for full
terms.
