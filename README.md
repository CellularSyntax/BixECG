# BixECG

<img src="https://github.com/CellularSyntax/BixECG/blob/master/img/bixecg_logo.png" width="1000"/>

[![CI](https://github.com/CellularSyntax/BixECG/actions/workflows/ci.yml/badge.svg)](https://github.com/CellularSyntax/BixECG/actions/workflows/ci.yml)
[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23196330.svg)](https://doi.org/10.5281/zenodo.23196330)

---

## About

Official code and pretrained weights for the paper **"A Compact, Uncertainty-Aware mLSTM Model
for Real-Time ECG Delineation on Edge Devices"** (Lung et al., 2026; manuscript under review).

**BixECG** performs **per-sample ECG waveform delineation** — labelling every sample of a
single-lead ECG as **No-Wave (NW), P-wave, QRS-complex, or T-wave** — using a bidirectional
**matrix-memory LSTM (Bi-mLSTM)**, the mLSTM block of the
[xLSTM](https://arxiv.org/abs/2405.04517) architecture.

**Key properties**

- **Compact:** the selected model has only **2,522 parameters** (≈10 kB float32) — roughly 5–10×
  smaller than comparable ECG delineation networks.
- **Accurate under deployment windowing:** tolerance-aware macro-F1 **0.921** on LUDB and **0.878**
  on LVAD-DB (pooled across two raters) when windows are anchored by the on-device R-peak detector
  (see [Pretrained models](#pretrained-models)).
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
| `bixecg/pretrained.py` | Loader for the bundled pretrained weights (`load_pretrained()`) |
| `bixecg/weights/` | Bundled weights: `bixecg_mlstm_2522_detaug.pt` (final, default) and `bixecg_mlstm_2522.pt` (pilot firmware) |
| `scripts/retrain_anchor_experiment.py` | Training/evaluation protocol of the final model (detector-anchored windowing, augmentation, segmented Pan–Tompkins detector) |
| `deployment/firmware/` | Firmware-format export (`.h` / `.bin` / `.json`) of the final weights + exporter with numpy reference check |
| `tests/` | Smoke tests (both weight sets load and run a forward pass; firmware export matches the weights) |
| `img/` | Logo and assets |

---

## Installation

BixECG is a Python package (Python ≥ 3.11; required by `xlstm`'s `mlstm_kernels` dependency).

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

### Final training / evaluation protocol (detector-anchored)

```bash
python scripts/retrain_anchor_experiment.py --anchor aug \
    --qtdb-csv <path>/all_ecg_data.csv \
    --eval ludb=<path>/ludb_stream.npz \
    --eval lvad_rater1=<path>/r1.npz --eval lvad_rater2=<path>/r2.npz \
    --det-band lvad_rater1=8,18 --det-band lvad_rater2=8,18 \
    --pool LVAD=lvad_rater1,lvad_rater2 --out runs/aug
```

`--anchor aug` is the recipe of the final weights; `ann` reproduces the earlier
(annotation-anchored, no augmentation) recipe of the pilot-firmware weights. Each `--eval` file is
an `.npz` with `x` (raw single-lead ECG at 250 Hz) and `y` (per-sample labels 0=NW, 1=P, 2=QRS,
3=T). Run with `--help` for all options.

### Legacy evaluation launcher

```bash
python validate_model.py --model_name <checkpoint> \
    --data_dir ../DATA/ludb --db_name ludb --filter_data --beat_aligned
```

Evaluation produces strict and tolerance-aware metrics, per-class ROC/AUC, confusion
matrices, calibration (ECE, reliability diagrams), uncertainty (entropy) summaries, and
shaded delineation plots. Model architecture, training hyperparameters, and tolerance
windows are set in the `conf/*.json` files.

---

## Pretrained models

Both weight sets use the same 2,522-parameter Bi-mLSTM (`bixecg.models.bixecg_net.BixECGNet`).
They are bundled with the package (`bixecg/weights/`) and also published on
[Hugging Face](https://huggingface.co/maxhaberbusch/BixECG) and in the
[Zenodo archive](https://doi.org/10.5281/zenodo.23196330).

| Name (`load_pretrained(...)`) | File | Training recipe | Role |
|:--|:--|:--|:--|
| **`"detaug"` (default)** | `bixecg_mlstm_2522_detaug.pt` | QTDB, **detector-anchored** windows (segmented Pan–Tompkins R peaks) **+ augmentation** (anchor jitter σ 40 ms, ±10 % time-stretch, baseline wander, Gaussian noise) | **Final model reported in the paper** — headline results below |
| `"pilot"` | `bixecg_mlstm_2522.pt` | QTDB, annotation-anchored windows, no augmentation (earlier recipe; identical to the v1.0.0 release) | Weights deployed in the on-device pilot reported in the paper |

**Headline results of the final (`detaug`) model** — tolerance-aware evaluation with
detector-anchored windows (P ±52 ms, QRS ±48 ms, T ±124 ms):

| External test set | Macro-F1 | Cohen's κ | ECE |
|:--|:--:|:--:|:--:|
| LUDB | 0.921 | 0.894 | 0.054 |
| LVAD-DB (pooled across two raters) | 0.878 | 0.837 | 0.038 |

2,522 parameters, ≈10 kB of float32 weights.

**Evaluation protocol change in v1.1.0.** The primary evaluation now anchors each 300-sample
window on the R peak found by the same segmented Pan–Tompkins detector that runs on the device
(shifted back by the R-to-QRS-onset offset), instead of on the reference QRS-onset annotation.
This matches the deployment signal chain; annotation-anchored windows (the v1.0.0 protocol)
place the beat ideally and are therefore optimistic. The final weights were trained with the
same detector-anchored windowing plus augmentation; see `scripts/retrain_anchor_experiment.py`.

```python
import torch
from bixecg.pretrained import load_pretrained, list_pretrained

print(list_pretrained())               # {'detaug': ..., 'pilot': ...}
model = load_pretrained()              # final detector-anchored + augmented weights (default)
pilot = load_pretrained("pilot")       # pilot-firmware weights

# x: (batch, 300, 1) single-lead ECG, 250 Hz, band-pass 0.5–40 Hz, per-window z-score,
#    window starting at the (detector-derived) QRS onset
x = torch.randn(1, 300, 1)
with torch.no_grad():
    probs = model(x).softmax(-1)       # (batch, 300, 4) over {NW, P, QRS, T}
labels = probs.argmax(-1)
```

The firmware export of the final weights for the Arduino Portenta H7 is in
[`deployment/firmware/`](deployment/README.md).

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
> 2026. (Manuscript under review.)

**Software archive:** [10.5281/zenodo.23196330](https://doi.org/10.5281/zenodo.23196330)
(concept DOI — always resolves to the latest version; v1.1.0 contains the final weights).
See [`CHANGELOG.md`](CHANGELOG.md) for the differences between releases.

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
