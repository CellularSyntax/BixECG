# Changelog

All notable changes to BixECG are documented here. Versions follow [Semantic Versioning](https://semver.org/).
Archived releases: concept DOI [10.5281/zenodo.23196330](https://doi.org/10.5281/zenodo.23196330).

## [1.1.0] — 2026-10-12

### Added
- **Final model weights** `bixecg/weights/bixecg_mlstm_2522_detaug.pt` — the model reported in the paper.
  Same 2,522-parameter Bi-mLSTM, trained on QTDB with detector-anchored windows (segmented adaptive
  Pan–Tompkins R peaks) plus augmentation (3 anchor-jittered copies per beat, σ 40 ms, clipped ±60 ms;
  ±10 % time-stretch; 0.1–0.5 Hz baseline wander; Gaussian noise). Detector-anchored, tolerance-aware
  macro-F1: LUDB 0.921, LVAD-DB (pooled across two raters) 0.878; κ 0.894 / 0.837; ECE 0.054 / 0.038.
- `bixecg.pretrained`: `load_pretrained(name=None)`, `load_state_dict()`, `weights_path()`, `list_pretrained()`
  with SHA-256 verification. Default = `"detaug"`; the earlier weights are available as `"pilot"`.
- `bixecg.models.bixecg_net.BixECGNet` (registered as `get_model("bixecgnet", ...)`): the exact architecture
  of both published weight sets.
- `deployment/firmware/`: firmware-format export of the final weights (`bixecg_weights_aug.h/.bin/.json`)
  and the exporter `export_fw_aug.py` with a pure-numpy recurrent reference forward (`--verify`).
- `scripts/retrain_anchor_experiment.py`: training/evaluation protocol of the final model
  (annotation- vs detector-anchored windowing, augmentation, segmented Pan–Tompkins detector); all inputs
  are CLI arguments.
- `tests/test_pretrained.py` smoke tests; CI now installs the full package (CPU PyTorch) and runs `pytest`.
- This changelog.

### Changed
- **Default weights** are now the final detector-anchored + augmented model.
- **Primary evaluation protocol** is detector-anchored: windows are anchored on the R peak found by the
  on-device detector rather than on the reference QRS-onset annotation (v1.0.0 protocol).
- README: model table describing both weight sets, headline results, protocol change, corrected
  loading example (the v1.0.0 snippet did not run: `get_model` takes a model name plus keyword
  arguments, and `BimLSTM` cannot be built with those arguments under `xlstm==2.0.3`; the published
  `state_dict` loads strictly into `BixECGNet`).
- **Minimum Python is now 3.11.** The pinned `xlstm==2.0.3` depends on `mlstm_kernels`, which requires
  Python ≥ 3.11, so the package's dependencies were never installable on 3.10 (v1.0.0 CI only smoke-imported
  the wheel with `--no-deps`). `requires-python`, classifiers and the CI matrix now say 3.11–3.12.
- Version 1.1.0 in `pyproject.toml`, `bixecg.__version__`, `CITATION.cff` and `.zenodo.json`; paper status
  stated as "under review".

### Kept
- `bixecg/weights/bixecg_mlstm_2522.pt` (unchanged from v1.0.0, SHA-256 `334327d6…3aa4`): earlier training
  recipe (annotation-anchored, no augmentation); weights deployed in the on-device pilot reported in the paper.

## [1.0.0] — 2026-10-06

- Initial public release: installable `bixecg` package, training/HPO/validation launchers, configs, CI.
