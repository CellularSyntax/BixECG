"""Bundled pretrained BixECG weights.

Two weight sets of the same 2,522-parameter Bi-mLSTM (:class:`bixecg.models.bixecg_net.BixECGNet`)
ship with the package:

``"detaug"`` (default)
    FINAL model reported in the paper. Trained on QTDB with detector-anchored windowing
    (segmented Pan-Tompkins R-peak anchors, as on the device) plus augmentation
    (anchor jitter, +-10 % time-stretch, baseline wander, Gaussian noise).
    File: ``bixecg_mlstm_2522_detaug.pt``.

``"pilot"``
    Earlier training recipe (annotation-anchored, no augmentation); weights deployed in
    the on-device pilot reported in the paper. File: ``bixecg_mlstm_2522.pt``
    (identical to the v1.0.0 release).

Example
-------
>>> from bixecg.pretrained import load_pretrained
>>> model = load_pretrained()            # final detector-anchored + augmented weights
>>> pilot = load_pretrained("pilot")     # weights flashed for the on-device pilot
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "DEFAULT_WEIGHTS",
    "WEIGHTS",
    "list_pretrained",
    "weights_path",
    "load_state_dict",
    "load_pretrained",
]

DEFAULT_WEIGHTS = "detaug"

WEIGHTS: Dict[str, dict] = {
    "detaug": {
        "file": "bixecg_mlstm_2522_detaug.pt",
        "sha256": "a5302a1309645dc81a8f12533e82049e7a2af0ec3ca82615c72079caa4ba24eb",
        "default": True,
        "description": (
            "FINAL model reported in the paper: detector-anchored training windows + augmentation "
            "(anchor jitter, time-stretch, baseline wander, noise)."
        ),
    },
    "pilot": {
        "file": "bixecg_mlstm_2522.pt",
        "sha256": "334327d690d7630f7342a20b741923dcb3e74ca52ee4e6ed72c2e3d906b13aa4",
        "default": False,
        "description": (
            "Earlier training recipe (annotation-anchored, no augmentation); weights deployed in the "
            "on-device pilot reported in the paper."
        ),
    },
}

_ALIASES = {
    "default": "detaug",
    "final": "detaug",
    "aug": "detaug",
    "bixecg_mlstm_2522_detaug": "detaug",
    "bixecg_mlstm_2522_detaug.pt": "detaug",
    "pilot-firmware": "pilot",
    "pilot_firmware": "pilot",
    "v1.0.0": "pilot",
    "bixecg_mlstm_2522": "pilot",
    "bixecg_mlstm_2522.pt": "pilot",
}


def _resolve(name: Optional[str]) -> str:
    key = DEFAULT_WEIGHTS if name is None else str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in WEIGHTS:
        raise ValueError(f"Unknown weight set {name!r}. Available: {sorted(WEIGHTS)} (aliases: {sorted(_ALIASES)})")
    return key


def list_pretrained() -> Dict[str, str]:
    """Return ``{name: description}`` for all bundled weight sets."""
    return {k: v["description"] for k, v in WEIGHTS.items()}


def weights_path(name: Optional[str] = None) -> Path:
    """Filesystem path of a bundled ``.pt`` file (default: the final ``detaug`` weights)."""
    key = _resolve(name)
    ref = resources.files("bixecg").joinpath("weights", WEIGHTS[key]["file"])
    return Path(str(ref))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_state_dict(name: Optional[str] = None, map_location="cpu", verify: bool = True):
    """Load the raw PyTorch ``state_dict`` of a bundled weight set."""
    import torch

    key = _resolve(name)
    path = weights_path(key)
    if verify:
        got = _sha256(path)
        if got != WEIGHTS[key]["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {path.name}: {got} != {WEIGHTS[key]['sha256']}")
    sd = torch.load(path, map_location=map_location, weights_only=True)
    return sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd


def load_pretrained(name: Optional[str] = None, device="cpu", verify: bool = True, eval_mode: bool = True):
    """Build :class:`BixECGNet` and load a bundled weight set (default: final ``detaug``)."""
    from bixecg.models.bixecg_net import BIXECG_2522_KWARGS, BixECGNet

    model = BixECGNet(**BIXECG_2522_KWARGS)
    model.load_state_dict(load_state_dict(name, map_location="cpu", verify=verify), strict=True)
    model.to(device)
    return model.eval() if eval_mode else model
