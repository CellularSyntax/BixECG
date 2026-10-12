"""Smoke tests: both bundled weight sets load strictly and run a forward pass."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import bixecg
from bixecg import pretrained

REPO = Path(__file__).resolve().parents[1]
FW_DIR = REPO / "deployment" / "firmware"


def test_version():
    assert bixecg.__version__ == "1.1.0"


def test_default_is_final_detaug():
    assert pretrained.DEFAULT_WEIGHTS == "detaug"
    assert pretrained.weights_path().name == "bixecg_mlstm_2522_detaug.pt"
    assert pretrained.weights_path("pilot").name == "bixecg_mlstm_2522.pt"
    assert set(pretrained.list_pretrained()) == {"detaug", "pilot"}
    assert pretrained.weights_path("final") == pretrained.weights_path()
    with pytest.raises(ValueError):
        pretrained.weights_path("does-not-exist")


@pytest.mark.parametrize("name", ["detaug", "pilot"])
def test_load_and_forward(name):
    model = pretrained.load_pretrained(name)  # verifies SHA-256 + strict state_dict load
    assert not model.training
    assert sum(p.numel() for p in model.parameters()) == 2522
    torch.manual_seed(0)
    x = torch.randn(2, 300, 1)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 300, 4)
    assert torch.isfinite(y).all()


def test_weight_sets_differ():
    a = pretrained.load_state_dict("detaug")
    b = pretrained.load_state_dict("pilot")
    assert list(a) == list(b)
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_get_model_registry():
    from bixecg.models.bixecg_net import BIXECG_2522_KWARGS
    from bixecg.models.models import get_model

    m = get_model("bixecgnet", **BIXECG_2522_KWARGS)
    m.load_state_dict(pretrained.load_state_dict(), strict=True)


@pytest.mark.skipif(not FW_DIR.is_dir(), reason="deployment/ not present (installed wheel)")
def test_firmware_export_matches_default_weights():
    man = json.loads((FW_DIR / "bixecg_weights_aug.json").read_text())
    flat = np.frombuffer((FW_DIR / "bixecg_weights_aug.bin").read_bytes(), dtype="<f4")
    assert flat.size == man["n_params"] == 2522
    sd = pretrained.load_state_dict("detaug")
    ref = np.concatenate([v.numpy().ravel() for v in sd.values()]).astype("<f4")
    np.testing.assert_array_equal(flat, ref)

    spec = importlib.util.spec_from_file_location("export_fw_aug", FW_DIR / "export_fw_aug.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    _, W = fw.load_exported(FW_DIR / "bixecg_weights_aug.json", FW_DIR / "bixecg_weights_aug.bin")
    model = pretrained.load_pretrained("detaug")
    x = np.random.default_rng(0).standard_normal(300)
    with torch.no_grad():
        t = model(torch.tensor(x, dtype=torch.float32)[None, :, None])[0].numpy()
    n = fw.numpy_forward(x, W)
    assert np.abs(n - t).max() < 1e-3
    assert (n.argmax(-1) == t.argmax(-1)).mean() == 1.0
