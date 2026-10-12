"""BixECG: efficient Bi-mLSTM ECG waveform delineation for edge deployment.

Official code for the paper "A Compact, Uncertainty-Aware mLSTM Model for Real-Time ECG
Delineation on Edge Devices" (Lung et al., 2026; under review).
Submodules (``bixecg.models``, ``bixecg.pretrained``, ``bixecg.training``, ``bixecg.hpo``,
``bixecg.utils``) are imported lazily to keep ``import bixecg`` lightweight.

Pretrained weights: ``bixecg.pretrained.load_pretrained()`` returns the final
(detector-anchored + augmented) model; ``load_pretrained("pilot")`` returns the
pilot-firmware weights.
"""

__version__ = "1.1.0"
__all__ = ["__version__"]
