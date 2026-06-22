"""Spine model registry — 4-class gesture classification.

Spine classes:
    0: e_extension        (bending backward)
    1: f_flexion          (bending forward)
    2: hg_hinge           (hinge)
    3: lf_lateral_flexion (bending sideways)
"""

from .gesture_cnn_v2 import GestureCNN
from .gesture_lstm import GestureLSTM
from .gesture_cnn_deep import GestureCNNDeep


def _make(cls, **kwargs):
    defaults = {"num_classes": 4, "device": "cpu"}
    defaults.update(kwargs)
    return cls(**defaults)


MODEL_REGISTRY = {
    "gesture_cnn":       lambda **kw: _make(GestureCNN, **kw),
    "gesture_lstm":      lambda **kw: _make(GestureLSTM, **kw),
    "gesture_cnn_deep":  lambda **kw: _make(GestureCNNDeep, **kw),
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def _make(cls, **kwargs):
    """Factory: inject Spine defaults (4 classes) into any gesture model."""
    defaults = {"num_classes": 4, "device": "cpu"}
    defaults.update(kwargs)
    return cls(**defaults)


MODEL_REGISTRY = {
    "gesture_cnn":       lambda **kw: _make(GestureCNN, **kw),
    "gesture_lstm":      lambda **kw: _make(GestureLSTM, **kw),
    "gesture_cnn_deep":  lambda **kw: _make(GestureCNNDeep, **kw),
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
