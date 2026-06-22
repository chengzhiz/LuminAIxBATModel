"""LimbExpression model registry — 3-class gesture classification.

Classes:
    0: lb_dl_sy_g  (LowerBody DualLimb Symmetric Ground)
    1: lb_sl_as_a  (LowerBody SingleLimb Asymmetric Air)
    2: lb_sl_as_g  (LowerBody SingleLimb Asymmetric Ground)
"""

from .gesture_cnn_v2 import GestureCNN
from .gesture_lstm import GestureLSTM
from .gesture_cnn_deep import GestureCNNDeep


def _make(cls, **kwargs):
    defaults = {"num_classes": 3, "device": "cpu"}
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
