"""Space model registry — 3-class gesture classification.

Classes:
    0: h_rv_high_revolution  (high revolution)
    1: h_sp_high_spring      (high spring)
    2: h_st_high_stationary  (high stationary)
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
