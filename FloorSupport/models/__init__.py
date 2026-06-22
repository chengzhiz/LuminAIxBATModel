from .simple_classifier import SimpleClassifier
from .cnn1d_classifier import CNN1DModel
from .gesture_cnn import GestureCNNModel
from .gesture_cnn_v2 import GestureCNN
from .gesture_lstm import GestureLSTM
from .gesture_cnn_deep import GestureCNNDeep


MODEL_REGISTRY = {
    "simple_classifier": SimpleClassifier,
    "cnn1d_classifier": CNN1DModel,
    "gesture_cnn": GestureCNNModel,
    "gesture_cnn_v2": GestureCNN,
    "gesture_lstm": GestureLSTM,
    "gesture_cnn_deep": GestureCNNDeep,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]
