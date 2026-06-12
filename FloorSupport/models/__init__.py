from .simple_classifier import SimpleClassifier


MODEL_REGISTRY = {
    "simple_classifier": SimpleClassifier,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]
