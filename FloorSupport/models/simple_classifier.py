from .base_model import BaseModel


class SimpleClassifier(BaseModel):
    def __init__(self, name: str = "simple_classifier"):
        super().__init__(name=name)

    def train(self, train_data, val_data=None):
        print(f"Training placeholder for SimpleClassifier with {len(train_data)} samples")
        if val_data:
            print(f"Validation samples: {len(val_data)}")
        return self

    def predict(self, data):
        predictions = []
        for features in data:
            if not features:
                predictions.append(0)
                continue
            score = sum(float(x) for x in features[:3])
            predictions.append(0 if score < 3.0 else 1)
        return predictions
