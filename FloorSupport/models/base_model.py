class BaseModel:
    def __init__(self, name: str = "base_model"):
        self.name = name

    def train(self, train_data, val_data=None):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
