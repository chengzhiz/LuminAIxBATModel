import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """1D CNN for classifying floor support from skeleton bodyframe features.

    Treats the flat feature vector as a 1-channel 1D signal.  Uses adaptive
    pooling so it works with any feature dimension.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.5,
        name: str = "cnn1d_classifier",
    ):
        super().__init__()
        self.name = name

        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 2: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 3: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # → (B, 128, 1) regardless of input length
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, feature_dim) → (B, 1, feature_dim)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN1DModel:
    """Wrapper that matches the existing BaseModel interface so it plugs
    into the Trainer unchanged."""

    def __init__(
        self,
        num_classes: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        device: str = "cpu",
        name: str = "cnn1d_classifier",
    ):
        self.name = name
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        self.model = CNN1DClassifier(
            num_classes=num_classes,
            dropout=dropout,
            name=name,
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    def train(self, train_data, val_data=None, epoch=0, total_epochs=0):
        """Train the model for one epoch."""
        self.model.train()

        # Convert data to tensors
        X_train = torch.tensor(
            [f for f, _ in train_data], dtype=torch.float32, device=self.device
        )
        y_train = torch.tensor(
            [l for _, l in train_data], dtype=torch.long, device=self.device
        )

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, drop_last=False
        )

        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = len(loader)
        report_every = max(1, n_batches // 10)
        epoch_label = f"Epoch {epoch}/{total_epochs}" if total_epochs else "Training"

        for i, (batch_X, batch_y) in enumerate(loader):
            self.optimizer.zero_grad()
            logits = self.model(batch_X)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)

            if (i + 1) % report_every == 0 or i == n_batches - 1:
                pct = (i + 1) / n_batches * 100
                bar_width = 20
                filled = int(bar_width * (i + 1) / n_batches)
                bar = "█" * filled + "░" * (bar_width - filled)
                running_loss = total_loss / total
                running_acc = correct / total
                print(
                    f"\r  {epoch_label}  {bar}  {pct:3.0f}%  "
                    f"loss={running_loss:.4f}  acc={running_acc:.3f}",
                    end="",
                    flush=True,
                )

        print()  # newline after progress line

        self.scheduler.step()

        train_loss = total_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        result = {"train_loss": train_loss, "train_acc": train_acc}

        if val_data:
            val_loss, val_acc = self._evaluate(val_data)
            result["val_loss"] = val_loss
            result["val_acc"] = val_acc

        return result

    def save(self, path):
        """Save model state to disk."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load(self, path):
        """Load model state from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(
                [f for f, _ in data], dtype=torch.float32, device=self.device
            )
            y = torch.tensor(
                [l for _, l in data], dtype=torch.long, device=self.device
            )
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        return loss, acc

    def predict(self, data):
        """Return class predictions for a list of feature vectors.

        Uses batched inference for efficiency on large datasets."""
        if not data:
            return []

        self.model.eval()
        predictions = []
        with torch.no_grad():
            # Batch all feature vectors into a single tensor
            X = torch.tensor(data, dtype=torch.float32, device=self.device)
            # Use a simple DataLoader-style batching for large tensors
            batch_size = 1024
            for i in range(0, len(X), batch_size):
                batch = X[i : i + batch_size]
                logits = self.model(batch)
                preds = logits.argmax(dim=1).tolist()
                predictions.extend(preds)

        # Handle empty features (shouldn't happen, but match legacy behaviour)
        if len(predictions) < len(data):
            predictions.extend([0] * (len(data) - len(predictions)))

        return predictions
