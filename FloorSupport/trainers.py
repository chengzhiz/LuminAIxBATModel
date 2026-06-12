import ast
import csv
from pathlib import Path
from typing import List, Tuple


def load_feature_vector_from_csv(csv_path: str | Path) -> List[float]:
    """Convert a single bodyframe CSV file into a flat numeric feature vector."""
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        return []

    cells = [cell.strip() for cell in rows[1] if cell and cell.strip()]
    if not cells:
        return []

    def extract_values(value):
        values = []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value

        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "v" in item:
                    values.extend(float(x) for x in item["v"])
                elif isinstance(item, (int, float)):
                    values.append(float(item))
        elif isinstance(parsed, dict):
            if "v" in parsed:
                values.extend(float(x) for x in parsed["v"])
        elif isinstance(parsed, (int, float)):
            values.append(float(parsed))
        elif isinstance(parsed, str):
            try:
                values.append(float(parsed))
            except (TypeError, ValueError):
                if "v" in parsed:
                    import re
                    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", parsed)]
                    values.extend(nums)
        return values

    combined_values = []
    if len(cells) > 1:
        combined_text = " ".join(cells)
        combined_values = extract_values(combined_text)

    if combined_values:
        return combined_values

    values = []
    for cell in cells:
        values.extend(extract_values(cell))

    return values


class Trainer:
    def __init__(self, model, data_dir, epochs=10, batch_size=32):
        self.model = model
        self.data_dir = Path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size

    def _load_split(self, split_name: str) -> List[Tuple[List[float], int]]:
        split_dir = self.data_dir / split_name
        samples = []
        if not split_dir.exists():
            return samples

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self._class_to_index(class_dir.name)
            for csv_path in sorted(class_dir.glob("*.csv")):
                features = load_feature_vector_from_csv(csv_path)
                if features:
                    samples.append((features, label))
        return samples

    def _class_to_index(self, class_name: str) -> int:
        mapping = {
            "d_ft_dual_feet": 0,
            "d_hn_dual_hand": 1,
            "s_ft_single_feet": 2,
        }
        return mapping.get(class_name, -1)

    def _compute_metrics(self, features_and_labels):
        if not features_and_labels:
            return 0.0, 0.0

        predictions = self.model.predict([features for features, _ in features_and_labels])
        true_labels = [label for _, label in features_and_labels]

        correct = sum(int(pred == label) for pred, label in zip(predictions, true_labels))
        accuracy = correct / len(true_labels) if true_labels else 0.0

        loss = 0.0
        for pred, label in zip(predictions, true_labels):
            loss += abs(pred - label)
        loss /= len(true_labels) if true_labels else 1.0

        return loss, accuracy

    def _format_bar(self, value: float, width: int = 30, invert: bool = False) -> str:
        if value < 0:
            value = 0.0
        if invert:
            scaled = 1.0 / (1.0 + value)
        else:
            scaled = min(max(value, 0.0), 1.0)
        filled = int(round(scaled * width))
        filled = max(0, min(width, filled))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def _format_progress(self, epoch: int, total_epochs: int) -> str:
        completed = epoch + 1
        width = 30
        filled = int(round((completed / total_epochs) * width)) if total_epochs else width
        filled = max(0, min(width, filled))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def run(self):
        print(f"Training started with model={self.model.__class__.__name__}")
        print(f"Data directory: {self.data_dir}")
        print(f"Epochs: {self.epochs}, batch size: {self.batch_size}")

        train_data = self._load_split("train")
        val_data = self._load_split("val")

        print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
        if not train_data:
            print("No training data found. Please check the dataset directory structure.")
            return

        self.model.train(train_data, val_data)

        print("\n=== Training progress ===")
        for epoch in range(self.epochs):
            train_loss, train_acc = self._compute_metrics(train_data)
            val_loss, val_acc = self._compute_metrics(val_data)
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"overall: {self._format_progress(epoch, self.epochs)}")
            print(f"train loss: {train_loss:.4f} {self._format_bar(train_loss, invert=True)}")
            print(f"train acc : {train_acc:.4f} {self._format_bar(train_acc)}")
            print(f"val loss  : {val_loss:.4f} {self._format_bar(val_loss, invert=True)}")
            print(f"val acc   : {val_acc:.4f} {self._format_bar(val_acc)}")

        print("\nTraining finished")
