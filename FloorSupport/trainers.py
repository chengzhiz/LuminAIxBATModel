import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Union


def _preprocess_json(text: str) -> str:
    """Handle MongoDB extended JSON: ISODate(), NaN, ObjectId()."""
    text = re.sub(r'ISODate\("(.+?)"\)', r'"\1"', text)
    text = re.sub(r'ObjectId\("(.+?)"\)', r'"\1"', text)
    text = text.replace("NaN", "null")
    return text


def _extract_sorted_values(entries: list) -> List[float]:
    """Extract v-values from a list of {k, v} dicts, sorted by k for consistency."""
    values = []
    for entry in sorted(entries, key=lambda x: x.get("k", 0)):
        v = entry.get("v", [])
        values.extend(float(x) for x in v)
    return values


def load_feature_vectors_from_json(json_path: Union[str, Path]) -> List[List[float]]:
    """Convert a single JSON recording into a list of flat numeric feature vectors,
    one per bodyframe.  Height normalization is handled by the model itself."""
    path = Path(json_path)
    text = path.read_text(encoding="utf-8")
    text = _preprocess_json(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"Warning: could not parse JSON: {path}")
        return []

    bodyframes = data.get("bodyFrames", [])
    all_features = []

    for bf in bodyframes:
        features = []
        for key in ["bodyFrameHuman", "bodyFrameHumanHeading", "bodyFrameHumanPos"]:
            entries = bf.get(key, [])
            if entries:
                features.extend(_extract_sorted_values(entries))

        root_rot = bf.get("initialRootRotation", [])
        if isinstance(root_rot, list):
            features.extend(float(x) for x in root_rot)

        init_rots = bf.get("initialRotations", [])
        if init_rots:
            features.extend(_extract_sorted_values(init_rots))

        if features:
            all_features.append(features)

    return all_features


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

        json_files = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self._class_to_index(class_dir.name)
            for json_path in sorted(class_dir.glob("*.json")):
                json_files.append((json_path, label))

        total = len(json_files)
        for i, (json_path, label) in enumerate(json_files):
            pct = (i + 1) / total * 100
            bar_width = 20
            filled = int(bar_width * (i + 1) / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  Loading {split_name}  {bar}  {pct:3.0f}%  ({i+1}/{total})",
                end="", flush=True,
            )
            feature_vectors = load_feature_vectors_from_json(json_path)
            for features in feature_vectors:
                if features:
                    samples.append((features, label))
        print()  # newline after loading bar
        return samples

    def _class_to_index(self, class_name: str) -> int:
        # Lazy-build mapping from the directory names (sorted = stable order)
        if not hasattr(self, "_class_map"):
            self._class_map = {}
            for split in ["train", "val", "test"]:
                split_dir = self.data_dir / split
                if split_dir.exists():
                    for d in sorted(split_dir.iterdir()):
                        if d.is_dir() and d.name not in self._class_map:
                            self._class_map[d.name] = len(self._class_map)
        return self._class_map.get(class_name, -1)

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

    @staticmethod
    def _sparkline(values, width=30, reverse=False):
        """Draw a min-max scaled sparkline of a list of values."""
        if len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        if mx == mn:
            return "─" * min(width, len(values))
        chars = "▁▂▃▄▅▆▇█"
        if reverse:
            values = [-v for v in values]
            mn, mx = -mx, -mn
        line = ""
        step = max(1, len(values) // width)
        for i in range(0, len(values), step):
            idx = min(len(values) - 1, i)
            v = (values[idx] - mn) / (mx - mn)
            line += chars[min(len(chars) - 1, int(v * len(chars)))]
        return line

    def run(self):
        model_name = self.model.__class__.__name__
        print(f"Training started with model={model_name}")
        print(f"Data directory: {self.data_dir}")
        print(f"Epochs: {self.epochs}, batch size: {self.batch_size}")

        train_data = self._load_split("train")
        val_data = self._load_split("val")

        print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
        if not train_data:
            print("No training data found. Please check the dataset directory structure.")
            return

        # Setup checkpoint directory
        model_name = getattr(self.model, "name", "model")
        model_lr = getattr(self.model, "lr", 0.0)
        ckpt_dir = (self.data_dir.parents[1] / "checkpoints" / model_name).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        history = []

        print(f"\n{'Epoch':>6s} {'Time':>7s}  {'tr_loss':>7s}  {'tr_acc':>6s}  "
              f"{'val_loss':>7s}  {'val_acc':>6s}  loss trend")
        print("─" * 78)

        for epoch in range(self.epochs):
            t_start = time.time()

            try:
                result = self.model.train(train_data, val_data,
                                          epoch=epoch + 1, total_epochs=self.epochs)
            except TypeError:
                result = self.model.train(train_data, val_data)

            elapsed = time.time() - t_start

            if isinstance(result, dict):
                train_loss = result.get("train_loss", 0.0)
                train_acc = result.get("train_acc", 0.0)
                val_loss = result.get("val_loss", 0.0)
                val_acc = result.get("val_acc", 0.0)

                # Only save best model — delete previous best, keep only one
                if hasattr(self.model, "save") and val_acc > best_val_acc:
                    # Remove old best
                    for old in ckpt_dir.glob(f"{model_name}_ep*.pt"):
                        old.unlink()
                    best_val_acc = val_acc
                    ckpt_name = f"{model_name}_ep{self.epochs}_lr{model_lr}_acc{val_acc:.3f}.pt"
                    self.model.save(str(ckpt_dir / ckpt_name))
            else:
                train_loss, train_acc = self._compute_metrics(train_data)
                val_loss, val_acc = self._compute_metrics(val_data)

            history.append((train_loss, val_loss))

            if elapsed < 60:
                time_str = f"{elapsed:.0f}s"
            else:
                time_str = f"{elapsed/60:.0f}m{elapsed%60:.0f}s"

            marker = " ★" if val_acc == best_val_acc and val_acc > 0 else "  "
            spark = self._sparkline([h[0] for h in history])
            print(f"{epoch+1:4d}/{self.epochs:<3d} {time_str:>7s}  "
                  f"{train_loss:7.4f}  {train_acc:6.1%}  "
                  f"{val_loss:7.4f}  {val_acc:6.1%}{marker}  {spark}")

        print("─" * 78)
        print(f"Training finished  |  best val acc: {best_val_acc:.4f}")

        # Save loss plot
        self._save_loss_plot(history, ckpt_dir, model_name, self.epochs)
        print(f"Checkpoint saved: {ckpt_dir.resolve()}")


    @staticmethod
    def _save_loss_plot(history, ckpt_dir, model_name, epochs):
        """Save a train/val loss curve as PNG in the checkpoint folder."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        train_losses = [h[0] for h in history]
        val_losses = [h[1] for h in history]
        epochs_range = range(1, len(history) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, train_losses, "b-o", label="Train Loss", markersize=6)
        ax.plot(epochs_range, val_losses, "r-s", label="Val Loss", markersize=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} — Loss Curve ({epochs} epochs)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        path = ckpt_dir / f"{model_name}_loss.png"
        fig.savefig(str(path), dpi=120)
        plt.close(fig)


class GestureTrainer(Trainer):
    """Trainer that treats each JSON file as one gesture sample.

    Unlike the base Trainer which flattens every bodyframe into an independent
    sample, this keeps all bodyframes from one recording together.  The model
    receives a variable-length sequence that gets uniformly sampled to a fixed
    number of frames (handled inside the model).
    """

    def _load_gestures(self, split_name: str) -> List[Tuple[List[List[float]], int]]:
        """Return one entry per JSON: (list_of_feature_vectors, label)."""
        split_dir = self.data_dir / split_name
        gestures = []
        if not split_dir.exists():
            return gestures

        json_files = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self._class_to_index(class_dir.name)
            for json_path in sorted(class_dir.glob("*.json")):
                json_files.append((json_path, label))

        total = len(json_files)
        for i, (json_path, label) in enumerate(json_files):
            pct = (i + 1) / total * 100
            bar_width = 20
            filled = int(bar_width * (i + 1) / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  Loading {split_name}  {bar}  {pct:3.0f}%  ({i+1}/{total})",
                end="", flush=True,
            )
            feature_vectors = load_feature_vectors_from_json(json_path)
            if feature_vectors:
                gestures.append((feature_vectors, label))
        print()
        return gestures

    def _compute_metrics_gesture(self, gestures_and_labels):
        """Compute metrics on gesture-level data using the model's predict."""
        if not gestures_and_labels:
            return 0.0, 0.0
        predictions = self.model.predict(gestures_and_labels)
        true_labels = [label for _, label in gestures_and_labels]
        correct = sum(int(p == l) for p, l in zip(predictions, true_labels))
        accuracy = correct / len(true_labels) if true_labels else 0.0
        loss = sum(abs(p - l) for p, l in zip(predictions, true_labels)) / len(true_labels)
        return loss, accuracy

    def run(self):
        model_name = self.model.__class__.__name__
        print(f"Training started with model={model_name}  [gesture-level: 1 JSON = 1 sample]")
        print(f"Data directory: {self.data_dir}")
        print(f"Epochs: {self.epochs}, batch size: {self.batch_size}")

        train_gestures = self._load_gestures("train")
        val_gestures = self._load_gestures("val")

        print(f"Loaded {len(train_gestures)} training gestures and {len(val_gestures)} validation gestures")
        if not train_gestures:
            print("No training data found. Please check the dataset directory structure.")
            return

        model_name = getattr(self.model, "name", "model")
        model_lr = getattr(self.model, "lr", 0.0)
        ckpt_dir = (self.data_dir.parents[1] / "checkpoints" / model_name).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        history = []

        print(f"\n{'Epoch':>6s} {'Time':>7s}  {'tr_loss':>7s}  {'tr_acc':>6s}  "
              f"{'val_loss':>7s}  {'val_acc':>6s}  loss trend")
        print("─" * 78)

        for epoch in range(self.epochs):
            t_start = time.time()

            try:
                result = self.model.train(train_gestures, val_gestures,
                                          epoch=epoch + 1, total_epochs=self.epochs)
            except TypeError:
                result = self.model.train(train_gestures, val_gestures)

            elapsed = time.time() - t_start

            if isinstance(result, dict):
                train_loss = result.get("train_loss", 0.0)
                train_acc = result.get("train_acc", 0.0)
                val_loss = result.get("val_loss", 0.0)
                val_acc = result.get("val_acc", 0.0)

                # Only save best model — delete previous best, keep only one
                if hasattr(self.model, "save") and val_acc > best_val_acc:
                    # Remove old best
                    for old in ckpt_dir.glob(f"{model_name}_ep*.pt"):
                        old.unlink()
                    best_val_acc = val_acc
                    ckpt_name = f"{model_name}_ep{self.epochs}_lr{model_lr}_acc{val_acc:.3f}.pt"
                    self.model.save(str(ckpt_dir / ckpt_name))
            else:
                train_loss, train_acc = self._compute_metrics_gesture(train_gestures)
                val_loss, val_acc = self._compute_metrics_gesture(val_gestures)

            history.append((train_loss, val_loss))

            if elapsed < 60:
                time_str = f"{elapsed:.0f}s"
            else:
                time_str = f"{elapsed/60:.0f}m{elapsed%60:.0f}s"

            marker = " ★" if val_acc == best_val_acc and val_acc > 0 else "  "
            spark = self._sparkline([h[0] for h in history])
            print(f"{epoch+1:4d}/{self.epochs:<3d} {time_str:>7s}  "
                  f"{train_loss:7.4f}  {train_acc:6.1%}  "
                  f"{val_loss:7.4f}  {val_acc:6.1%}{marker}  {spark}")

        print("─" * 78)
        print(f"Training finished  |  best val acc: {best_val_acc:.4f}")
        self._save_loss_plot(history, ckpt_dir, model_name, self.epochs)
        print(f"Checkpoints saved in: {ckpt_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════════════
# Multi-label trainer  —  uses plumbline features + multi-hot labels
# ═══════════════════════════════════════════════════════════════════════

class MultiLabelGestureTrainer:
    """Trainer for multi-label gesture classification.

    Uses plumbline_features (73 dims) instead of raw 732-dim features.
    Each directory name is parsed into a multi-hot vector via the subclass.
    """

    def __init__(self, model, data_dir, epochs=30, batch_size=16):
        self.model = model
        self.data_dir = Path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Override in subclass to parse directory names → multi-hot vector
    # ------------------------------------------------------------------
    def _parse_label(self, dir_name: str):
        """Parse a directory name into a multi-hot vector (list of 0/1 floats).

        Subclasses MUST override this.
        """
        raise NotImplementedError("Subclass must implement _parse_label()")

    # ------------------------------------------------------------------
    def _load_gestures(self, split_name: str):
        """Return one entry per JSON: (list_of_73d_feature_vectors, multi_hot_label)."""
        from plumbline_features import load_plumbline_features_from_json

        split_dir = self.data_dir / split_name
        gestures = []
        if not split_dir.exists():
            return gestures

        json_files = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self._parse_label(class_dir.name)
            for json_path in sorted(class_dir.glob("*.json")):
                json_files.append((json_path, label))

        total = len(json_files)
        for i, (json_path, label) in enumerate(json_files):
            pct = (i + 1) / total * 100
            filled = int(20 * (i + 1) / total)
            bar = "█" * filled + "░" * (20 - filled)
            print(
                f"\r  Loading {split_name}  {bar}  {pct:3.0f}%  ({i+1}/{total})",
                end="", flush=True,
            )
            feature_vectors = load_plumbline_features_from_json(json_path)
            if feature_vectors:
                gestures.append((feature_vectors, label))
        print()
        return gestures

    # ------------------------------------------------------------------
    def run(self):
        model_name = getattr(self.model, "name", "model")
        num_codes = getattr(self.model, "num_codes", 0)
        print(f"Training started with model={model_name}  [multilabel, {num_codes} codes]")
        print(f"Data directory: {self.data_dir}")
        print(f"Epochs: {self.epochs}, batch size: {self.batch_size}")

        train_gestures = self._load_gestures("train")
        val_gestures = self._load_gestures("val")

        print(f"Loaded {len(train_gestures)} training gestures "
              f"and {len(val_gestures)} validation gestures")
        if not train_gestures:
            print("No training data found.")
            return

        model_name_s = getattr(self.model, "name", "model")
        model_lr = getattr(self.model, "lr", 0.0)
        ckpt_dir = (self.data_dir.parents[1] / "checkpoints" / model_name_s).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        history = []

        print(f"\n{'Epoch':>6s} {'Time':>7s}  {'tr_loss':>7s}  {'tr_acc':>6s}  "
              f"{'val_loss':>7s}  {'val_acc':>6s}  loss trend")
        print("─" * 78)

        for epoch in range(self.epochs):
            t_start = time.time()

            result = self.model.train(train_gestures, val_gestures,
                                      epoch=epoch + 1, total_epochs=self.epochs)

            elapsed = time.time() - t_start

            train_loss = result.get("train_loss", 0.0)
            train_acc = result.get("train_acc", 0.0)
            val_loss = result.get("val_loss", 0.0)
            val_acc = result.get("val_acc", 0.0)

            if hasattr(self.model, "save") and val_acc > best_val_acc:
                for old in ckpt_dir.glob(f"{model_name_s}_ep*.pt"):
                    old.unlink()
                best_val_acc = val_acc
                ckpt_name = f"{model_name_s}_ep{self.epochs}_lr{model_lr}_acc{val_acc:.3f}.pt"
                self.model.save(str(ckpt_dir / ckpt_name))

            history.append((train_loss, val_loss))

            if elapsed < 60:
                time_str = f"{elapsed:.0f}s"
            else:
                time_str = f"{elapsed/60:.0f}m{elapsed%60:.0f}s"

            marker = " ★" if val_acc == best_val_acc and val_acc > 0 else "  "
            spark = self._sparkline([h[0] for h in history])
            print(f"{epoch+1:4d}/{self.epochs:<3d} {time_str:>7s}  "
                  f"{train_loss:7.4f}  {train_acc:6.1%}  "
                  f"{val_loss:7.4f}  {val_acc:6.1%}{marker}  {spark}")

        print("─" * 78)
        print(f"Training finished  |  best val acc: {best_val_acc:.4f}")
        self._save_loss_plot(history, ckpt_dir, model_name_s, self.epochs)
        print(f"Checkpoints saved in: {ckpt_dir.resolve()}")

    @staticmethod
    def _sparkline(values, width=30, reverse=False):
        if len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        if mx == mn:
            return "─" * min(width, len(values))
        chars = "▁▂▃▄▅▆▇█"
        if reverse:
            values = [-v for v in values]
            mn, mx = -mx, -mn
        line = ""
        step = max(1, len(values) // width)
        for i in range(0, len(values), step):
            idx = min(len(values) - 1, i)
            v = (values[idx] - mn) / (mx - mn)
            line += chars[min(len(chars) - 1, int(v * len(chars)))]
        return line

    @staticmethod
    def _save_loss_plot(history, ckpt_dir, model_name, epochs):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        train_losses = [h[0] for h in history]
        val_losses = [h[1] for h in history]
        epochs_range = range(1, len(history) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, train_losses, "b-o", label="Train Loss", markersize=6)
        ax.plot(epochs_range, val_losses, "r-s", label="Val Loss", markersize=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} — Loss Curve ({epochs} epochs)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        path = ckpt_dir / f"{model_name}_loss.png"
        fig.savefig(str(path), dpi=120)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Floor Support multi-label trainer
# ═══════════════════════════════════════════════════════════════════════

# Code order matching Unity: FT, HN, S, D
_FLOOR_CODES = ["FT", "HN", "S", "D"]


def _parse_floor_label(dir_name: str):
    """Parse FloorSupport directory names into 4-bit multi-hot vectors.

    d_ft_dual_feet   → [1, 0, 0, 1]  (FT, D)
    d_hn_dual_hand   → [0, 1, 0, 1]  (HN, D)
    s_ft_single_feet → [1, 0, 1, 0]  (FT, S)
    """
    # Default all zeros
    label = [0.0, 0.0, 0.0, 0.0]  # [FT, HN, S, D]

    parts = dir_name.split("_")
    code_set = set(p.upper() for p in parts)

    if "FT" in code_set:
        label[0] = 1.0
    if "HN" in code_set:
        label[1] = 1.0
    if "S" == parts[0].upper() or "SINGLE" in code_set:
        label[2] = 1.0  # single
    if "D" == parts[0].upper() or "DUAL" in code_set:
        label[3] = 1.0  # dual

    # More precise parsing based on prefix
    prefix = parts[0].upper()
    if prefix == "S":
        label[2] = 1.0  # Single
    elif prefix == "D":
        label[3] = 1.0  # Dual

    return label


class FloorSupportMultiLabelTrainer(MultiLabelGestureTrainer):
    """Trainer that parses FloorSupport directory names into multi-hot labels."""

    def _parse_label(self, dir_name: str):
        return _parse_floor_label(dir_name)
