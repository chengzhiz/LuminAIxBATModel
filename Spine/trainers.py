"""Single-label trainer for Spine — 6-class softmax: E, F, HG, LF, SR, U.

SR and U have no training data but are included as classes 4 and 5.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

_floor = Path(__file__).resolve().parents[1] / "FloorSupport"
sys.path.insert(0, str(_floor))
from plumbline_features import load_plumbline_features_from_json


# Class indices matching Unity spine_codes order: E, F, HG, LF, SR, U
_SPINE_CLASS_NAMES = {
    "e_extension": 0,
    "f_flexion": 1,
    "hg_hinge": 2,
    "lf_lateral_flexion": 3,
    # SR (4) and U (5) — no training data yet
}


def parse_spine_label(dir_name: str) -> int:
    """Parse Spine directory name into class index 0-5."""
    for prefix, idx in _SPINE_CLASS_NAMES.items():
        if dir_name.startswith(prefix):
            return idx
    raise ValueError(f"Unknown Spine class: {dir_name}")


class SpineTrainer:
    """Single-label trainer for Spine (6-class softmax).

    Uses plumbline features (73 dims) — matching BATPreprocess.cs.
    """

    def __init__(self, model, data_dir, epochs=30, batch_size=16):
        self.model = model
        self.data_dir = Path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size

    def _load_gestures(self, split_name: str) -> List[Tuple[List[List[float]], int]]:
        split_dir = self.data_dir / split_name
        gestures = []
        if not split_dir.exists():
            return gestures

        json_files = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = parse_spine_label(class_dir.name)
            for json_path in sorted(class_dir.glob("*.json")):
                json_files.append((json_path, label))

        total = len(json_files)
        for i, (json_path, label) in enumerate(json_files):
            pct = (i + 1) / total * 100
            filled = int(20 * (i + 1) / total)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\r  Loading {split_name}  {bar}  {pct:3.0f}%  ({i+1}/{total})",
                  end="", flush=True)
            feature_vectors = load_plumbline_features_from_json(json_path)
            if feature_vectors:
                gestures.append((feature_vectors, label))
        print()
        return gestures

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

    def run(self):
        model_name = getattr(self.model, "name", "model")
        num_classes = getattr(self.model, "num_classes", 0)
        print(f"Training started with model={model_name}  [single-label, {num_classes} classes]")
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
        print(f"Checkpoints saved in: {ckpt_dir.resolve()}")
