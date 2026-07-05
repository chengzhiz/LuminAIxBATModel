"""Training entry point for Spine movement classification."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure this package is importable
_here = Path(__file__).resolve().parent
_floor = _here.parent / "FloorSupport"
sys.path.insert(0, str(_here))
sys.path.append(str(_floor))

# Import Spine's own modules
from models.multilabel_spine import MultiLabelSpineModel
from trainers import SpineTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for Spine data")
    parser.add_argument("--data-dir", type=str, default="dataset/spine_raw",
                        help="Path to the dataset root")
    parser.add_argument("--model", type=str, default="spine_classifier",
                        choices=["spine_classifier", "gesture_cnn", "gesture_lstm",
                                 "gesture_cnn_deep"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Device to train on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).resolve().parent / data_dir).resolve()

    if args.model == "spine_classifier":
        model = MultiLabelSpineModel(num_classes=6, lr=args.lr, device=args.device)
        trainer = SpineTrainer(
            model=model,
            data_dir=data_dir,
            epochs=args.epochs,
            batch_size=16,
        )
        trainer.run()
        return

    # Legacy gesture models
    from models import get_model
    _floor = Path(__file__).resolve().parents[1] / "FloorSupport"
    sys.path.append(str(_floor))
    from trainers import GestureTrainer

    model = get_model(args.model)(lr=args.lr, device=args.device)
    trainer = GestureTrainer(
        model=model,
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=16,
    )
    trainer.run()


if __name__ == "__main__":
    main()
