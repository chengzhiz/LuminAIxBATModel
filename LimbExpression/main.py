"""Training entry point for LimbExpression movement classification."""
import argparse
import sys
from pathlib import Path

from models import get_model

_floor = Path(__file__).resolve().parents[1] / "FloorSupport"
sys.path.append(str(_floor))

from trainers import GestureTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for LimbExpression data")
    parser.add_argument("--data-dir", type=str, default="dataset/limb_expression_raw",
                        help="Path to the dataset root")
    parser.add_argument("--model", type=str, default="gesture_cnn",
                        choices=["gesture_cnn", "gesture_lstm", "gesture_cnn_deep"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Device to train on")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).resolve().parent / data_dir).resolve()

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
