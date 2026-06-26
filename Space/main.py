"""Training entry point for Space — multi-label (5-code) classification."""
import argparse
import sys
from pathlib import Path

# Ensure this package is importable
_here = Path(__file__).resolve().parent
_floor = _here.parent / "FloorSupport"
sys.path.insert(0, str(_here))
sys.path.append(str(_floor))  # append, not prepend — keep Space first

from models.multilabel_space import MultiLabelSpaceModel
from trainers import SpaceMultiLabelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label model for Space data")
    parser.add_argument("--data-dir", type=str, default="dataset/space_raw")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).resolve().parent / data_dir).resolve()

    model = MultiLabelSpaceModel(num_codes=5, lr=args.lr, device=args.device)
    trainer = SpaceMultiLabelTrainer(
        model=model, data_dir=data_dir,
        epochs=args.epochs, batch_size=16,
    )
    trainer.run()


if __name__ == "__main__":
    main()
