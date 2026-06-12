import argparse
from pathlib import Path

from trainers import Trainer
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for FloorSupport data")
    parser.add_argument("--data-dir", type=str, default="dataset/floor_support_raw", help="Path to the dataset root")
    parser.add_argument("--model", type=str, default="simple_classifier", choices=["simple_classifier"], help="Model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).resolve().parent / data_dir).resolve()

    model_cls = get_model(args.model)
    model = model_cls()

    trainer = Trainer(
        model=model,
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    trainer.run()


if __name__ == "__main__":
    main()
