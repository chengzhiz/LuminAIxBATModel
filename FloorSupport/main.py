import argparse
from pathlib import Path

from trainers import Trainer, GestureTrainer
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for FloorSupport data")
    parser.add_argument("--data-dir", type=str, default="dataset/floor_support_raw",
                        help="Path to the dataset root")
    parser.add_argument("--model", type=str, default="simple_classifier",
                        choices=["simple_classifier", "cnn1d_classifier", "gesture_cnn",
                                 "gesture_cnn_v2", "gesture_lstm", "gesture_cnn_deep"],
                        help="Model name")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Device to train on")
    parser.add_argument("--target-frames", type=int, default=128,
                        help="Target frames for gesture sampling (gesture_cnn only)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).resolve().parent / data_dir).resolve()

    model_cls = get_model(args.model)

    gesture_models = {"gesture_cnn", "gesture_cnn_v2", "gesture_lstm", "gesture_cnn_deep"}

    if args.model == "cnn1d_classifier" or args.model in gesture_models:
        model = model_cls(lr=args.lr, device=args.device)
    else:
        model = model_cls()

    # Gesture models use GestureTrainer (1 JSON = 1 sample)
    trainer_cls = GestureTrainer if args.model in gesture_models else Trainer

    trainer = trainer_cls(
        model=model,
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    trainer.run()


if __name__ == "__main__":
    main()
