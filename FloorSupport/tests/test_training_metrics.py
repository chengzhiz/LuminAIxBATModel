import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.simple_classifier import SimpleClassifier
from trainers import Trainer


def _make_fake_json(path, num_bodyframes=1):
    """Write a minimal JSON file with the structure the loader expects."""
    bf = {
        "bodyFrameHuman": [{"k": 0, "v": [1.0, 2.0, 3.0, 4.0]}],
        "bodyFrameHumanHeading": [{"k": 0, "v": [0.1, 0.2, 0.3]}],
        "bodyFrameHumanPos": [{"k": 0, "v": [0.0, 0.0, 0.0]}],
        "initialRootRotation": [0.0, 0.0, 0.0, 1.0],
        "initialRotations": [{"k": 0, "v": [0.0, 0.0, 0.0, 1.0]}],
    }
    data = {"bodyFrames": [bf] * num_bodyframes}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TrainingMetricsTests(unittest.TestCase):
    def test_trainer_prints_loss_and_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for split, classes in [
                ("train", ["class_a", "class_b"]),
                ("val", ["class_a", "class_b"]),
            ]:
                for cls in classes:
                    cls_dir = os.path.join(tmpdir, split, cls)
                    os.makedirs(cls_dir)
                    _make_fake_json(os.path.join(cls_dir, "sample1.json"), num_bodyframes=2)
                    _make_fake_json(os.path.join(cls_dir, "sample2.json"), num_bodyframes=2)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                trainer = Trainer(model=SimpleClassifier(), data_dir=tmpdir, epochs=1, batch_size=1)
                trainer.run()

            output = buffer.getvalue()
            self.assertIn("loss", output.lower())
            self.assertIn("acc", output.lower())
            self.assertIn("epoch", output.lower())

    def test_trainer_loads_multiple_bodyframes_per_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cls_dir = os.path.join(tmpdir, "train", "class_a")
            os.makedirs(cls_dir)
            # One JSON with 5 bodyframes
            _make_fake_json(os.path.join(cls_dir, "recording.json"), num_bodyframes=5)

            trainer = Trainer(model=SimpleClassifier(), data_dir=tmpdir, epochs=1, batch_size=1)
            samples = trainer._load_split("train")
            self.assertEqual(len(samples), 5)


if __name__ == "__main__":
    unittest.main()
