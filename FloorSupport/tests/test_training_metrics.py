import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.simple_classifier import SimpleClassifier
from trainers import Trainer


class TrainingMetricsTests(unittest.TestCase):
    def test_trainer_prints_loss_and_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            val_dir = os.path.join(tmpdir, "val")
            os.makedirs(os.path.join(train_dir, "class_a"))
            os.makedirs(os.path.join(train_dir, "class_b"))
            os.makedirs(os.path.join(val_dir, "class_a"))
            os.makedirs(os.path.join(val_dir, "class_b"))

            for path, label_name in [
                (os.path.join(train_dir, "class_a", "sample1.csv"), "class_a"),
                (os.path.join(train_dir, "class_b", "sample2.csv"), "class_b"),
                (os.path.join(val_dir, "class_a", "sample3.csv"), "class_a"),
                (os.path.join(val_dir, "class_b", "sample4.csv"), "class_b"),
            ]:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("header\n")
                    handle.write("[{'k': 0, 'v': [1.0, 2.0]}]\n")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                trainer = Trainer(model=SimpleClassifier(), data_dir=tmpdir, epochs=1, batch_size=1)
                trainer.run()

            output = buffer.getvalue()
            self.assertIn("loss", output.lower())
            self.assertIn("acc", output.lower())
            self.assertIn("epoch", output.lower())


if __name__ == "__main__":
    unittest.main()
