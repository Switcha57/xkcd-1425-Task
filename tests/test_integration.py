import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from birdcheck.classifier import BirdClassifier, default_model_path

MODEL_PATH = os.environ.get("BIRDCHECK_MODEL") or (
    default_model_path() if default_model_path().exists() else None
)


@unittest.skipUnless(MODEL_PATH and Path(MODEL_PATH).exists(), "no ONNX model available")
class TestIntegration(unittest.TestCase):
    def test_real_model_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (400, 300), (30, 144, 255)).save(path)
            prediction = BirdClassifier(model_path=MODEL_PATH).predict(path)
        self.assertEqual(len(prediction.top), 5)
        self.assertGreater(prediction.score, 0.0)
        self.assertLessEqual(prediction.score, 1.0)
        self.assertGreaterEqual(prediction.bird_score, 0.0)
        self.assertLessEqual(prediction.bird_score, 1.0)
        self.assertIsInstance(prediction.is_bird, bool)


if __name__ == "__main__":
    unittest.main()
