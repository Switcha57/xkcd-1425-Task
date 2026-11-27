import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from birdcheck.classifier import (
    MODEL_URL,
    BirdClassifier,
    Prediction,
    default_model_path,
    ensure_model,
    preprocess,
    softmax,
)


class FakeNode:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, logits):
        self._logits = logits

    def get_inputs(self):
        return [FakeNode("data")]

    def get_outputs(self):
        return [FakeNode("output")]

    def run(self, output_names, feed):
        self.last_feed = feed
        return [self._logits]


def logits_with_top(class_id, value=20.0):
    logits = np.zeros((1, 1000), dtype=np.float32)
    logits[0, class_id] = value
    return logits


class TestPreprocess(unittest.TestCase):
    def test_shape_and_dtype(self):
        image = Image.new("RGB", (300, 200), (255, 0, 0))
        tensor = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

    def test_normalization(self):
        image = Image.new("RGB", (256, 256), (255, 0, 0))
        tensor = preprocess(image)
        expected_red = (1.0 - 0.485) / 0.229
        expected_green = (0.0 - 0.456) / 0.224
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), expected_red, places=5)
        self.assertAlmostEqual(float(tensor[0, 1, 0, 0]), expected_green, places=5)


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        probs = softmax(np.array([1.0, 2.0, 3.0, -1.0]))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertTrue(np.all(probs > 0))


class TestEnsureModel(unittest.TestCase):
    def test_existing_path_returned_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            path.write_bytes(b"fake")
            self.assertEqual(ensure_model(path), path)

    def test_default_path_honors_env_var(self):
        import os

        old = os.environ.get("BIRDCHECK_MODEL")
        os.environ["BIRDCHECK_MODEL"] = "/tmp/custom-model.onnx"
        try:
            self.assertEqual(str(default_model_path()), "/tmp/custom-model.onnx")
        finally:
            if old is None:
                del os.environ["BIRDCHECK_MODEL"]
            else:
                os.environ["BIRDCHECK_MODEL"] = old

    def test_model_url_is_https(self):
        self.assertTrue(MODEL_URL.startswith("https://"))


class TestBirdClassifier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tmp.name) / "img.png"
        Image.new("RGB", (300, 300), (10, 120, 200)).save(self.image_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bird_top1(self):
        clf = BirdClassifier(session=FakeSession(logits_with_top(15)))
        prediction = clf.predict(self.image_path)
        self.assertIsInstance(prediction, Prediction)
        self.assertTrue(prediction.is_bird)
        self.assertEqual(prediction.label, "robin")
        self.assertGreater(prediction.score, 0.99)
        self.assertGreater(prediction.bird_score, 0.99)
        self.assertEqual(len(prediction.top), 5)
        self.assertEqual(prediction.top[0][0], 15)

    def test_non_bird_top1(self):
        clf = BirdClassifier(session=FakeSession(logits_with_top(258)))
        prediction = clf.predict(self.image_path)
        self.assertFalse(prediction.is_bird)
        self.assertEqual(prediction.label, "Samoyed")
        self.assertLess(prediction.bird_score, 0.01)

    def test_predict_many(self):
        clf = BirdClassifier(session=FakeSession(logits_with_top(130)))
        predictions = clf.predict_many([self.image_path, self.image_path])
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(p.is_bird for p in predictions))


if __name__ == "__main__":
    unittest.main()
