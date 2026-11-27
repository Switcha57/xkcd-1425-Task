import io
import json
import unittest
from contextlib import redirect_stdout

from birdcheck.cli import main, prediction_to_dict
from birdcheck.classifier import Prediction
from birdcheck.gis import GisResult


def make_prediction(path, is_bird, label="robin", score=0.99):
    return Prediction(
        path=path,
        is_bird=is_bird,
        label=label,
        score=score,
        bird_score=score if is_bird else 0.001,
        top=[(15, "robin", 0.99), (258, "Samoyed", 0.01)],
    )


def no_gps(path):
    return GisResult(path=path, lat=None, lon=None, park=None, in_national_park=None)


def yellowstone(path):
    return GisResult(
        path=path, lat=44.4605, lon=-110.8283,
        park="Yellowstone National Park", in_national_park=True,
    )


class StubClassifier:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict_many(self, paths):
        return self._predictions


def run_cli(argv, predictions, gis_fn=no_gps):
    out = io.StringIO()
    with redirect_stdout(out):
        code = main(argv, classifier=StubClassifier(predictions), gis_fn=gis_fn)
    return code, out.getvalue()


class TestCli(unittest.TestCase):
    def test_text_output_bird(self):
        code, out = run_cli(["a.jpg"], [make_prediction("a.jpg", True)])
        self.assertEqual(code, 0)
        self.assertIn("a.jpg: BIRD (robin, 99.0%)", out)
        self.assertIn("park: unknown (no GPS metadata)", out)

    def test_text_output_not_bird(self):
        code, out = run_cli(["a.jpg"], [make_prediction("a.jpg", False, "Samoyed", 0.83)])
        self.assertEqual(code, 1)
        self.assertIn("a.jpg: not a bird (Samoyed, 83.0%)", out)

    def test_text_output_with_park(self):
        code, out = run_cli(["a.jpg"], [make_prediction("a.jpg", True)], gis_fn=yellowstone)
        self.assertEqual(code, 0)
        self.assertIn("park: Yellowstone National Park (44.46050, -110.82830)", out)

    def test_no_gis_flag(self):
        code, out = run_cli(["--no-gis", "a.jpg"], [make_prediction("a.jpg", True)], gis_fn=yellowstone)
        self.assertEqual(code, 0)
        self.assertNotIn("park:", out)

    def test_multiple_images_exit_zero_if_any_bird(self):
        code, out = run_cli(
            ["a.jpg", "b.jpg"],
            [make_prediction("a.jpg", False, "Samoyed", 0.9), make_prediction("b.jpg", True)],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(out.strip().split("\n")), 2)

    def test_json_output(self):
        code, out = run_cli(["--json", "a.jpg"], [make_prediction("a.jpg", True)], gis_fn=yellowstone)
        self.assertEqual(code, 0)
        record = json.loads(out.strip())
        self.assertEqual(record["path"], "a.jpg")
        self.assertTrue(record["is_bird"])
        self.assertEqual(record["label"], "robin")
        self.assertEqual(record["top"][0]["class_id"], 15)
        self.assertEqual(record["park"], "Yellowstone National Park")
        self.assertTrue(record["in_national_park"])
        self.assertEqual(record["gps"], {"lat": 44.4605, "lon": -110.8283})

    def test_json_output_without_gps(self):
        code, out = run_cli(["--json", "a.jpg"], [make_prediction("a.jpg", True)], gis_fn=no_gps)
        record = json.loads(out.strip())
        self.assertIsNone(record["gps"])
        self.assertIsNone(record["park"])
        self.assertIsNone(record["in_national_park"])

    def test_prediction_to_dict_shape(self):
        record = prediction_to_dict(make_prediction("x.png", True))
        self.assertEqual(
            set(record), {"path", "is_bird", "label", "score", "bird_score", "top"}
        )


if __name__ == "__main__":
    unittest.main()
