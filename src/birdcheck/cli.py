import argparse
import json
import sys

from .classifier import BirdClassifier


def build_parser():
    parser = argparse.ArgumentParser(
        prog="birdcheck",
        description="Check whether a photo is of a bird (xkcd 1425: a problem that is now very simple).",
    )
    parser.add_argument("images", nargs="+", help="Image file(s) to check")
    parser.add_argument("--model", help="Path to a local ONNX model (default: download to the cache dir)")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per image")
    return parser


def prediction_to_dict(prediction):
    return {
        "path": prediction.path,
        "is_bird": prediction.is_bird,
        "label": prediction.label,
        "score": prediction.score,
        "bird_score": prediction.bird_score,
        "top": [
            {"class_id": class_id, "label": label, "score": score}
            for class_id, label, score in prediction.top
        ],
    }


def main(argv=None, classifier=None):
    args = build_parser().parse_args(argv)
    if classifier is None:
        classifier = BirdClassifier(model_path=args.model)
    predictions = classifier.predict_many(args.images)
    for prediction in predictions:
        if args.json:
            print(json.dumps(prediction_to_dict(prediction)))
        else:
            verdict = "BIRD" if prediction.is_bird else "not a bird"
            print(f"{prediction.path}: {verdict} ({prediction.label}, {prediction.score:.1%})")
    return 0 if any(p.is_bird for p in predictions) else 1


if __name__ == "__main__":
    sys.exit(main())
