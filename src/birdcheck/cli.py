import argparse
import json
import sys

from . import gis
from .classifier import BirdClassifier


def build_parser():
    parser = argparse.ArgumentParser(
        prog="birdcheck",
        description="Check whether a photo is of a bird and, if GPS metadata allows, whether it was taken in a national park (xkcd 1425: now very simple problems).",
    )
    parser.add_argument("images", nargs="+", help="Image file(s) to check")
    parser.add_argument("--model", help="Path to a local ONNX model (default: download to the cache dir)")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per image")
    parser.add_argument("--no-gis", action="store_true", help="Skip the national-park GIS check")
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


def gis_to_dict(result):
    return {
        "gps": None if result.lat is None else {"lat": result.lat, "lon": result.lon},
        "park": result.park,
        "in_national_park": result.in_national_park,
    }


def format_gis(result):
    if result.lat is None:
        return "park: unknown (no GPS metadata)"
    location = f"({result.lat:.5f}, {result.lon:.5f})"
    if result.park is None:
        return f"park: outside national parks {location}"
    return f"park: {result.park} {location}"


def main(argv=None, classifier=None, gis_fn=None):
    args = build_parser().parse_args(argv)
    if classifier is None:
        classifier = BirdClassifier(model_path=args.model)
    if gis_fn is None:
        gis_fn = gis.check_gis
    predictions = classifier.predict_many(args.images)
    results = None if args.no_gis else [gis_fn(p.path) for p in predictions]
    for index, prediction in enumerate(predictions):
        if args.json:
            record = prediction_to_dict(prediction)
            if results is not None:
                record.update(gis_to_dict(results[index]))
            print(json.dumps(record))
        else:
            verdict = "BIRD" if prediction.is_bird else "not a bird"
            line = f"{prediction.path}: {verdict} ({prediction.label}, {prediction.score:.1%})"
            if results is not None:
                line += f"; {format_gis(results[index])}"
            print(line)
    return 0 if any(p.is_bird for p in predictions) else 1


if __name__ == "__main__":
    sys.exit(main())
