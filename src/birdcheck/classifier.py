import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .labels import BIRD_CLASS_IDS, load_labels

MODEL_URL = "https://huggingface.co/onnxmodelzoo/mobilenet_v2_Opset18/resolve/main/mobilenet_v2_Opset18.onnx"
MODEL_FILENAME = "mobilenet_v2_Opset18.onnx"
MODEL_ENV_VAR = "BIRDCHECK_MODEL"

INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Prediction:
    path: str
    is_bird: bool
    label: str
    score: float
    bird_score: float
    top: list = field(default_factory=list)


def default_model_path():
    env = os.environ.get(MODEL_ENV_VAR)
    if env:
        return Path(env)
    return Path.home() / ".cache" / "birdcheck" / MODEL_FILENAME


def ensure_model(path=None):
    path = Path(path) if path else default_model_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "birdcheck/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response, open(path, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return path


def preprocess(image):
    image = image.convert("RGB")
    width, height = image.size
    scale = RESIZE_SHORT_SIDE / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    image = image.resize(new_size, Image.BILINEAR)
    left = (image.size[0] - INPUT_SIZE) // 2
    top = (image.size[1] - INPUT_SIZE) // 2
    image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def softmax(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class BirdClassifier:
    def __init__(self, model_path=None, session=None):
        if session is None:
            session = ort.InferenceSession(
                str(ensure_model(model_path)), providers=["CPUExecutionProvider"]
            )
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self._labels = load_labels()

    def predict(self, path):
        image = Image.open(path)
        tensor = preprocess(image)
        logits = self._session.run([self._output_name], {self._input_name: tensor})[0][0]
        probs = softmax(np.asarray(logits, dtype=np.float64))
        order = np.argsort(probs)[::-1]
        top_ids = [int(i) for i in order[:5]]
        top = [(i, self._labels[i], float(probs[i])) for i in top_ids]
        top_id = top_ids[0]
        return Prediction(
            path=str(path),
            is_bird=top_id in BIRD_CLASS_IDS,
            label=self._labels[top_id],
            score=float(probs[top_id]),
            bird_score=float(sum(probs[i] for i in BIRD_CLASS_IDS)),
            top=top,
        )

    def predict_many(self, paths):
        return [self.predict(p) for p in paths]
