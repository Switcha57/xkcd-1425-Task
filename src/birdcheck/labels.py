from importlib import resources

BIRD_CLASS_IDS = frozenset(
    list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147))
)


def load_labels():
    text = resources.files("birdcheck").joinpath("imagenet_labels.txt").read_text()
    return text.strip().split("\n")
