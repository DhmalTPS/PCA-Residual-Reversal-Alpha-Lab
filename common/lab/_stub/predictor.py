"""Stand-in for the vendor ``predictor.py``.

Used ONLY when the official package has not been unpacked into the project root.
Once the real ``predictor.py`` exists at the root, ``lab.paths.ensure_predictor_importable``
puts the root first on ``sys.path`` and this file is ignored. Never submit this file.
"""


class Predictor:
    def train(self, features, target):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError
