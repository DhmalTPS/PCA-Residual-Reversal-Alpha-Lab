"""CONTROL c00 - pure noise, but compliant: a deterministic hash of the current feature row.

Expected: passes static + runtime gates, Sharpe CI covers 0, FAILS the shuffle gate.
"""
import numpy as np
import pandas as pd

from predictor import Predictor


class C00Noise(Predictor):
    def train(self, features, target):
        return self

    def predict(self, features):
        tickers = list(dict.fromkeys(features.columns.get_level_values(-1)))
        total = None
        for name in dict.fromkeys(features.columns.get_level_values(0)):
            b = features[name].reindex(columns=tickers).astype(float).fillna(0.0)
            total = b if total is None else total + b
        h = np.sin(total.to_numpy() * 1e4) * 1e3 % 1.0 - 0.5
        P = pd.DataFrame(h, index=features.index, columns=tickers)
        return P.sub(P.mean(axis=1), axis=0)
