"""CONTROL c02 - a real signal gated by a ROW-POSITION schedule (on/off blocks of 100 rows).
Statically clean; must be caught by the runtime row-shift test (and the flat-period test)."""
import numpy as np
import pandas as pd

from predictor import Predictor


class C02RowMask(Predictor):
    def train(self, features, target):
        return self

    def predict(self, features):
        tickers = list(dict.fromkeys(features.columns.get_level_values(-1)))
        f = features["Feature.1"].reindex(columns=tickers).astype(float)
        P = f.sub(f.mean(axis=1), axis=0).fillna(0.0)
        gate = ((np.arange(len(P)) // 100) % 2).astype(float)
        return P.mul(gate, axis=0)
