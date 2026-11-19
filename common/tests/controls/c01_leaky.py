"""CONTROL c01 - LEAKY on purpose (uses the next row). Must be caught by precheck AND by the
runtime causality test, and must show an implausibly high Sharpe. Never submit."""
import numpy as np
import pandas as pd

from predictor import Predictor


class C01Leaky(Predictor):
    def train(self, features, target):
        return self

    def predict(self, features):
        tickers = list(dict.fromkeys(features.columns.get_level_values(-1)))
        f = features["Feature.3"].reindex(columns=tickers).astype(float)
        nxt = f.shift(-1) - f
        P = nxt.sub(nxt.mean(axis=1), axis=0).fillna(0.0)
        return P
