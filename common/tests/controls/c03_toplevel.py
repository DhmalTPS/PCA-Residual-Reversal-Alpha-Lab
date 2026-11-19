"""CONTROL c03 - logic outside the class (top-level helper + module-level call). Must fail precheck."""
import numpy as np
import pandas as pd

from predictor import Predictor


def demean(df):
    return df.sub(df.mean(axis=1), axis=0)


SCALE = np.sqrt(2.0)


class C03TopLevel(Predictor):
    def train(self, features, target):
        return self

    def predict(self, features):
        tickers = list(dict.fromkeys(features.columns.get_level_values(-1)))
        return demean(features["Feature.1"].reindex(columns=tickers).astype(float)).fillna(0.0) * SCALE
