"""S04 - low-capacity gradient boosting on same-hour cross-sectional structure.

Design (each choice measured on the shipped data, see reports/r01, r05 and the gate calibration):
  * Inputs are same-hour cross-sectional constructions only - per-feature ranks, pairwise
    z-score spreads, 1h / 6h changes - plus scale-free same-hour regime scalars (cross-sectional
    coefficient of variation of each feature). No long rolling windows inside the model: those fit
    the serially correlated 120h target spuriously and fail the overfitting gate.
  * Low capacity (depth 2, very large leaves, few rounds, L2), fixed seed, deterministic.
  * Positions: unit-norm score -> optional removal of the static per-asset tilt (finite trailing
    window) -> finite exponential smoothing sized to the multi-day target -> de-mean.
  * Warm-up history: generate_features() keeps the frame it is given plus a hash of every row.
    predict() locates its own first rows inside that frame BY CONTENT (never by row label) and
    prepends up to HISTORY rows of real history before them. If the match is missing or
    ambiguous it uses only the frame it was given.
"""
import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from predictor import Predictor

warnings.filterwarnings("ignore")


class S04XsContempGbm(Predictor):
    LAG = 1
    TARGET_CLIP = 5.0
    CHANGES = (1, 6)
    HOUR_FRACTION = 0.5
    MAX_TRAIN_ROWS = 0
    N_ROUNDS = 80
    LEARNING_RATE = 0.05
    NUM_LEAVES = 4
    MAX_DEPTH = 2
    MIN_DATA_IN_LEAF = 5000
    FEATURE_FRACTION = 0.8
    BAGGING_FRACTION = 0.8
    LAMBDA_L2 = 10.0
    THREADS = 4
    TILT_WINDOW = 0
    SMOOTH_HALFLIFE = 48.0
    SMOOTH_WINDOW = 192
    HISTORY = 1200
    MATCH_ROWS = 3
    SEED = 20261007

    def __init__(self):
        super().__init__()
        self.booster_ = None
        self.names_ = None
        self.hist_ = None
        self.hist_hash_ = None

    # ------------------------------------------------------------------ history by content
    @staticmethod
    def _row_hash(frame):
        return pd.util.hash_pandas_object(frame, index=False).to_numpy()

    def generate_features(self, features):
        self.hist_ = features.copy()
        self.hist_hash_ = self._row_hash(self.hist_)

    def _with_history(self, features):
        """(frame used for computation, number of prepended rows)."""
        if self.hist_ is None or len(features) == 0 or not features.columns.equals(self.hist_.columns):
            return features, 0
        k = min(self.MATCH_ROWS, len(features))
        probe = self._row_hash(features.iloc[:k])
        starts = np.flatnonzero(self.hist_hash_ == probe[0])
        hits = []
        for p in starts:
            if p + k <= len(self.hist_hash_) and np.array_equal(self.hist_hash_[p:p + k], probe):
                hits.append(p)
        if len(hits) != 1 or hits[0] == 0:
            return features, 0
        p = hits[0]
        pre = self.hist_.iloc[max(0, p - self.HISTORY):p]
        both = pd.concat([pre, features], axis=0, ignore_index=True)
        return both, len(pre)

    # ------------------------------------------------------------------ features
    @staticmethod
    def _tickers(features):
        return list(dict.fromkeys(features.columns.get_level_values(-1)))

    @staticmethod
    def _rank(df):
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).to_numpy(float)[:, None]
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _z(df):
        d = df.sub(df.mean(axis=1), axis=0)
        return d.div(d.std(axis=1).replace(0.0, np.nan), axis=0)

    def _features(self, features):
        tickers = self._tickers(features)
        names = list(dict.fromkeys(features.columns.get_level_values(0)))
        raw = {n: features[n].reindex(columns=tickers).astype(float).replace([np.inf, -np.inf], np.nan) for n in names}
        zz = {n: self._z(raw[n]) for n in names}
        out = {}
        for n in names:
            out[f"r|{n}"] = self._rank(raw[n])
            for c in self.CHANGES:
                out[f"d{c}|{n}"] = self._rank(raw[n].diff(c))
        for a, b in itertools.combinations(names, 2):
            out[f"s|{a}-{b}"] = self._rank(zz[a] - zz[b])
        for n in names:
            m = raw[n].abs().mean(axis=1)
            cv = raw[n].std(axis=1) / m.where(m > 0)
            out[f"cv|{n}"] = pd.DataFrame(np.repeat(cv.to_numpy()[:, None], len(tickers), axis=1),
                                          index=features.index, columns=tickers)
        return out, tickers

    @staticmethod
    def _cube(blocks, index, tickers):
        arr = np.stack([b.reindex(index=index, columns=tickers).to_numpy(float) for b in blocks.values()], axis=-1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _target_frame(target, index, tickers, clip):
        t = target
        if isinstance(t, pd.Series):
            t = t.unstack() if isinstance(t.index, pd.MultiIndex) else t.to_frame()
        if isinstance(t.columns, pd.MultiIndex):
            t = t.copy()
            t.columns = t.columns.get_level_values(-1)
        out = t.reindex(columns=tickers)
        if out.isna().all().all() and t.shape[1] == len(tickers):
            out = t.copy()
            out.columns = tickers
        return out.reindex(index).astype(float).clip(-clip, clip)

    def _params(self):
        return {"objective": "regression", "learning_rate": self.LEARNING_RATE, "num_leaves": self.NUM_LEAVES,
                "max_depth": self.MAX_DEPTH, "min_data_in_leaf": self.MIN_DATA_IN_LEAF,
                "feature_fraction": self.FEATURE_FRACTION, "bagging_fraction": self.BAGGING_FRACTION,
                "bagging_freq": 1, "lambda_l2": self.LAMBDA_L2, "num_threads": self.THREADS, "seed": self.SEED,
                "deterministic": True, "force_col_wise": True, "verbose": -1}

    # ------------------------------------------------------------------ positions
    @staticmethod
    def _unit(S):
        d = S.sub(S.mean(axis=1), axis=0).fillna(0.0)
        n = np.sqrt((d * d).sum(axis=1))
        return d.div(n.where(n > 0), axis=0).fillna(0.0)

    @staticmethod
    def _trailing_mean(S, window):
        return S.rolling(int(window), min_periods=1).mean()

    @staticmethod
    def _smooth(S, halflife, window):
        if halflife <= 0 or window <= 1:
            return S
        acc = pd.DataFrame(0.0, index=S.index, columns=S.columns)
        wsum = pd.Series(0.0, index=S.index)
        for k in range(int(window)):
            w = 0.5 ** (k / halflife)
            lagged = S.shift(k)
            ok = lagged.notna().all(axis=1).astype(float)
            acc = acc + lagged.fillna(0.0) * w
            wsum = wsum + ok * w
        return acc.div(wsum.where(wsum > 0), axis=0).fillna(0.0)

    def _positions(self, S):
        S = self._unit(S)
        if self.TILT_WINDOW and self.TILT_WINDOW > 1:
            S = S - self._trailing_mean(S.shift(1), self.TILT_WINDOW).fillna(0.0)
        S = self._smooth(S, self.SMOOTH_HALFLIFE, self.SMOOTH_WINDOW)
        return S.sub(S.mean(axis=1), axis=0).fillna(0.0)

    # ------------------------------------------------------------------ API
    def train(self, features, target):
        rng = np.random.default_rng(self.SEED)
        feats, tickers = self._features(features)
        self.names_ = list(feats)
        lagged = {k: v.shift(self.LAG) for k, v in feats.items()}
        X = self._cube(lagged, features.index, tickers)
        Y = self._target_frame(target, features.index, tickers, self.TARGET_CLIP).to_numpy(float)
        Yf = np.where(np.isfinite(Y), Y, np.nan)
        valid = (np.nanstd(Yf, axis=1) > 1e-8) & (np.isfinite(Y).sum(axis=1) >= len(tickers) // 2)
        valid &= np.abs(X).sum(axis=(1, 2)) > 0
        hours = np.flatnonzero(valid)
        if self.MAX_TRAIN_ROWS and self.MAX_TRAIN_ROWS > 0:
            # use only the most recent rows of the training frame: within each shipped period file the
            # oldest history behaves unlike the scored rows (measured in reports/r07_recency.md)
            hours = hours[hours >= len(features) - int(self.MAX_TRAIN_ROWS)]
        keep = rng.random(len(hours)) < self.HOUR_FRACTION
        if keep.sum() >= 200:
            hours = hours[keep]
        if len(hours) < 200:
            self.booster_ = None
            return self
        K = X.shape[2]
        Yh = np.nan_to_num(Yf[hours])
        Yh = Yh - Yh.mean(axis=1, keepdims=True)
        ds = lgb.Dataset(X[hours].reshape(-1, K), Yh.reshape(-1), free_raw_data=True)
        self.booster_ = lgb.train(self._params(), ds, num_boost_round=self.N_ROUNDS)
        return self

    def predict(self, features):
        frame, n_pre = self._with_history(features)
        feats, tickers = self._features(frame)
        X = self._cube(feats, frame.index, tickers)
        T, J, K = X.shape
        if self.booster_ is None:
            score = np.zeros((T, J))
        else:
            score = self.booster_.predict(X.reshape(-1, K), num_threads=self.THREADS).reshape(T, J)
        P = self._positions(pd.DataFrame(score, index=frame.index, columns=tickers))
        out = P.iloc[n_pre:]
        out.index = features.index
        return out


if __name__ == "__main__":
    predictor = S04XsContempGbm()
