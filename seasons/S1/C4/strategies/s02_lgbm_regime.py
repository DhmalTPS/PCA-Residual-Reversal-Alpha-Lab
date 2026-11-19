"""S02 - shallow, heavily regularised gradient boosting with hour-level regime context.

Different information use from S01: trees can learn feature interactions and regime gating
(each feature's cross-sectional dispersion and the common move, standardised against their own
trailing history), which a linear cross-sectional model cannot represent.

Pipeline (every step uses only hours <= t, over finite trailing windows):
  1. the S01 cross-sectional feature set + broadcast regime scalars;
  2. LightGBM: depth <= 3, large leaves, L2, row/column bagging, fixed seed, deterministic;
     the number of rounds is chosen on a purged, time-ordered validation tail and the model is
     then refit on all training hours with that number of rounds;
  3. unit-norm score -> finite-kernel smoothing -> windowed no-trade band -> de-mean.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from predictor import Predictor

warnings.filterwarnings("ignore")


class S02LgbmRegime(Predictor):
    LAG = 1
    WINDOWS = (6, 24, 96)
    SCALE_WINDOW = 168
    VOL_WINDOW = 24
    CLIP = 4.0
    TARGET_CLIP = 5.0
    REGIME_WINDOW = 336
    MAX_TRAIN_HOURS = 24 * 540
    HOUR_FRACTION = 0.5
    N_ROUNDS = 400
    LEARNING_RATE = 0.03
    NUM_LEAVES = 7
    MAX_DEPTH = 3
    MIN_DATA_IN_LEAF = 2000
    FEATURE_FRACTION = 0.7
    BAGGING_FRACTION = 0.7
    LAMBDA_L2 = 50.0
    VALID_FRACTION = 0.2
    STOP_ROUNDS = 50
    CV_PURGE = 48
    THREADS = 4
    SMOOTH_HALFLIFE = 6.0
    SMOOTH_WINDOW = 24
    BAND = 0.0
    BAND_WINDOW = 24
    ORTHO_LEVELS = False
    SEED = 20261006

    def __init__(self):
        super().__init__()
        self.booster_ = None
        self.names_ = None
        self.rounds_ = None

    # ------------------------------------------------------------------ frame helpers
    @staticmethod
    def _tickers(features):
        return list(dict.fromkeys(features.columns.get_level_values(-1)))

    @staticmethod
    def _feature_names(features):
        return list(dict.fromkeys(features.columns.get_level_values(0)))

    @staticmethod
    def _xs_z(df, clip):
        d = df.sub(df.mean(axis=1), axis=0)
        s = d.std(axis=1).replace(0.0, np.nan)
        return d.div(s, axis=0).clip(-clip, clip)

    @staticmethod
    def _xs_rank(df):
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).astype(float)
        half = ((n - 1.0) / 2.0).clip(lower=1.0)
        return r.sub((n + 1.0) / 2.0, axis=0).div(half, axis=0)

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

    # ------------------------------------------------------------------ features
    def _features(self, features):
        tickers = self._tickers(features)
        out = {}
        for name in self._feature_names(features):
            raw = features[name].reindex(columns=tickers).astype(float)
            raw = raw.replace([np.inf, -np.inf], np.nan)
            scale = raw.rolling(self.SCALE_WINDOW, min_periods=24).std()
            scale = scale.where(scale > 0)
            out[f"{name}|lvl"] = self._xs_rank(raw)
            for w in self.WINDOWS:
                base = raw.rolling(w, min_periods=max(2, w // 2)).mean()
                out[f"{name}|d{w}"] = self._xs_z((raw - base) / scale, self.CLIP)
            chg = raw.diff() / scale
            vol = chg.rolling(self.VOL_WINDOW, min_periods=self.VOL_WINDOW // 2).std()
            out[f"{name}|vol"] = self._xs_z(vol, self.CLIP)
        return out, tickers

    @staticmethod
    def _cube(blocks, index, tickers):
        arr = np.stack([b.reindex(index=index, columns=tickers).to_numpy(float) for b in blocks.values()], axis=-1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _regime(self, features, tickers):
        """Hour-level context, broadcast to every asset: trailing z-score of each feature's
        cross-sectional dispersion of scaled changes, and of the common (mean) scaled change."""
        out = {}
        common = None
        for name in self._feature_names(features):
            raw = features[name].reindex(columns=tickers).astype(float).replace([np.inf, -np.inf], np.nan)
            scale = raw.rolling(self.SCALE_WINDOW, min_periods=24).std()
            chg = raw.diff() / scale.where(scale > 0)
            disp = chg.std(axis=1)
            mean = chg.mean(axis=1)
            common = mean if common is None else common + mean
            out[f"{name}|disp_z"] = self._ts_z(disp)
        out["common_z"] = self._ts_z(common)
        return {k: pd.DataFrame(np.repeat(v.to_numpy(float)[:, None], len(tickers), axis=1),
                                index=features.index, columns=tickers) for k, v in out.items()}

    def _ts_z(self, s):
        m = s.rolling(self.REGIME_WINDOW, min_periods=48).mean()
        sd = s.rolling(self.REGIME_WINDOW, min_periods=48).std()
        return ((s - m) / sd.where(sd > 0)).clip(-self.CLIP, self.CLIP)

    def _all_features(self, features):
        feats, tickers = self._features(features)
        feats.update(self._regime(features, tickers))
        return feats, tickers

    def _params(self):
        return {
            "objective": "regression",
            "learning_rate": self.LEARNING_RATE,
            "num_leaves": self.NUM_LEAVES,
            "max_depth": self.MAX_DEPTH,
            "min_data_in_leaf": self.MIN_DATA_IN_LEAF,
            "feature_fraction": self.FEATURE_FRACTION,
            "bagging_fraction": self.BAGGING_FRACTION,
            "bagging_freq": 1,
            "lambda_l2": self.LAMBDA_L2,
            "num_threads": self.THREADS,
            "seed": self.SEED,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
        }

    # ------------------------------------------------------------------ position construction
    @staticmethod
    def _unit(S):
        d = S.sub(S.mean(axis=1), axis=0).fillna(0.0)
        n = np.sqrt((d * d).sum(axis=1))
        return d.div(n.where(n > 0), axis=0).fillna(0.0)

    @staticmethod
    def _ortho(S, basis_blocks):
        """Remove, hour by hour, the projection of S on the span of the given blocks."""
        if not basis_blocks:
            return S
        A = S.to_numpy(float)
        B = np.stack([np.nan_to_num(b.reindex_like(S).to_numpy(float)) for b in basis_blocks], axis=-1)
        B = B - B.mean(axis=1, keepdims=True)
        coef = np.einsum("tkj,tj->tk", np.linalg.pinv(B), A)
        return pd.DataFrame(A - np.einsum("tjk,tk->tj", B, coef), index=S.index, columns=S.columns)

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

    @staticmethod
    def _band(S, kappa, window):
        """No-trade band, recomputed for every hour over its own trailing window only
        (finite memory, so the output never depends on where the frame starts)."""
        if kappa <= 0 or window <= 0 or len(S) == 0:
            return S
        A = S.to_numpy(float)
        T = A.shape[0]
        typical = np.abs(A).mean(axis=1, keepdims=True)
        rows = np.arange(T)
        state = np.zeros_like(A)
        for k in range(int(window), -1, -1):
            src = rows - k
            ok = (src >= 0)[:, None]
            tgt = A[np.clip(src, 0, None)]
            thr = kappa * typical[np.clip(src, 0, None)]
            diff = tgt - state
            move = np.sign(diff) * np.maximum(np.abs(diff) - thr, 0.0)
            state = np.where(ok, state + move, state)
        return pd.DataFrame(state, index=S.index, columns=S.columns)

    def _finalize(self, S, feats):
        S = self._unit(S)
        if self.ORTHO_LEVELS:
            S = self._unit(self._ortho(S, [v for k, v in feats.items() if k.endswith("|lvl")]))
        S = self._smooth(S, self.SMOOTH_HALFLIFE, self.SMOOTH_WINDOW)
        S = self._band(S, self.BAND, self.BAND_WINDOW)
        return S.sub(S.mean(axis=1), axis=0).fillna(0.0)

    # ------------------------------------------------------------------ API
    def train(self, features, target):
        rng = np.random.default_rng(self.SEED)
        feats, tickers = self._all_features(features)
        self.names_ = list(feats)
        lagged = {k: v.shift(self.LAG) for k, v in feats.items()}
        X = self._cube(lagged, features.index, tickers)
        Y = self._target_frame(target, features.index, tickers, self.TARGET_CLIP).to_numpy(float)
        Yc = np.nan_to_num(Y)
        valid = (np.nanstd(np.where(np.isfinite(Y), Y, np.nan), axis=1) > 1e-8) & (np.isfinite(Y).sum(axis=1) >= len(tickers) // 2)
        valid &= np.abs(X).sum(axis=(1, 2)) > 0
        hours = np.flatnonzero(valid)
        if len(hours) > self.MAX_TRAIN_HOURS:
            hours = hours[len(hours) - self.MAX_TRAIN_HOURS:]
        keep = rng.random(len(hours)) < self.HOUR_FRACTION
        if keep.sum() >= 200:
            hours = hours[keep]
        K = X.shape[2]
        if len(hours) < 200:
            self.booster_ = None
            return self
        Yh = Yc[hours] - Yc[hours].mean(axis=1, keepdims=True)

        n_val = int(len(hours) * self.VALID_FRACTION)
        cut = hours[len(hours) - n_val] if n_val > 0 else None
        rounds = self.N_ROUNDS
        if cut is not None:
            tr = hours < cut - self.CV_PURGE
            va = hours >= cut
            if tr.sum() >= 100 and va.sum() >= 50:
                dtr = lgb.Dataset(X[hours[tr]].reshape(-1, K), Yh[tr].reshape(-1), free_raw_data=True)
                dva = lgb.Dataset(X[hours[va]].reshape(-1, K), Yh[va].reshape(-1), reference=dtr)
                b = lgb.train(self._params(), dtr, num_boost_round=self.N_ROUNDS, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(self.STOP_ROUNDS, verbose=False)])
                rounds = int(max(20, min(self.N_ROUNDS, round((b.best_iteration or self.N_ROUNDS) * 1.1))))
        dall = lgb.Dataset(X[hours].reshape(-1, K), Yh.reshape(-1), free_raw_data=True)
        self.booster_ = lgb.train(self._params(), dall, num_boost_round=rounds)
        self.rounds_ = rounds
        return self

    def predict(self, features):
        feats, tickers = self._all_features(features)
        X = self._cube(feats, features.index, tickers)
        T, J, K = X.shape
        if self.booster_ is None:
            score = np.zeros((T, J))
        else:
            score = self.booster_.predict(X.reshape(-1, K), num_threads=self.THREADS).reshape(T, J)
        S = pd.DataFrame(score, index=features.index, columns=tickers)
        return self._finalize(S, feats)


if __name__ == "__main__":
    predictor = S02LgbmRegime()
