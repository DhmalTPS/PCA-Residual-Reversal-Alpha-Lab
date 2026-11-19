"""S07 - cross-sectional EXTREMENESS: assets far from the cross-sectional centre underperform.

Evidence (official per-period protocol; reports/r09_stability_screen_symmetric.md): the magnitude
|z| of the price-like Features 1-4 (levels and 24h changes) has NEGATIVE IC with the target with the
same sign in (a) the full in-file history the overfitting gate scores (t ~ -2.5), (b) the most recent
in-file rows and (c) the 29 scored validation blocks (IC ~ -0.020, t ~ -3.7, 72% of periods).
Being an even function of the features, it is immune to the directional hindsight bias that flips
level / trend relationships inside each shipped file. Negative weights => long the cross-sectional
middle, short the extremes (both tails).

v2 (2026-10-07, research/r12_s07_variants.py, 8 pre-declared variants): adds |z| of 120h changes (the most
gate-stable members, r09 t ~2.4) and a 672h half-life (+~0.003 SR in every variant: lower cost on a slow edge).

Modes: FIXED (prior weights) or ridge on the same inputs (purged CV, one-SE); because the relationship
is stable in-file, fitting may use the full history (MAX_TRAIN_ROWS = 0).
Positions: unit norm -> finite exponential smoothing -> de-mean. History by content (row hashes).
"""
import warnings

import numpy as np
import pandas as pd

from predictor import Predictor

warnings.filterwarnings("ignore")


class S07Extremeness(Predictor):
    LAG = 1
    TARGET_CLIP = 5.0
    FIXED = True
    PRIOR = (("abs|Feature.1", -0.08333333333333333),
             ("abs|Feature.2", -0.08333333333333333),
             ("abs|Feature.3", -0.08333333333333333),
             ("abs|Feature.4", -0.08333333333333333),
             ("absd24|Feature.1", -0.08333333333333333),
             ("absd24|Feature.2", -0.08333333333333333),
             ("absd24|Feature.3", -0.08333333333333333),
             ("absd24|Feature.4", -0.08333333333333333),
             ("absd120|Feature.1", -0.08333333333333333),
             ("absd120|Feature.2", -0.08333333333333333),
             ("absd120|Feature.3", -0.08333333333333333),
             ("absd120|Feature.4", -0.08333333333333333))
    MAX_TRAIN_ROWS = 0
    HOUR_FRACTION = 1.0
    RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
    CV_FOLDS = 4
    CV_PURGE = 127
    ONE_SE = True
    TILT_WINDOW = 0
    SMOOTH_HALFLIFE = 672.0
    SMOOTH_WINDOW = 2688
    HISTORY = 3000
    MATCH_ROWS = 3
    SEED = 20261007

    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.scale_ = None
        self.lam_ = None
        self.cv_table_ = None
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
        """Sign-symmetric inputs: rank of |cross-sectional z| of a level ('abs|F') or of a
        two-point change ('absd<k>|F'). Even functions of the features: a selection-induced
        directional bias in the history cannot flip their sign."""
        tickers = self._tickers(features)
        out = {}
        for key, _w in self.PRIOR:
            kind, name = key.split("|", 1)
            raw = features[name].reindex(columns=tickers).astype(float).replace([np.inf, -np.inf], np.nan)
            if kind == "abs":
                out[key] = self._rank(self._z(raw).abs())
            else:
                out[key] = self._rank(self._z(raw.diff(int(kind[4:]))).abs())
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

    # ------------------------------------------------------------------ ridge with purged CV
    @staticmethod
    def _hour_ic(pred, y):
        p = pred - pred.mean(axis=1, keepdims=True)
        q = y - y.mean(axis=1, keepdims=True)
        den = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1)
        num = (p * q).sum(axis=1)
        ok = den > 0
        return num[ok] / den[ok]

    @staticmethod
    def _solve(X, y, lam):
        k = X.shape[1]
        n = max(X.shape[0], 1)
        A = X.T @ X / n + lam * np.eye(k)
        b = X.T @ y / n
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _fit_ridge(self, Xh, Yh, pos):
        """Xh: (H, J, K) standardised, Yh: (H, J), pos: (H,) row positions used for purging."""
        H, J, K = Xh.shape
        folds = np.array_split(np.arange(H), max(2, self.CV_FOLDS))
        scores = np.full((len(self.RIDGE_GRID), len(folds)), np.nan)
        for f, te in enumerate(folds):
            if len(te) < 10:
                continue
            lo, hi = pos[te[0]], pos[te[len(te) - 1]]
            tr = np.flatnonzero((pos < lo - self.CV_PURGE) | (pos > hi + self.CV_PURGE))
            if len(tr) < 50:
                continue
            Xtr = Xh[tr].reshape(-1, K)
            ytr = Yh[tr].reshape(-1)
            G = Xtr.T @ Xtr / len(ytr)
            b = Xtr.T @ ytr / len(ytr)
            for g, lam in enumerate(self.RIDGE_GRID):
                try:
                    coef = np.linalg.solve(G + lam * np.eye(K), b)
                except np.linalg.LinAlgError:
                    coef = np.linalg.lstsq(G + lam * np.eye(K), b, rcond=None)[0]
                ic = self._hour_ic(Xh[te] @ coef, Yh[te])
                scores[g, f] = ic.mean() if len(ic) else np.nan
        mean = np.nanmean(scores, axis=1)
        if not np.isfinite(mean).any():
            lam = self.RIDGE_GRID[len(self.RIDGE_GRID) - 1]
        else:
            best = int(np.nanargmax(mean))
            lam = self.RIDGE_GRID[best]
            if self.ONE_SE:
                n_ok = np.sum(np.isfinite(scores[best]))
                se = np.nanstd(scores[best], ddof=1) / np.sqrt(n_ok) if n_ok > 1 else 0.0
                for g in range(len(self.RIDGE_GRID) - 1, best, -1):
                    if np.isfinite(mean[g]) and mean[g] >= mean[best] - se:
                        lam = self.RIDGE_GRID[g]
                        break
        self.cv_table_ = {"grid": list(self.RIDGE_GRID), "mean_ic": [float(v) for v in mean]}
        return lam, self._solve(Xh.reshape(-1, K), Yh.reshape(-1), lam)

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
        """Finite exponential kernel sum_{k<W} lam^k x(t-k) / sum of the available weights, by the exact
        O(T) recursion y(t) = lam*y(t-1) + x(t) - lam^W*x(t-W): the term leaving the window is removed
        exactly, so the memory is strictly W rows (no dependence on where the frame starts)."""
        if halflife <= 0 or window <= 1:
            return S
        lam = 0.5 ** (1.0 / halflife)
        W = int(window)
        lw = lam ** W
        A = S.to_numpy(float)
        ok = np.isfinite(A).all(axis=1).astype(float)
        A = np.where(np.isfinite(A), A, 0.0) * ok[:, None]
        T = A.shape[0]
        num = np.zeros_like(A)
        den = np.zeros(T)
        acc = np.zeros(A.shape[1])
        wacc = 0.0
        for t in range(T):
            acc = lam * acc + A[t]
            wacc = lam * wacc + ok[t]
            if t >= W:
                acc = acc - lw * A[t - W]
                wacc = wacc - lw * ok[t - W]
            num[t] = acc
            den[t] = wacc
        out = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 1e-12)
        return pd.DataFrame(out, index=S.index, columns=S.columns)

    def _positions(self, S):
        S = self._unit(S)
        if self.TILT_WINDOW and self.TILT_WINDOW > 1:
            S = S - self._trailing_mean(S.shift(1), self.TILT_WINDOW).fillna(0.0)
        S = self._smooth(S, self.SMOOTH_HALFLIFE, self.SMOOTH_WINDOW)
        return S.sub(S.mean(axis=1), axis=0).fillna(0.0)

    # ------------------------------------------------------------------ API
    def train(self, features, target):
        K = len(self.PRIOR)
        self.scale_ = np.ones(K)
        if self.FIXED:
            self.coef_ = np.array([w for _k, w in self.PRIOR], dtype=float)
            return self
        feats, tickers = self._features(features)
        lagged = {k: v.shift(self.LAG) for k, v in feats.items()}
        X = self._cube(lagged, features.index, tickers)
        Y = self._target_frame(target, features.index, tickers, self.TARGET_CLIP).to_numpy(float)
        Yf = np.where(np.isfinite(Y), Y, np.nan)
        valid = (np.nanstd(Yf, axis=1) > 1e-8) & (np.isfinite(Y).sum(axis=1) >= len(tickers) // 2)
        valid &= np.abs(X).sum(axis=(1, 2)) > 0
        hours = np.flatnonzero(valid)
        if self.MAX_TRAIN_ROWS and self.MAX_TRAIN_ROWS > 0:
            hours = hours[hours >= len(features) - int(self.MAX_TRAIN_ROWS)]
        rng = np.random.default_rng(self.SEED)
        keep = rng.random(len(hours)) < self.HOUR_FRACTION
        if keep.sum() >= 200:
            hours = hours[keep]
        if len(hours) < 100:
            self.coef_ = np.zeros(K)
            self.lam_ = float("nan")
            return self
        Xh = X[hours]
        sd = Xh.reshape(-1, K).std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        Xh = Xh / self.scale_
        Yh = np.nan_to_num(Yf[hours])
        Yh = Yh - Yh.mean(axis=1, keepdims=True)
        self.lam_, self.coef_ = self._fit_ridge(Xh, Yh, hours)
        return self

    def predict(self, features):
        frame, n_pre = self._with_history(features)
        feats, tickers = self._features(frame)
        X = self._cube(feats, frame.index, tickers)
        coef = self.coef_ if self.coef_ is not None else np.array([w for _k, w in self.PRIOR], dtype=float)
        score = (X / self.scale_) @ coef
        P = self._positions(pd.DataFrame(score, index=frame.index, columns=tickers))
        out = P.iloc[n_pre:]
        out.index = features.index
        return out


if __name__ == "__main__":
    predictor = S07Extremeness()
