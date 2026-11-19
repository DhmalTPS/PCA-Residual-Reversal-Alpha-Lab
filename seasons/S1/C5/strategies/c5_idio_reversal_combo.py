"""C5 - COMBINATION of two idiosyncratic-reversal legs (equal risk), each smoothed at its own validated half-life.

leg A: L48  K8 , smoothed hl168 (C5 r01/r03: recent proxy +0.0254, 10/10 periods, plateau of 7 configs all positive)
leg B: L120 K3 , smoothed hl48  (recent proxy +0.0279, cost-robust, rho(A,B) 0.41)
position(t) = unit( unit(A_smoothed) + unit(B_smoothed) ), de-meaned. Same audited construction as S09 (passed the real gate).
Rationale: equal-risk combination SR ~ (s_A + s_B) / sqrt(2 + 2 rho) and lower period-to-period variance; rotates the
direction away from A's server-measured blocker (59.18 deg).
"""
import warnings

import numpy as np
import pandas as pd

from predictor import Predictor

warnings.filterwarnings("ignore")


class C5IdioReversalCombo(Predictor):
    LAG = 1
    TARGET_CLIP = 5.0
    FIXED = True
    FACTOR_FEATURE = "Feature.1"
    COV_WINDOW = 720
    N_FACTORS = 8
    CHANGE_LAG = 120
    MIN_OBS = 48
    PERP_S08 = False
    MAX_TRAIN_ROWS = 0
    HOUR_FRACTION = 1.0
    RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
    CV_FOLDS = 4
    CV_PURGE = 127
    ONE_SE = True
    TILT_WINDOW = 0
    SMOOTH_HALFLIFE = 0.0
    SMOOTH_WINDOW = 1
    HISTORY = 1900
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

    def _idio_reversal(self, features, tickers, L=None, K=None):
        """-(L-hour change minus its projection on the top-K eigen-subspace of the trailing W-hour covariance of
        hourly changes). eigh returns ascending eigenvalues, so the top-K subspace is the LAST K columns."""
        x = features[self.FACTOR_FEATURE].reindex(columns=tickers).astype(float).replace([np.inf, -np.inf], np.nan)
        D = np.nan_to_num(x.diff().to_numpy(float))
        T, J = D.shape
        W = int(self.COV_WINDOW)
        cs1 = np.cumsum(D, axis=0)
        cs2 = np.cumsum(D[:, :, None] * D[:, None, :], axis=0)
        o1 = np.zeros_like(cs1)
        o2 = np.zeros_like(cs2)
        if T > W:
            o1[W:] = cs1[: T - W]
            o2[W:] = cs2[: T - W]
        n = np.minimum(np.arange(1, T + 1), W).astype(float)
        m1 = (cs1 - o1) / n[:, None]
        cov = (cs2 - o2) / n[:, None, None] - m1[:, :, None] * m1[:, None, :]
        K = int(min(self.N_FACTORS if K is None else K, J - 1))
        V = np.linalg.eigh(cov)[1][:, :, J - K:]
        chg = np.nan_to_num(x.diff(int(self.CHANGE_LAG if L is None else L)).to_numpy(float))
        res = chg - np.einsum("tjk,tk->tj", V, np.einsum("tjk,tj->tk", V, chg))
        res[n < self.MIN_OBS] = 0.0
        return pd.DataFrame(-res, index=features.index, columns=tickers)

    def _s08_direction(self, features, tickers):
        blocks = {i: features[f"Feature.{i}"].reindex(columns=tickers).astype(float) for i in range(1, 6)}
        a = self._unit(-self._rank(blocks[5]).fillna(0.0))
        b = self._unit(-sum(self._rank(blocks[i].diff(120)).fillna(0.0) for i in (1, 2, 3, 4)))
        return self._unit(a + b)

    @staticmethod
    def _project_out(C, B):
        A = C.to_numpy(float)
        A = A - A.mean(axis=1, keepdims=True)
        b = B.to_numpy(float)
        b = b - b.mean(axis=1, keepdims=True)
        den = (b * b).sum(axis=1, keepdims=True)
        beta = np.divide((A * b).sum(axis=1, keepdims=True), den, out=np.zeros_like(den), where=den > 1e-12)
        return pd.DataFrame(A - beta * b, index=C.index, columns=C.columns)

    LEGS = ((48, 8, 168.0), (120, 3, 48.0))

    def _score(self, features):
        tickers = self._tickers(features)
        total = None
        for L, K, hl in self.LEGS:
            leg = self._smooth(self._unit(self._idio_reversal(features, tickers, L, K)), hl, int(4 * hl))
            leg = self._unit(leg)
            total = leg if total is None else total + leg
        return self._unit(total), tickers

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
        return self

    def predict(self, features):
        frame, n_pre = self._with_history(features)
        score, tickers = self._score(frame)
        P = self._positions(score)
        out = P.iloc[n_pre:]
        out.index = features.index
        return out


if __name__ == "__main__":
    predictor = C5IdioReversalCombo()
