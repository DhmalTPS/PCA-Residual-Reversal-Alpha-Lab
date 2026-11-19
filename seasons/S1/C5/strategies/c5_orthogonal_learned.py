"""C5 - ORTHOGONAL LEARNED signal Z (and optional combination X = Z + AB idiosyncratic reversal).

Z: fixed linear weights (fitted offline in S1/C4 ONLY on honest dev validation rows 001-029, grouped PCA-5) on 81
   same-hour / trailing-window cross-sectional bases of the 6 features (ranks of levels, 1/6/24/120h changes, |z| of
   levels and changes, z x change, 24/168h volatility, 168h time-series z, z-spreads), smoothed (hl 1344h), then
   residualised hour by hour on the three occupied directions (level reversal, extremeness, F5 + 5-day reversal).
X (INCLUDE_AB = True): unit( unit(Z) + unit(AB) ), AB = two idiosyncratic-reversal legs (L48 K8 hl168, L120 K3 hl48).
Evidence (research/c5_r05_orthogonal_dilution.py): Z recent proxy +0.0333 (10/10 periods), vault (out-of-sample for the
weights) +0.012, IC t 5.8, cos 0 with occupied legs, passes admissible_v2. X proxy +0.0457, rho(X,AB) 0.733.
All trailing windows are EXACT (each row from its own window) so results never depend on the frame start.
"""
import warnings

import numpy as np
import pandas as pd

from predictor import Predictor

warnings.filterwarnings("ignore")


class C5OrthogonalLearned(Predictor):
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
    HISTORY = 6200
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

    W_BASE = {"lvl1": 0.311512927201, "d1_1": 0.008000911615, "d6_1": 0.026410309493, "d24_1": 0.019785338714, "d120_1": -0.050412437464, "az0_1": -0.087525894116, "az24_1": -0.161347230805, "az120_1": -0.017825780253, "zxd24_1": -0.105189782206, "vol24_1": -0.233140204508, "vol168_1": -0.25311976459, "tsz_1": -0.056736483176, "lvl2": 0.311512927201, "d1_2": 0.008000911615, "d6_2": 0.026410309493, "d24_2": 0.019785338714, "d120_2": -0.050412437464, "az0_2": -0.087525894116, "az24_2": -0.161347230805, "az120_2": -0.017825780253, "zxd24_2": -0.105189782206, "vol24_2": -0.233140204508, "vol168_2": -0.25311976459, "tsz_2": -0.056736483176, "lvl3": 0.311512927201, "d1_3": 0.008000911615, "d6_3": 0.026410309493, "d24_3": 0.019785338714, "d120_3": -0.050412437464, "az0_3": -0.087525894116, "az24_3": -0.161347230805, "az120_3": -0.017825780253, "zxd24_3": -0.105189782206, "vol24_3": -0.233140204508, "vol168_3": -0.25311976459, "tsz_3": -0.056736483176, "lvl4": 0.311512927201, "d1_4": 0.008000911615, "d6_4": 0.026410309493, "d24_4": 0.019785338714, "d120_4": -0.050412437464, "az0_4": -0.087525894116, "az24_4": -0.161347230805, "az120_4": -0.017825780253, "zxd24_4": -0.105189782206, "vol24_4": -0.233140204508, "vol168_4": -0.25311976459, "tsz_4": -0.056736483176, "lvl5": 0.337942731778, "d1_5": 0.009767201881, "d6_5": 0.054694139955, "d24_5": -0.007179016353, "d120_5": 0.057929354201, "az0_5": 1.175137761778, "az24_5": 1.174773884607, "az120_5": 1.398687976313, "zxd24_5": 0.51755460127, "vol24_5": 3.454290894462, "vol168_5": 5.982776095369, "tsz_5": 0.129161439794, "lvl6": 2.003519424418, "d1_6": 0.039239631026, "d6_6": 0.089601636696, "d24_6": -0.082051201164, "d120_6": -0.123282374276, "az0_6": 7.925204559775, "az24_6": 1.127058422857, "az120_6": 2.091271912662, "zxd24_6": -1.198270155707, "vol24_6": -4.695968839273, "vol168_6": -10.128449033492, "tsz_6": -0.800612573565, "sp1_6": 0.039035755329, "sp1_5": 0.147207114349, "sp2_6": 0.039035755329, "sp2_5": 0.147207114349, "sp3_6": 0.039035755329, "sp3_5": 0.147207114349, "sp4_6": 0.039035755329, "sp4_5": 0.147207114349, "sp5_6": -0.424956418802}
    Z_HALFLIFE = 1344.0
    LEG_DEF_HL = (("S06", 336.0), ("S07", 672.0), ("S08", 672.0))
    INCLUDE_AB = False

    @staticmethod
    def _zx(df):
        d = df.sub(df.mean(axis=1), axis=0)
        return d.div(d.std(axis=1).replace(0.0, np.nan), axis=0)

    @staticmethod
    def _roll(df, w, minp, stat):
        """Trailing-window mean / std (ddof 1) in O(T): block-anchored prefix sums on data re-centred by each block's own
        mean (no cancellation), identical to the exact per-window computation to ~1e-14."""
        A = df.to_numpy(float)
        T, J = A.shape
        out = np.full((T, J), np.nan)
        if T == 0:
            return pd.DataFrame(out, index=df.index, columns=df.columns)
        padded = np.vstack([np.full((w - 1, J), np.nan), A])
        block = 2048
        for lo in range(0, T, block):
            hi = min(T, lo + block)
            seg = padded[lo:hi + w - 1]
            ok = np.isfinite(seg)
            with np.errstate(all="ignore"):
                c = np.nanmean(seg, axis=0)
            c = np.where(np.isfinite(c), c, 0.0)
            y = np.where(ok, seg - c, 0.0)
            z = np.zeros((1, J))
            s1 = np.vstack([z, np.cumsum(y, axis=0)])
            s2 = np.vstack([z, np.cumsum(y * y, axis=0)])
            sn = np.vstack([z, np.cumsum(ok, axis=0)])
            a1 = s1[w:] - s1[: len(s1) - w]
            a2 = s2[w:] - s2[: len(s2) - w]
            n = sn[w:] - sn[: len(sn) - w]
            with np.errstate(all="ignore"):
                if stat == "mean":
                    v = c + a1 / n
                else:
                    v = np.sqrt(np.maximum((a2 - a1 * a1 / n) / (n - 1), 0.0))
            v[n < max(minp, 2 if stat == "std" else 1)] = np.nan
            out[lo:hi] = v
        # constant windows EXACTLY (integer run lengths): std 0 and mean = the value, as the per-window computation gives
        # window constant <=> its finite values are all equal: count changes of the causally forward-filled series
        Bf = pd.DataFrame(A).ffill().to_numpy(float)
        chg = np.zeros((T, J), dtype=np.int64)
        chg[1:] = (np.isfinite(Bf[1:]) & np.isfinite(Bf[:-1]) & (Bf[1:] != Bf[:-1])).astype(np.int64)
        cc = np.vstack([np.zeros((1, J), dtype=np.int64), np.cumsum(chg, axis=0)])
        rows = np.arange(T)
        lo_idx = np.maximum(rows - w + 2, 0)
        changes_in_window = cc[rows + 1] - cc[lo_idx]
        const = (changes_in_window == 0) & np.isfinite(out)
        out[const] = 0.0 if stat == "std" else A[const]
        return pd.DataFrame(out, index=df.index, columns=df.columns)

    def _bases(self, features, tickers):
        g = {i: features[f"Feature.{i}"].reindex(columns=tickers).astype(float).replace([np.inf, -np.inf], np.nan) for i in range(1, 7)}
        z = {i: self._zx(g[i]) for i in g}
        for i in range(1, 7):
            yield f"lvl{i}", g[i]
            for k in (1, 6, 24, 120):
                yield f"d{k}_{i}", g[i].diff(k)
            yield f"az0_{i}", z[i].abs()
            yield f"az24_{i}", self._zx(g[i].diff(24)).abs()
            yield f"az120_{i}", self._zx(g[i].diff(120)).abs()
            yield f"zxd24_{i}", z[i] * self._zx(g[i].diff(24))
            d1 = g[i].diff(1)
            yield f"vol24_{i}", self._roll(d1, 24, 12, "std")
            yield f"vol168_{i}", self._roll(d1, 168, 84, "std")
            yield f"tsz_{i}", (g[i] - self._roll(g[i], 168, 84, "mean")) / self._roll(g[i], 168, 84, "std")
        for i in (1, 2, 3, 4):
            yield f"sp{i}_6", z[i] - z[6]
            yield f"sp{i}_5", z[i] - z[5]
        yield "sp5_6", z[5] - z[6]

    def _legs(self, features, tickers):
        g = {i: features[f"Feature.{i}"].reindex(columns=tickers).astype(float) for i in range(1, 7)}
        r = {i: self._rank(g[i]).fillna(0.0) for i in g}
        rk = lambda d: self._rank(d).fillna(0.0)  # noqa: E731
        s06 = self._unit(-(r[1] + r[2] + r[3] + r[4]) / 4 + 0.5 * r[6] - 0.5 * r[5] - 0.5 * rk(g[1].diff(120)))
        s07 = self._unit(-sum(rk(self._zx(g[i]).abs()) + rk(self._zx(g[i].diff(24)).abs()) + rk(self._zx(g[i].diff(120)).abs()) for i in (1, 2, 3, 4)))
        s08 = self._unit(self._unit(-r[5]) + self._unit(-sum(rk(g[i].diff(120)) for i in (1, 2, 3, 4))))
        raw = {"S06": s06, "S07": s07, "S08": s08}
        return [self._smooth(raw[k], hl, int(4 * hl)) for k, hl in self.LEG_DEF_HL]

    @staticmethod
    def _residualize(C, bases):
        A = C.fillna(0.0).to_numpy(float)
        A = A - A.mean(axis=1, keepdims=True)
        B = np.stack([b.reindex_like(C).fillna(0.0).to_numpy(float) for b in bases], axis=-1)
        B = B - B.mean(axis=1, keepdims=True)
        coef = np.einsum("tkj,tj->tk", np.linalg.pinv(B), A)
        return pd.DataFrame(A - np.einsum("tjk,tk->tj", B, coef), index=C.index, columns=C.columns)

    def _z_component(self, features, tickers):
        acc = None
        for k, v in self._bases(features, tickers):
            w = self.W_BASE.get(k)
            if w:
                u = self._unit(self._rank(v).fillna(0.0)) * w
                acc = u if acc is None else acc + u
        sm = self._smooth(acc, self.Z_HALFLIFE, int(4 * self.Z_HALFLIFE))
        return self._unit(self._residualize(sm, self._legs(features, tickers)))

    LEGS = ((48, 8, 168.0), (120, 3, 48.0))

    def _score(self, features):
        tickers = self._tickers(features)
        z = self._z_component(features, tickers)
        if not self.INCLUDE_AB:
            return z, tickers
        total = None
        for L, K, hl in self.LEGS:
            leg = self._unit(self._smooth(self._unit(self._idio_reversal(features, tickers, L, K)), hl, int(4 * hl)))
            total = leg if total is None else total + leg
        return self._unit(self._unit(total) + z), tickers

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
    predictor = C5OrthogonalLearned()
