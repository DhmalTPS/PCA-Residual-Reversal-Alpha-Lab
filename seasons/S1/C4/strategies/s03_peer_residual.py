"""S03 - peer-relative structure: gaps to a trailing correlation-network neighbourhood.

Different information from S01/S02: instead of each asset's own feature state, use how far an
asset sits from its PEERS, where peers are defined by the trailing co-movement of scaled feature
changes (finite window). Catch-up (peers moved, asset has not) and divergence from the peer
neighbourhood are classic cross-sectional relationships that per-asset models do not see.
Each gap is also interacted with a dispersion regime z-score.

Pipeline (every step uses only hours <= t, over finite trailing windows):
  1. trailing (PEER_WINDOW) correlation of a combined change vector -> non-negative,
     row-normalised peer weights W(t) with zero diagonal;
  2. per feature: level gap x - W x, change gap, catch-up W dx - dx, and gap x regime;
  3. ridge with purged blocked CV (one-SE rule), then the shared position pipeline.
"""
import warnings

import numpy as np
import pandas as pd

from predictor import Predictor

warnings.filterwarnings("ignore")


class S03PeerResidual(Predictor):
    LAG = 1
    WINDOWS = (6, 24, 96)
    SCALE_WINDOW = 168
    VOL_WINDOW = 24
    CLIP = 4.0
    TARGET_CLIP = 5.0
    PEER_WINDOW = 168
    REGIME_WINDOW = 336
    MAX_TRAIN_HOURS = 24 * 540
    HOUR_FRACTION = 0.6
    RIDGE_GRID = (0.003, 0.03, 0.3, 3.0, 30.0)
    CV_FOLDS = 4
    CV_PURGE = 48
    ONE_SE = True
    SMOOTH_HALFLIFE = 6.0
    SMOOTH_WINDOW = 24
    BAND = 0.0
    BAND_WINDOW = 24
    ORTHO_LEVELS = False
    SEED = 20261006

    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.scale_ = None
        self.names_ = None
        self.lam_ = None
        self.cv_table_ = None

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

    def _peer_weights(self, D):
        """D: (T, J) combined change. Trailing-window correlation via windowed sums
        (each hour uses exactly its own last PEER_WINDOW hours)."""
        T, J = D.shape
        W = int(self.PEER_WINDOW)
        cs1 = np.cumsum(D, axis=0)
        cs2 = np.cumsum(D[:, :, None] * D[:, None, :], axis=0)
        old1 = np.zeros_like(cs1)
        old2 = np.zeros_like(cs2)
        if T > W:
            old1[W:] = cs1[: T - W]
            old2[W:] = cs2[: T - W]
        n = np.minimum(np.arange(1, T + 1), W).astype(float)
        m1 = (cs1 - old1) / n[:, None]
        m2 = (cs2 - old2) / n[:, None, None]
        cov = m2 - m1[:, :, None] * m1[:, None, :]
        sd = np.sqrt(np.clip(np.einsum("tjj->tj", cov), 0.0, None))
        den = sd[:, :, None] * sd[:, None, :]
        corr = np.divide(cov, den, out=np.zeros_like(cov), where=den > 1e-12)
        corr[n < 24] = 0.0
        Wt = np.clip(corr, 0.0, None)
        idx = np.arange(J)
        Wt[:, idx, idx] = 0.0
        rs = Wt.sum(axis=2, keepdims=True)
        uniform = (1.0 - np.eye(J)) / (J - 1)
        return np.where(rs > 1e-12, Wt / np.where(rs > 1e-12, rs, 1.0), uniform[None])

    def _peer_features(self, features):
        base, tickers = self._features(features)
        names = self._feature_names(features)
        idx = features.index
        chg = []
        for name in names:
            raw = features[name].reindex(columns=tickers).astype(float).replace([np.inf, -np.inf], np.nan)
            scale = raw.rolling(self.SCALE_WINDOW, min_periods=24).std()
            chg.append(self._xs_z(raw.diff() / scale.where(scale > 0), self.CLIP).to_numpy(float))
        D = np.nan_to_num(np.nanmean(np.stack(chg, axis=0), axis=0))
        Wt = self._peer_weights(D)
        disp = pd.Series(D.std(axis=1), index=idx)
        m = disp.rolling(self.REGIME_WINDOW, min_periods=48).mean()
        s = disp.rolling(self.REGIME_WINDOW, min_periods=48).std()
        rz = ((disp - m) / s.where(s > 0)).clip(-3, 3).fillna(0.0).to_numpy() / 3.0

        out = {}
        for name in names:
            lvl = np.nan_to_num(base[f"{name}|lvl"].to_numpy(float))
            d24 = np.nan_to_num(base[f"{name}|d24"].to_numpy(float))
            d6 = np.nan_to_num(base[f"{name}|d6"].to_numpy(float))
            gap_l = lvl - np.einsum("tjk,tk->tj", Wt, lvl)
            gap_d = d24 - np.einsum("tjk,tk->tj", Wt, d24)
            catch = np.einsum("tjk,tk->tj", Wt, d6) - d6
            for key, arr in ((f"{name}|gap_lvl", gap_l), (f"{name}|gap_d24", gap_d), (f"{name}|catch_d6", catch),
                             (f"{name}|gap_lvl_x_reg", gap_l * rz[:, None]), (f"{name}|gap_d24_x_reg", gap_d * rz[:, None])):
                out[key] = self._xs_z(pd.DataFrame(arr, index=idx, columns=tickers), self.CLIP)
        return out, tickers

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
        feats, tickers = self._peer_features(features)
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
        if len(hours) < 100:
            self.coef_ = np.zeros(K)
            self.scale_ = np.ones(K)
            self.lam_ = float("nan")
            return self
        Xh = X[hours]
        sd = Xh.reshape(-1, K).std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        Xh = Xh / self.scale_
        Yh = Yc[hours] - Yc[hours].mean(axis=1, keepdims=True)
        self.lam_, self.coef_ = self._fit_ridge(Xh, Yh, hours)
        return self

    def predict(self, features):
        feats, tickers = self._peer_features(features)
        X = self._cube(feats, features.index, tickers)
        if self.coef_ is None:
            score = np.zeros(X.shape[:2])
        else:
            score = (X / self.scale_) @ self.coef_
        S = pd.DataFrame(score, index=features.index, columns=tickers)
        return self._finalize(S, feats)


if __name__ == "__main__":
    predictor = S03PeerResidual()
