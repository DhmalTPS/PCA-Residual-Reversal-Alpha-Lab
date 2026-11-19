"""RUNTIME compliance gate: executes the strategy the way the server's checks do.

  demean        rows sum to 0                                   (NOT_DEMEANED)
  shape/finite  DataFrame, same index, every ticker, no NaN/inf (CANT_RUN)
  causality     predictions for hours <= t are IDENTICAL whether or not later rows are present
                (catches any look-ahead the static scan cannot see)            (PEEK_THE_FUTURE)
  row_shift     prepending history rows does not change predictions for hours well after the
                start (catches row counters, block schedules, first-row anchoring) (ROW_MASK_DETECTED)
  determinism   two identical calls agree exactly                (ROW_MASK "failed to run")
  reload        pickle -> unpickle -> identical predictions       (daily scorer reload)
  nan_robust    1% NaN injected into features -> still finite output
  shuffle_robust  train + predict on row-permuted data does not crash      (gate refusal)
  short_batch   train on a ~400-row slice, and on an all-zero target, does not crash
  flat          flat-row fraction / longest flat run              (FLAT_PERIODS)
  timing        train <= 240 s (budget 30 s), predict <= 60 s, projected gate cost

Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab.backtest import flat_stats, row_norms  # noqa: E402
from lab.io import dev_vault_split, load_period  # noqa: E402
from lab.loader import load_strategy  # noqa: E402
from lab.paths import REPORTS, archive_existing, data_root, settings  # noqa: E402


def _rel_diff(A: pd.DataFrame, B: pd.DataFrame) -> tuple[float, float]:
    """(max |A-B| / median row L2 norm of A, fraction of entries with matching sign)."""
    idx = A.index.intersection(B.index)
    a = A.loc[idx].to_numpy(float)
    b = B.loc[idx, A.columns].to_numpy(float)
    med = float(np.median(np.linalg.norm(a, axis=1))) or 1.0
    big = np.abs(a) > 1e-3 * med / np.sqrt(a.shape[1])
    sign = float((np.sign(a[big]) == np.sign(b[big])).mean()) if big.any() else 1.0
    return float(np.nanmax(np.abs(a - b)) / med) if len(idx) else float("nan"), sign


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail, level="FAIL"):
        self.rows.append({"check": name, "status": "PASS" if ok else level, "detail": detail})

    def failed(self):
        return any(r["status"] == "FAIL" for r in self.rows)

    def show(self):
        w = max(len(r["check"]) for r in self.rows)
        for r in self.rows:
            print(f"  [{r['status']:>4}] {r['check']:<{w}}  {r['detail']}")


def run(path: str, data: str | None = None) -> dict:
    s = settings()
    cls = load_strategy(path)
    panel = load_period(data_root(data))
    dev, _, _, _ = dev_vault_split(panel, int(s["vault_months"]))
    F = panel.features.loc[dev]
    T = panel.target.loc[dev]
    tickers = panel.tickers
    n = len(F)
    cut = int(n * 0.7)
    rep = Report()
    lim = s["limits"]

    def guarded(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            rep.add(name, False, f"exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    # --- train + predict timing
    t0 = time.perf_counter()
    model = cls()
    t_init = time.perf_counter() - t0
    has_hook = hasattr(model, "generate_features")
    if has_hook:
        model.generate_features(F)
    t0 = time.perf_counter()
    if guarded("train", lambda: model.train(F.iloc[:cut], T.iloc[:cut])) is None and rep.failed():
        rep.show()
        return {"passed": False, "checks": rep.rows}
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    P = guarded("predict", lambda: model.predict(F))
    t_pred = time.perf_counter() - t0
    if P is None:
        rep.show()
        return {"passed": False, "checks": rep.rows}

    # --- shape / finiteness
    okshape = isinstance(P, pd.DataFrame) and P.index.equals(F.index) and set(P.columns) == set(tickers)
    rep.add("shape", okshape, f"type={type(P).__name__} shape={getattr(P, 'shape', None)}")
    P = pd.DataFrame(P, index=F.index).reindex(columns=tickers)
    finite = bool(np.isfinite(P.to_numpy(float)).all())
    rep.add("finite", finite, "no NaN/inf" if finite else f"{int((~np.isfinite(P.to_numpy(float))).sum())} non-finite")

    # --- demean
    rs = P.sum(axis=1).abs()
    scale = row_norms(P, 1).median() or 1.0
    rep.add("demean", float(rs.max() / scale) < 1e-9, f"max |row sum| / median L1 = {rs.max() / scale:.2e}")

    # --- determinism / reload
    P2 = guarded("determinism", lambda: model.predict(F))
    if P2 is not None:
        d, _ = _rel_diff(P, pd.DataFrame(P2, index=F.index).reindex(columns=tickers))
        rep.add("determinism", d == 0.0 or d < 1e-12, f"max rel diff = {d:.2e}")
    blob = guarded("reload", lambda: pickle.dumps(model))
    if blob is not None:
        m2 = pickle.loads(blob)
        P3 = guarded("reload", lambda: m2.predict(F))
        if P3 is not None:
            d, _ = _rel_diff(P, pd.DataFrame(P3, index=F.index).reindex(columns=tickers))
            rep.add("reload", d < 1e-12, f"pickle {len(blob) / 1e6:.2f} MB, max rel diff = {d:.2e}")

    # --- causality (exact)
    tc = cut + (n - cut) // 2
    Pa = guarded("causality", lambda: model.predict(F.iloc[:tc]))
    if Pa is not None:
        Pa = pd.DataFrame(Pa, index=F.index[:tc]).reindex(columns=tickers)
        d1, _ = _rel_diff(Pa, P.iloc[:tc])
        rep.add("causality", d1 < 1e-9, f"rows<=t with vs without {n - tc} later rows: max rel diff = {d1:.2e}")

    # --- row shift (prepend history)
    pad = int(s["row_shift"]["pad_hours"])
    burn = int(s["row_shift"]["burn_in_hours"])
    Ps = guarded("row_shift", lambda: model.predict(F.iloc[pad:]))
    if Ps is not None:
        Ps = pd.DataFrame(Ps, index=F.index[pad:]).reindex(columns=tickers)
        cmp_idx = F.index[pad + burn:]
        d2, sign = _rel_diff(P.loc[cmp_idx], Ps.loc[cmp_idx])
        tol = float(s["row_shift"]["rel_tol"])
        rep.add("row_shift", d2 < tol, f"start moved by {pad} rows, compared after {burn}-row burn-in: max rel diff = {d2:.2e} (tol {tol:g}), sign agreement {sign:.4f}")

    # --- history hook: a lone 800-row block must equal the same rows computed with full history
    if has_hook:
        vb = F.iloc[n - 800:]
        Pv = guarded("history_hook", lambda: model.predict(vb))
        if Pv is not None:
            Pv = pd.DataFrame(Pv, index=vb.index).reindex(columns=tickers)
            d3, sign3 = _rel_diff(P.loc[vb.index], Pv)
            rep.add("history_hook", d3 < float(s["row_shift"]["rel_tol"]),
                    f"lone 800-row block vs full-history rows: max rel diff = {d3:.2e}, sign agreement {sign3:.4f}")
        fresh = cls()
        fresh.train(F.iloc[:cut], T.iloc[:cut])
        Pn = guarded("no_hook_fallback", lambda: fresh.predict(vb))
        if Pn is not None:
            ok = bool(np.isfinite(pd.DataFrame(Pn).to_numpy(float)).all())
            rep.add("no_hook_fallback", ok, "predict without generate_features still runs (cold start)")

    # --- robustness
    rng = np.random.default_rng(0)

    def nan_test():
        Fn = F.copy()
        mask = rng.random(Fn.shape) < 0.01
        Fn = Fn.mask(mask)
        out = pd.DataFrame(model.predict(Fn), index=F.index)
        return bool(np.isfinite(out.to_numpy(float)).all())
    r = guarded("nan_robust", nan_test)
    if r is not None:
        rep.add("nan_robust", r, "1% NaN injected")

    def shuffle_test():
        perm = rng.permutation(cut)
        Fp = pd.DataFrame(F.iloc[:cut].to_numpy()[perm], index=F.index[:cut], columns=F.columns)
        Tp = pd.DataFrame(T.iloc[:cut].to_numpy()[perm], index=T.index[:cut], columns=T.columns)
        m = cls()
        m.train(Fp, Tp)
        out = pd.DataFrame(m.predict(Fp), index=Fp.index)
        return bool(np.isfinite(out.to_numpy(float)).all())
    r = guarded("shuffle_robust", shuffle_test)
    if r is not None:
        rep.add("shuffle_robust", r, "row-permuted train/predict")

    def short_test():
        m = cls()
        m.train(F.iloc[:400], T.iloc[:400])
        a = pd.DataFrame(m.predict(F.iloc[:400]), index=F.index[:400])
        m = cls()
        m.train(F.iloc[:cut], T.iloc[:cut] * 0.0)
        b = pd.DataFrame(m.predict(F.iloc[:600]), index=F.index[:600])
        return bool(np.isfinite(a.to_numpy(float)).all() and np.isfinite(b.to_numpy(float)).all())
    r = guarded("short_batch", short_test)
    if r is not None:
        rep.add("short_batch", r, "400-row batch and all-zero target")

    # --- flat periods (on the out-of-sample part)
    fs = flat_stats(P.iloc[cut:], float(s["flat"]["rel_threshold"]))
    fs_all = flat_stats(P, float(s["flat"]["rel_threshold"]))
    okf = fs["flat_fraction"] <= s["flat"]["max_fraction"] and fs["longest_flat_run"] <= s["flat"]["max_run_hours"]
    rep.add("flat", okf, f"OOS flat fraction {fs['flat_fraction']:.4f}, longest run {fs['longest_flat_run']} h "
                         f"(whole frame incl. warm-up: {fs_all['flat_fraction']:.4f}, {fs_all['longest_flat_run']} h)")

    # --- timing
    per_retrain = t_init + t_train + t_pred
    rep.add("train_time", t_train <= lim["train_seconds_hard"], f"{t_train:.1f} s (hard {lim['train_seconds_hard']} s)")
    rep.add("train_budget", t_train <= lim["train_seconds_budget"], f"{t_train:.1f} s (gate-comfort budget {lim['train_seconds_budget']} s)", level="WARN")
    rep.add("predict_time", t_pred <= lim["predict_seconds_hard"], f"{t_pred:.1f} s on {n} rows (hard {lim['predict_seconds_hard']} s)")
    rep.add("gate_projection", per_retrain * 50 < 3600, f"init+train+predict = {per_retrain:.1f} s -> x50 retrains = {per_retrain * 50 / 60:.1f} min", level="WARN")

    rep.show()
    out = {"file": path, "passed": not rep.failed(), "checks": rep.rows,
           "timing": {"init": t_init, "train": t_train, "predict": t_pred}}
    REPORTS.mkdir(exist_ok=True)
    archive_existing(REPORTS / f"dynamic_{Path(path).stem}.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    print(f"dynamic_check {a.file}")
    r = run(a.file, a.data)
    print("RESULT:", "PASS" if r["passed"] else "FAIL")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
