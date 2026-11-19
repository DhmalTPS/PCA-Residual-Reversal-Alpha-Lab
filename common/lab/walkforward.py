"""Purged, embargoed, expanding-window walk-forward.

For each test block (a calendar month by default):
    train on rows with time < block_start - embargo     (purges overlapping-label leakage)
    predict on [block_start - history, block_end]       (history only for warm-up)
    keep the block's rows.
The vault months are never touched here unless `vault=True`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lab.io import Panel, dev_vault_split, month_labels


def _offset(index: pd.Index, hours: int):
    return pd.Timedelta(hours=hours) if isinstance(index, pd.DatetimeIndex) else hours


def _rows_before(index: pd.Index, t0, hours: int) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return np.asarray(index < t0 - pd.Timedelta(hours=hours))
    pos = index.get_loc(t0)
    return np.arange(len(index)) < pos - hours


def run(cls, panel: Panel, s: dict, mode: str = "full", retrain_every: int = 1, log=print) -> tuple[pd.DataFrame, list]:
    """Returns (OOS predictions P, per-fold info)."""
    dev_mask, vault_mask, dev_months, vault_months = dev_vault_split(panel, int(s["vault_months"]))
    months = panel.months()
    if mode == "vault":
        test_months = vault_months
    else:
        p_blocks = [m for m in dev_months if m.startswith("P")]
        # integer-index (shipped) data: test on the blocks each period file adds (the official folds);
        # datetime data: calendar months after a minimum training span
        test_months = p_blocks if p_blocks else dev_months[int(s["min_train_months"]):]
        if mode == "quick":
            test_months = test_months[len(test_months) - int(s["quick_test_months"]):]
    if not test_months:
        raise ValueError("no test months: not enough history for min_train_months + vault_months")

    emb = int(s["embargo_hours"])
    hist = int(s["predict_history_hours"])
    idx = panel.index
    preds, info = [], []
    model = None
    for k, m in enumerate(test_months):
        in_block = (months == m).to_numpy()
        rows = np.flatnonzero(in_block)
        t0, t1 = idx[rows[0]], idx[rows[-1]]
        if model is None or k % max(1, retrain_every) == 0:
            tr = _rows_before(idx, t0, emb)
            if mode != "vault":
                tr &= dev_mask
            t_start = time.perf_counter()
            model = cls()
            if hasattr(model, "generate_features"):
                # official parity: the hook receives the period's FULL feature frame (up to the block end)
                model.generate_features(panel.features.iloc[: rows[-1] + 1])
            model.train(panel.features.loc[tr], panel.target.loc[tr])
            t_train = time.perf_counter() - t_start
            n_train = int(tr.sum())
        if hasattr(model, "generate_features"):
            if k % max(1, retrain_every) != 0:
                model.generate_features(panel.features.iloc[: rows[-1] + 1])
            lo = rows[0]                      # official: predict on the block alone
        else:
            lo = max(0, rows[0] - hist)       # no hook: give explicit warm-up history
        frame = panel.features.iloc[lo: rows[-1] + 1]
        t_start = time.perf_counter()
        out = model.predict(frame)
        t_pred = time.perf_counter() - t_start
        out = pd.DataFrame(out, index=frame.index).reindex(columns=panel.tickers)
        preds.append(out.loc[t0:t1])
        info.append({"month": m, "train_rows": n_train, "train_s": round(t_train, 2), "predict_s": round(t_pred, 2)})
        log(f"  fold {m}: train_rows={n_train} train={t_train:.1f}s predict={t_pred:.1f}s")
    P = pd.concat(preds)
    P = P[~P.index.duplicated(keep="last")]
    return P, info
