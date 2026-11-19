"""Loading the shipped panel.

Period files are cumulative, so only the latest period is loaded by default
(concatenating periods would duplicate history). Tickers are rotated per period, so
nothing here ever matches tickers across periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lab.paths import data_root


@dataclass
class Panel:
    features: pd.DataFrame            # index: time, columns: MultiIndex (feature, ticker)
    returns: pd.DataFrame             # time x ticker (the series the backtest multiplies)
    target: pd.DataFrame              # time x ticker (xs z-scored forward target, clipped +-5)
    dispersion: pd.Series | None      # time -> xs std of the target-window return (FORWARD-LOOKING)
    period: str
    root: Path
    gauge: dict = field(default_factory=dict)
    period_rows: list = field(default_factory=list)   # [(period, n_rows)] of every period file on disk

    @property
    def index(self) -> pd.Index:
        return self.features.index

    @property
    def tickers(self) -> list:
        return list(dict.fromkeys(self.features.columns.get_level_values(-1)))

    @property
    def feature_names(self) -> list:
        return list(dict.fromkeys(self.features.columns.get_level_values(0)))

    def months(self) -> pd.Series:
        """Fold label per row (see month_labels)."""
        return month_labels(self.index, self.period_rows)

    def rows(self, mask) -> "Panel":
        mask = np.asarray(mask, dtype=bool)
        return Panel(
            features=self.features.loc[mask],
            returns=self.returns.loc[mask],
            target=self.target.loc[mask],
            dispersion=None if self.dispersion is None else self.dispersion.loc[mask],
            period=self.period,
            root=self.root,
            gauge=self.gauge,
            period_rows=self.period_rows,
        )


def list_periods(root: Path | None = None) -> list[str]:
    root = Path(root) if root else data_root()
    return sorted({p.name.split(".")[0] for p in root.glob("*.small.features.parquet")})


def _as_time_by_ticker(obj, tickers: list, index: pd.Index) -> pd.DataFrame:
    df = obj
    if isinstance(df, pd.Series):
        df = df.unstack() if isinstance(df.index, pd.MultiIndex) else df.to_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    out = df.reindex(columns=tickers)
    if out.isna().all().all() and df.shape[1] == len(tickers):
        out = df.copy()
        out.columns = tickers
    return out.reindex(index).astype(float)


def _read(path: Path):
    return pd.read_parquet(path) if path.exists() else None


def load_period(root: Path | str | None = None, period: str | None = None) -> Panel:
    root = Path(root) if root else data_root()
    periods = list_periods(root)
    if not periods:
        raise FileNotFoundError(f"no *.small.features.parquet under {root}")
    period = period or periods[-1]
    stem = root / f"{period}.small"

    feats = pd.read_parquet(f"{stem}.features.parquet")
    if not isinstance(feats.columns, pd.MultiIndex):
        raise ValueError("features must have MultiIndex columns (feature, ticker)")
    feats = feats[~feats.index.duplicated(keep="last")].sort_index()
    tickers = list(dict.fromkeys(feats.columns.get_level_values(-1)))
    idx = feats.index

    rets = _as_time_by_ticker(pd.read_parquet(f"{stem}.returns.parquet"), tickers, idx)
    tgt = _as_time_by_ticker(pd.read_parquet(f"{stem}.target.parquet"), tickers, idx)

    disp = _read(Path(f"{stem}.dispersion.parquet"))
    if disp is not None:
        if isinstance(disp, pd.DataFrame):
            disp = disp.iloc[:, 0]
        disp = disp[~disp.index.duplicated(keep="last")].reindex(idx).astype(float)

    gauge = {}
    gdir = root / "gauge"
    if gdir.exists():
        for g in sorted(gdir.glob(f"{period}.*.npy")):
            gauge[g.name] = g

    import pyarrow.parquet as pq

    period_rows = [(p, pq.ParquetFile(root / f"{p}.small.features.parquet").metadata.num_rows) for p in periods]
    return Panel(feats, rets, tgt, disp, period, root, gauge, period_rows)


def month_labels(index: pd.Index, period_rows: list | None = None, block: int = 730) -> pd.Series:
    """Fold label per row.

    * DatetimeIndex: calendar month.
    * Integer (row-number) index, as shipped: the rows each cumulative period file ADDS form one
      block labelled 'P<period>' (about one month of hours); rows already in the first period file
      are split into 730-row blocks 'H<k>' counted back from its end. Requires period_rows.
    """
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(index.to_period("M").astype(str), index=index)
    pos = np.arange(len(index))
    labels = np.empty(len(index), dtype=object)
    if period_rows:
        first_n = period_rows[0][1]
        k = (first_n - 1 - pos) // block
        labels[:] = [f"H{int(v):03d}" for v in k]
        prev = first_n
        for p, n in period_rows[1:]:
            labels[(pos >= prev) & (pos < n)] = f"P{p}"
            prev = n
        labels[pos >= prev] = "P_after"
    else:
        labels[:] = [f"B{int(v):04d}" for v in pos // block]
    return pd.Series(labels, index=index)


def dev_vault_split(panel: Panel, vault_months: int) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Boolean masks (dev, vault) and the month lists. The vault is the LAST `vault_months` months."""
    m = panel.months()
    months = list(dict.fromkeys(m.tolist()))
    vault = months[len(months) - vault_months:] if vault_months > 0 else []
    dev = [x for x in months if x not in vault]
    vmask = m.isin(vault).to_numpy()
    return ~vmask, vmask, dev, vault
