"""Adjusted closes for the six-asset panel, using main.py's Yahoo call.

`StockFetcher.fetch_ohlcv` calls `Ticker.history(period='max')` and does not
pass `auto_adjust`. In the installed yfinance that default is True, so the
close column is the adjusted close. The first row is then dropped with the
missing log return. This module repeats that sequence and does not import it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATA_DIR
from .data_prep import (
    _append_manifest,
    _utc_now,
    daily_index,
    read_manifest,
    sha256_file,
)

ASSETS = ["SPY", "QQQ", "IWM", "TLT", "^VIX", "DX-Y.NYB"]
OHLC_COLUMNS = ["open", "high", "low", "close", "volume"]


def _flatten_history(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(column[0]) for column in out.columns]
    out.columns = [str(column).strip().lower().replace(" ", "_") for column in out.columns]
    out.index = daily_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _drop_first_log_return(frame: pd.DataFrame) -> pd.DataFrame:
    """Match fetch_ohlcv, which drops the row whose log return is missing."""
    close = frame["close"].astype(float)
    log_return = np.log(close).diff()
    kept = frame.loc[log_return.notna() & close.notna()].copy()
    kept.index.name = "date"
    return kept


def fetch_ohlcv_like_main(symbol: str, auto_adjust: bool) -> pd.DataFrame:
    import yfinance as yf

    if auto_adjust:
        # Same signature as StockFetcher.fetch_ohlcv: the default is left in place.
        downloaded = yf.Ticker(symbol).history(period="max", interval="1d")
    else:
        downloaded = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)
    if downloaded is None or len(downloaded) == 0:
        raise RuntimeError(f"no Yahoo history for {symbol}")
    frame = _flatten_history(downloaded)
    missing = [column for column in OHLC_COLUMNS if column not in frame.columns]
    if missing and not (auto_adjust is False and "adj_close" in frame.columns):
        raise RuntimeError(f"{symbol} history is missing {missing}")
    if "close" not in frame.columns and "adj_close" in frame.columns:
        frame["close"] = frame["adj_close"]
    return _drop_first_log_return(frame)


def _panel_path(name: str) -> Path:
    return DATA_DIR / name


def _load_panel(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    frame.index = daily_index(frame.index)
    panels = {}
    for asset in frame.columns.get_level_values(0).unique():
        panel = frame[asset].copy()
        panel.index = frame.index
        panels[str(asset)] = panel
    return panels


def _write_panel(panels: dict[str, pd.DataFrame], path: Path, source: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(panels, axis=1)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["asset", "field"])
    frame.index.name = "date"
    temporary = path.with_suffix(".csv.partial")
    frame.to_csv(temporary)
    temporary.replace(path)
    record = {
        "kind": "asset_ohlc",
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
        "created_at_utc": _utc_now(),
        "source": source,
    }
    _append_manifest(record, DATA_DIR)
    return record


def ensure_asset_panels() -> dict:
    """Load or download adjusted and raw panels. Existing files are not replaced."""
    adjusted_path = _panel_path("asset_ohlc_main_convention_v1.csv")
    raw_path = _panel_path("asset_ohlc_raw_v1.csv")
    created = False
    if not adjusted_path.exists() or not raw_path.exists():
        adjusted = {symbol: fetch_ohlcv_like_main(symbol, auto_adjust=True) for symbol in ASSETS}
        raw = {symbol: fetch_ohlcv_like_main(symbol, auto_adjust=False) for symbol in ASSETS}
        if not adjusted_path.exists():
            _write_panel(adjusted, adjusted_path, "yahoo:history(period=max) default auto_adjust")
            created = True
        if not raw_path.exists():
            _write_panel(raw, raw_path, "yahoo:history(period=max, auto_adjust=False)")
            created = True
    return {
        "adjusted": _load_panel(adjusted_path),
        "raw": _load_panel(raw_path),
        "adjusted_path": adjusted_path,
        "raw_path": raw_path,
        "adjusted_sha256": sha256_file(adjusted_path),
        "raw_sha256": sha256_file(raw_path),
        "created": created,
    }


def close_frame(panels: dict[str, pd.DataFrame], column: str = "close") -> pd.DataFrame:
    series = []
    for asset, frame in panels.items():
        values = frame[column].copy()
        values.name = asset
        series.append(values)
    return pd.concat(series, axis=1).sort_index()
