"""Read the research matrix in place and archive each new hash.

`data/asset_indicator_matrix.csv` is the matrix this package analyzes. This
module never writes that file. A changed hash is copied to a new
`feature_matrix_vN.csv`; older snapshots stay as they are.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATA_DIR, FEATURE_SOURCE, MANIFEST_NAME, TRAIN_END, ZERO_VARIANCE_TOLERANCE


class FeatureSchemaError(RuntimeError):
    """The live matrix does not match the columns a saved model was fitted on."""


class FrozenTrainingMismatch(RuntimeError):
    """The training window no longer matches a saved fit. The fit is left unchanged."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def daily_index(index) -> pd.DatetimeIndex:
    """Calendar dates, keeping the wall-clock date of a timezone-aware stamp."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def load_feature_matrix(path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    frame.index = daily_index(frame.index)
    frame = frame.sort_index()
    frame.index.name = "date"
    if frame.index.has_duplicates:
        raise ValueError(f"{path} has duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} dates are not sorted")
    return frame


def column_labels(frame: pd.DataFrame) -> list[str]:
    return [f"{asset}|{indicator}" for asset, indicator in frame.columns]


def _version_number(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.rsplit("_v", 1)[1])


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_path(data_dir: Path) -> Path:
    return Path(data_dir) / MANIFEST_NAME


def read_manifest(data_dir: Path = DATA_DIR) -> list[dict]:
    path = _manifest_path(data_dir)
    if not path.exists():
        return _migrate_legacy_manifest(data_dir)
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_manifest(record: dict, data_dir: Path) -> None:
    path = _manifest_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    _render_manifest(read_manifest(data_dir), data_dir)


def _render_manifest(records: list[dict], data_dir: Path) -> None:
    lines = [
        "Primary feature source, read only: data/asset_indicator_matrix.csv",
        "Snapshots below are archival copies. This package does not write the source file.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"kind: {record['kind']}",
                f"file: {record['file']}",
                f"sha256: {record['sha256']}",
                f"rows: {record['rows']}",
                f"columns: {record.get('columns', '')}",
                f"start: {record['start']}",
                f"end: {record['end']}",
                f"created_at_utc: {record['created_at_utc']}",
                "",
            ]
        )
    (Path(data_dir) / "MANIFEST.txt").write_text("\n".join(lines))


def _legacy_created_at(data_dir: Path) -> str:
    path = Path(data_dir) / "MANIFEST.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("written_at_utc:"):
                return line.split(":", 1)[1].strip()
    return _utc_now()


def _migrate_legacy_manifest(data_dir: Path) -> list[dict]:
    records = []
    created = _legacy_created_at(data_dir)
    feature = Path(data_dir) / "feature_matrix_v1.csv"
    if feature.exists():
        frame = load_feature_matrix(feature)
        records.append(
            {
                "kind": "feature_matrix",
                "file": feature.name,
                "sha256": sha256_file(feature),
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                "start": frame.index.min().date().isoformat(),
                "end": frame.index.max().date().isoformat(),
                "created_at_utc": created,
                "source": "data/asset_indicator_matrix.csv",
            }
        )
    prices = Path(data_dir) / "qqq_ohlc_v1.csv"
    if prices.exists():
        frame = load_qqq_prices(prices)
        records.append(
            {
                "kind": "qqq_ohlc",
                "file": prices.name,
                "sha256": sha256_file(prices),
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "start": frame.index.min().date().isoformat(),
                "end": frame.index.max().date().isoformat(),
                "created_at_utc": created,
                "source": "yahoo:QQQ",
            }
        )
    if records:
        path = _manifest_path(data_dir)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        _render_manifest(records, data_dir)
    return records


def archive_feature_matrix(source=FEATURE_SOURCE, frame: pd.DataFrame | None = None, data_dir: Path = DATA_DIR) -> dict:
    """Copy the source only when its hash is not already archived."""
    source = Path(source)
    data_dir = Path(data_dir)
    if not source.exists():
        raise FileNotFoundError(f"research matrix is missing: {source}")
    source_hash = sha256_file(source)
    if frame is None:
        frame = load_feature_matrix(source)
    records = [record for record in read_manifest(data_dir) if record["kind"] == "feature_matrix"]
    for record in records:
        if record["sha256"] == source_hash:
            return {
                "path": data_dir / record["file"],
                "sha256": source_hash,
                "created": False,
                "file": record["file"],
            }
    version = 1 + max((_version_number(record["file"]) for record in records), default=0)
    filename = f"feature_matrix_v{version}.csv"
    dest = data_dir / filename
    if dest.exists():
        raise FileExistsError(f"{filename} already exists; refusing to overwrite an archived snapshot")
    data_dir.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".csv.partial")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise IOError("snapshot copy hash does not match the research matrix")
    temporary.replace(dest)
    record = {
        "kind": "feature_matrix",
        "file": filename,
        "sha256": source_hash,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
        "created_at_utc": _utc_now(),
        "source": "data/asset_indicator_matrix.csv",
    }
    _append_manifest(record, data_dir)
    return {"path": dest, "sha256": source_hash, "created": True, "file": filename}


def assert_schema_matches_model(features: pd.DataFrame, required_columns: list[str], known_columns: list[str] | None = None) -> None:
    """Stop when the live matrix cannot feed a saved model without changing that model."""
    incoming = column_labels(features)
    missing = [name for name in required_columns if name not in incoming]
    if missing:
        preview = ", ".join(missing[:8])
        raise FeatureSchemaError(
            "Stopping without changing the saved model. "
            f"{len(missing)} required columns are missing from data/asset_indicator_matrix.csv. "
            f"First missing: {preview}."
        )
    known = set(known_columns or required_columns)
    unexpected = [name for name in incoming if name not in known]
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise FeatureSchemaError(
            "Stopping without changing the saved model. "
            f"data/asset_indicator_matrix.csv has {len(unexpected)} columns the fitted model does not use. "
            f"First new columns: {preview}."
        )


def assert_training_matches_scaler(train_frame: pd.DataFrame, scaler, kept_columns) -> None:
    values = train_frame.loc[:, kept_columns].to_numpy(dtype=float)
    seen = int(scaler.n_samples_seen_)
    if len(values) != seen:
        raise FrozenTrainingMismatch(
            "Stopping without refitting. "
            f"The current training window has {len(values)} rows and the saved scaler was fit on {seen}."
        )
    if not np.allclose(values.mean(axis=0), scaler.mean_, rtol=1e-6, atol=1e-8):
        raise FrozenTrainingMismatch(
            "Stopping without refitting. Training-period column means no longer match the saved scaler."
        )


def _flatten_yahoo(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]
    out.index = daily_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def download_qqq_ohlc(start: str = "1999-01-01", end: str | None = None) -> pd.DataFrame:
    """Raw Yahoo OHLC plus adjusted OHLC. Adjusted close is the later return input."""
    import yfinance as yf

    raw = yf.download(
        "QQQ",
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    adjusted = yf.download(
        "QQQ",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("QQQ download returned no rows")
    raw = _flatten_yahoo(raw)
    adjusted = _flatten_yahoo(adjusted)
    out = pd.DataFrame(index=raw.index)
    out.index.name = "date"
    for column in (
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
        "capital_gains",
    ):
        if column in raw.columns:
            out[column] = raw[column]
    for source_name, dest_name in (
        ("open", "open_adjusted"),
        ("high", "high_adjusted"),
        ("low", "low_adjusted"),
        ("close", "close_adjusted"),
    ):
        if source_name in adjusted.columns:
            out[dest_name] = adjusted[source_name].reindex(out.index)
    if "close_adjusted" not in out.columns:
        if "adj_close" not in out.columns:
            raise RuntimeError("QQQ download has neither adjusted close nor Adj Close")
        out["close_adjusted"] = out["adj_close"]
    out = out.dropna(subset=["close_adjusted"]).sort_index()
    return out


def ensure_qqq_prices(feature_index=None, data_dir: Path = DATA_DIR, start: str = "1999-01-01") -> dict:
    """Use the latest archived QQQ file when it covers the feature dates.

    A short file is left in place. Coverage is added as the next qqq_ohlc version.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    needed = None if feature_index is None else set(daily_index(feature_index))
    records = [record for record in read_manifest(data_dir) if record["kind"] == "qqq_ohlc"]
    if records:
        latest = data_dir / records[-1]["file"]
        prices = load_qqq_prices(latest)
        if needed is None or needed.issubset(set(prices.index)):
            return {"path": latest, "sha256": sha256_file(latest), "downloaded": False, "file": latest.name}
    version = 1 + max((_version_number(record["file"]) for record in records), default=0)
    filename = f"qqq_ohlc_v{version}.csv"
    dest = data_dir / filename
    if dest.exists():
        raise FileExistsError(f"{filename} already exists; refusing to overwrite archived QQQ prices")
    frame = download_qqq_ohlc(start=start)
    if needed is not None:
        missing = sorted(needed - set(frame.index))
        if missing:
            preview = ", ".join(stamp.date().isoformat() for stamp in missing[:8])
            raise RuntimeError(
                "Stopping before PCA. The QQQ download does not cover "
                f"{len(missing)} feature dates, first: {preview}."
            )
    temporary = dest.with_suffix(".csv.partial")
    frame.to_csv(temporary)
    temporary.replace(dest)
    record = {
        "kind": "qqq_ohlc",
        "file": filename,
        "sha256": sha256_file(dest),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
        "created_at_utc": _utc_now(),
        "source": "yahoo:QQQ",
    }
    _append_manifest(record, data_dir)
    return {"path": dest, "sha256": record["sha256"], "downloaded": True, "file": filename}


def load_qqq_prices(path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = daily_index(frame.index)
    frame = frame.sort_index()
    frame.index.name = "date"
    if frame.index.has_duplicates:
        raise ValueError(f"{path} has duplicate QQQ dates")
    if "close_adjusted" not in frame.columns:
        raise ValueError(f"{path} is missing close_adjusted")
    return frame


def calendar_gaps(index, min_days: int = 5) -> list[dict]:
    """Gaps strictly longer than a four-day holiday weekend."""
    idx = daily_index(index).sort_values()
    if len(idx) < 2:
        return []
    delta = idx.to_series().diff().dt.days
    gaps = []
    for stamp, days in delta[delta > min_days].items():
        gaps.append({"date": pd.Timestamp(stamp).date().isoformat(), "gap_days": int(days)})
    return gaps


def split_by_cutoff(index, train_end=TRAIN_END):
    """Chronological split. Training is inclusive of train_end. No shared dates."""
    idx = daily_index(index)
    cutoff = pd.Timestamp(train_end).normalize()
    train = idx[idx <= cutoff]
    test = idx[idx > cutoff]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"cutoff {cutoff.date()} does not leave both a train and a test sample")
    if train.max() > cutoff or test.min() <= cutoff:
        raise AssertionError("train/test cutoff was not applied")
    if len(set(train) & set(test)) != 0:
        raise AssertionError("train and test dates overlap")
    return train, test


def audit_alignment(features: pd.DataFrame, prices: pd.DataFrame) -> dict:
    feature_index = daily_index(features.index)
    price_index = daily_index(prices.index)
    feature_dates = set(feature_index)
    price_dates = set(price_index)
    missing_price = sorted(feature_dates - price_dates)
    prices_outside = sorted(price_dates - feature_dates)
    not_midnight = int((pd.DatetimeIndex(features.index) != feature_index).sum())
    adjusted_gap = None
    if {"close", "close_adjusted"}.issubset(prices.columns):
        overlap = feature_index.intersection(price_index)
        both = prices.reindex(overlap)[["close", "close_adjusted"]].dropna()
        if len(both):
            adjusted_gap = float((both["close_adjusted"] - both["close"]).abs().max())
    return {
        "feature_start": feature_index.min().date().isoformat(),
        "feature_end": feature_index.max().date().isoformat(),
        "n_feature_rows": int(len(feature_index)),
        "n_feature_columns": int(features.shape[1]),
        "price_start": price_index.min().date().isoformat(),
        "price_end": price_index.max().date().isoformat(),
        "n_price_rows": int(len(price_index)),
        "n_intersection": int(len(feature_dates & price_dates)),
        "n_feature_dates_missing_qqq": int(len(missing_price)),
        "feature_dates_missing_qqq_head": [ts.date().isoformat() for ts in missing_price[:10]],
        "n_qqq_dates_outside_features": int(len(prices_outside)),
        "feature_duplicate_dates": int(feature_index.duplicated().sum()),
        "price_duplicate_dates": int(price_index.duplicated().sum()),
        "feature_monotonic": bool(feature_index.is_monotonic_increasing),
        "price_monotonic": bool(price_index.is_monotonic_increasing),
        "feature_timestamps_not_midnight": not_midnight,
        "feature_gaps_longer_than_5_days": calendar_gaps(feature_index),
        "max_abs_close_vs_adjusted_on_feature_dates": adjusted_gap,
        "qqq_columns": list(prices.columns),
    }


def audit_features(features: pd.DataFrame, train_end=TRAIN_END) -> dict:
    values = features.to_numpy(dtype=float)
    train, test = split_by_cutoff(features.index, train_end)
    train_frame = features.loc[train]
    train_std = train_frame.std(axis=0, ddof=0)
    zero_variance = [
        f"{asset}|{indicator}"
        for (asset, indicator), sigma in train_std.items()
        if float(sigma) <= ZERO_VARIANCE_TOLERANCE
    ]
    assets = list(dict.fromkeys(features.columns.get_level_values(0)))
    indicator_lists = {asset: features[asset].columns.tolist() for asset in assets}
    first_list = next(iter(indicator_lists.values()))
    identical_indicators = all(names == first_list for names in indicator_lists.values())
    column_hashes = {}
    duplicate_groups = []
    train_values = train_frame.to_numpy(dtype=float)
    for position, column in enumerate(train_frame.columns):
        digest = hashlib.sha256(np.ascontiguousarray(train_values[:, position]).tobytes()).hexdigest()
        column_hashes.setdefault(digest, []).append(f"{column[0]}|{column[1]}")
    for group in column_hashes.values():
        if len(group) > 1:
            duplicate_groups.append(group)
    return {
        "assets": assets,
        "n_indicators_per_asset": {asset: len(names) for asset, names in indicator_lists.items()},
        "indicators": first_list,
        "indicators_identical_across_assets": identical_indicators,
        "missing_values": int(np.isnan(values).sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": train.min().date().isoformat(),
        "train_end": train.max().date().isoformat(),
        "test_start": test.min().date().isoformat(),
        "test_end": test.max().date().isoformat(),
        "zero_variance_train_columns": zero_variance,
        "exact_duplicate_train_column_groups": duplicate_groups,
    }
