#!/usr/bin/env python
"""Preprocess LamaH-CE daily data for GB-STGNN style experiments.

The script follows the preprocessing protocol described in the project PDF:
site filtering, daily time alignment, short-gap interpolation, supervised
masking for long gaps, physical river graph construction, train-only
normalization, and optional sliding
window export.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(message, flush=True)


DEFAULT_DYNAMIC_FEATURES = [
    "qobs",
    "prec",
    "2m_temp_mean",
    "total_et",
    "swe",
]

DEFAULT_STATIC_FEATURES = [
    "area_calc", "elev_mean", "slope_mean", "p_mean", "et0_mean",
    "frac_snow", "forest_fra", "lake_fra", "urban_fra",
]

@dataclass(frozen=True)
class MemberPaths:
    basin_root: str
    basin_timeseries_prefix: str
    gauge_timeseries_prefix: str
    catchment_attributes: str
    stream_dist: str | None
    gauge_hierarchy: str | None
    gauge_attributes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess LamaH-CE daily tar.gz into GB-STGNN tensors."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("dataset/2_LamaH-CE_daily.tar.gz"),
        help="Path to 2_LamaH-CE_daily.tar.gz.",
    )
    parser.add_argument(
        "--basin-root",
        default="B_basins_intermediate_all",
        choices=[
            "A_basins_total_upstrm",
            "B_basins_intermediate_all",
            "C_basins_intermediate_lowimp",
        ],
        help="LamaH-CE basin definition to use. C is low-impact and has gauge hierarchy.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/processed_lamah_ce"))
    parser.add_argument("--start-date", default="1981-01-01")
    parser.add_argument("--end-date", default="2017-12-31")
    parser.add_argument("--split-mode", choices=["date", "ratio"], default="date")
    parser.add_argument("--train-end", default="2009-12-31")
    parser.add_argument("--val-end", default="2013-12-31")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--input-window", type=int, default=30)
    parser.add_argument("--horizons", default="1,3,7", help="Comma-separated forecast horizons in days.")
    parser.add_argument("--use-log-q", action="store_true", help="Apply log1p to qobs before train-only standardization.")
    parser.add_argument("--max-missing-rate", type=float, default=0.90)
    parser.add_argument(
        "--max-train-missing-rate",
        type=float,
        default=0.90,
        help=(
            "Remove a station from train/validation/test when its training-period "
            "qobs missing rate is greater than this threshold."
        ),
    )
    parser.add_argument("--max-short-gap", type=int, default=3)
    parser.add_argument("--max-impact", choices=["low", "strong", "any"], default="any")
    parser.add_argument("--features", default=",".join(DEFAULT_DYNAMIC_FEATURES))
    parser.add_argument("--static-features", default=",".join(DEFAULT_STATIC_FEATURES))
    parser.add_argument("--corr-threshold", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--max-corr-lag", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--corr-top-k", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N loaded site CSVs.")
    parser.add_argument("--save-windows", action="store_true")
    parser.add_argument("--max-sites", type=int, default=None, help="Debug option for a small subset.")
    return parser.parse_args()


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), sep=";", na_values=["", "NA", "NaN", "-999", "-9999"])


def resolve_member_paths(members: set[str], basin_root: str) -> MemberPaths:
    basin_prefix = f"{basin_root}/2_timeseries/daily/"
    hierarchy = f"{basin_root}/1_attributes/Gauge_hierarchy.csv"
    stream_dist = f"{basin_root}/1_attributes/Stream_dist.csv"
    paths = MemberPaths(
        basin_root=basin_root,
        basin_timeseries_prefix=basin_prefix,
        gauge_timeseries_prefix="D_gauges/2_timeseries/daily/",
        catchment_attributes=f"{basin_root}/1_attributes/Catchment_attributes.csv",
        stream_dist=stream_dist if stream_dist in members else None,
        gauge_hierarchy=hierarchy if hierarchy in members else None,
        gauge_attributes="D_gauges/1_attributes/Gauge_attributes.csv",
    )
    required = [
        paths.basin_timeseries_prefix,
        paths.gauge_timeseries_prefix,
        paths.catchment_attributes,
        paths.gauge_attributes,
    ]
    missing = [p for p in required if not any(m.startswith(p) for m in members)]
    if missing:
        raise FileNotFoundError(f"Missing expected LamaH-CE members: {missing}")
    return paths


def ids_from_members(members: Iterable[str], prefix: str) -> set[int]:
    ids: set[int] = set()
    for name in members:
        if name.startswith(prefix) and name.endswith(".csv"):
            stem = Path(name).stem
            if stem.startswith("ID_"):
                ids.add(int(stem.replace("ID_", "")))
    return ids


def load_tables(
    archive: Path,
    paths: MemberPaths,
    selected_ids: set[int],
    include_basin_ts: bool = True,
    include_gauge_ts: bool = True,
    progress_label: str | None = None,
    progress_every: int = 50,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[str, pd.DataFrame]]:
    basin_ts: dict[int, pd.DataFrame] = {}
    gauge_ts: dict[int, pd.DataFrame] = {}
    tables: dict[str, pd.DataFrame] = {}
    wanted = {
        paths.catchment_attributes: "catchment_attributes",
        paths.gauge_attributes: "gauge_attributes",
    }
    if paths.stream_dist:
        wanted[paths.stream_dist] = "stream_dist"
    if paths.gauge_hierarchy:
        wanted[paths.gauge_hierarchy] = "gauge_hierarchy"

    loaded_site_files = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            table_key = wanted.get(name)
            if table_key:
                tables[table_key] = read_csv_bytes(tf.extractfile(member).read())
                continue
            if include_basin_ts and name.startswith(paths.basin_timeseries_prefix) and name.endswith(".csv"):
                site_id = int(Path(name).stem.replace("ID_", ""))
                if site_id in selected_ids:
                    basin_ts[site_id] = read_csv_bytes(tf.extractfile(member).read())
                    loaded_site_files += 1
                    if progress_label and loaded_site_files % progress_every == 0:
                        log(f"{progress_label}: loaded {loaded_site_files} site CSVs")
                continue
            if include_gauge_ts and name.startswith(paths.gauge_timeseries_prefix) and name.endswith(".csv"):
                site_id = int(Path(name).stem.replace("ID_", ""))
                if site_id in selected_ids:
                    gauge_ts[site_id] = read_csv_bytes(tf.extractfile(member).read())
                    loaded_site_files += 1
                    if progress_label and loaded_site_files % progress_every == 0:
                        log(f"{progress_label}: loaded {loaded_site_files} site CSVs")
    if progress_label:
        log(
            f"{progress_label}: done, basin={len(basin_ts)}, gauge={len(gauge_ts)}, "
            f"attribute_tables={len(tables)}"
        )
    return basin_ts, gauge_ts, tables


def date_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))


def normalize_quality_flags(q: pd.DataFrame) -> pd.Series:
    valid = q["qobs"].notna()
    for col in ["ckhs", "qceq", "qcol"]:
        if col in q.columns:
            valid &= q[col].fillna(0).eq(0) | q[col].fillna(0).eq(1)
    return valid


def select_sites(
    candidate_ids: list[int],
    gauge_attrs: pd.DataFrame,
    gauge_ts: dict[int, pd.DataFrame],
    start: str,
    end: str,
    max_missing_rate: float,
    max_impact: str,
) -> list[int]:
    date_range = pd.date_range(start, end, freq="D")
    attrs = gauge_attrs.set_index("ID")
    selected: list[int] = []
    for site_id in candidate_ids:
        if site_id not in gauge_ts or site_id not in attrs.index:
            continue
        if max_impact == "low" and str(attrs.loc[site_id].get("degimpact", "")).lower() not in {"l", "-"}:
            continue
        if max_impact == "strong" and str(attrs.loc[site_id].get("degimpact", "")).lower() == "s":
            continue
        q = gauge_ts[site_id].copy()
        q.index = date_index(q)
        q = q.reindex(date_range)
        valid = normalize_quality_flags(q) if "qobs" in q else pd.Series(False, index=date_range)
        missing_rate = 1.0 - float(valid.mean())
        if missing_rate <= max_missing_rate:
            selected.append(site_id)
    return selected


def prepare_site_frame(
    site_id: int,
    basin_df: pd.DataFrame,
    gauge_df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    max_short_gap: int,
) -> tuple[pd.DataFrame, pd.Series]:
    basin = basin_df.copy()
    basin.index = date_index(basin)
    basin = basin.reindex(date_range)
    gauge = gauge_df.copy()
    gauge.index = date_index(gauge)
    gauge = gauge.reindex(date_range)
    y_mask = normalize_quality_flags(gauge)
    qobs = gauge["qobs"].where(y_mask)
    frame = basin.drop(columns=[c for c in ["YYYY", "MM", "DD", "DOY"] if c in basin], errors="ignore")
    frame.insert(0, "qobs", qobs)
    raw_mask = frame.notna()
    frame = frame.interpolate(limit=max_short_gap, limit_direction="both")
    return frame, raw_mask["qobs"].rename(site_id)


def train_mask(dates: pd.DatetimeIndex, train_end: str) -> np.ndarray:
    return dates <= pd.Timestamp(train_end)


def make_split_indices(
    dates: pd.DatetimeIndex,
    split_mode: str,
    train_end: str,
    val_end: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    if split_mode == "date":
        train_indices = np.flatnonzero(dates <= pd.Timestamp(train_end))
        val_indices = np.flatnonzero((dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end)))
        test_indices = np.flatnonzero(dates > pd.Timestamp(val_end))
    else:
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"train/val/test ratios must sum to 1.0 for split-mode=ratio, got {total:.6f}"
            )
        n = len(dates)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        test_n = n - train_n - val_n
        if min(train_n, val_n, test_n) <= 0:
            raise ValueError(
                "Each ratio split must contain at least one timestep. "
                f"Got train={train_n}, val={val_n}, test={test_n}."
            )
        train_indices = np.arange(0, train_n, dtype=np.int64)
        val_indices = np.arange(train_n, train_n + val_n, dtype=np.int64)
        test_indices = np.arange(train_n + val_n, n, dtype=np.int64)

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "Empty split produced. Check --split-mode, --train-end, --val-end, or ratio arguments."
        )

    ranges = {
        "train": [dates[train_indices[0]].strftime("%Y-%m-%d"), dates[train_indices[-1]].strftime("%Y-%m-%d")],
        "val": [dates[val_indices[0]].strftime("%Y-%m-%d"), dates[val_indices[-1]].strftime("%Y-%m-%d")],
        "test": [dates[test_indices[0]].strftime("%Y-%m-%d"), dates[test_indices[-1]].strftime("%Y-%m-%d")],
    }
    return train_indices, val_indices, test_indices, ranges


def standardize_train_only(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = np.where(mask[..., None], values, np.nan)
    mean = np.nanmean(train_values, axis=(0, 1))
    std = np.nanstd(train_values, axis=(0, 1))
    std = np.where((std == 0) | np.isnan(std), 1.0, std)
    mean = np.where(np.isnan(mean), 0.0, mean)
    scaled = (values - mean) / std
    return scaled, mean, std


def standardize_static(static: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    numeric = static.apply(pd.to_numeric, errors="coerce")
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0).replace(0, 1.0).fillna(1.0)
    filled = numeric.fillna(mean)
    scaled = ((filled - mean) / std).fillna(0.0)
    stats = {c: {"mean": float(mean[c]), "std": float(std[c])} for c in scaled.columns}
    return scaled.to_numpy(dtype=np.float32), stats


def build_physical_adjacency(site_ids: list[int], hierarchy: pd.DataFrame | None, stream_dist: pd.DataFrame | None) -> np.ndarray:
    n = len(site_ids)
    index = {site_id: i for i, site_id in enumerate(site_ids)}
    adj = np.zeros((n, n), dtype=np.float32)
    dist = {}
    if stream_dist is not None and {"ID", "dist_hup"}.issubset(stream_dist.columns):
        dist = dict(zip(stream_dist["ID"].astype(int), pd.to_numeric(stream_dist["dist_hup"], errors="coerce")))
    scale = np.nanmedian(list(dist.values())) if dist else 1.0
    scale = float(scale) if math.isfinite(scale) and scale > 0 else 1.0
    if hierarchy is None:
        return adj
    for _, row in hierarchy.iterrows():
        src = int(row["ID"])
        dst_raw = row.get("NEXTDOWNID", 0)
        if src not in index or pd.isna(dst_raw):
            continue
        for token in str(dst_raw).split(","):
            token = token.strip()
            if not token or token == "0":
                continue
            dst = int(float(token))
            if dst in index:
                w = 1.0
                if src in dist and dst in dist:
                    w = float(np.exp(-abs(float(dist[src]) - float(dist[dst])) / scale))
                adj[index[src], index[dst]] = max(adj[index[src], index[dst]], w)
    return adj


def lag_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag > 0:
        x = x[:-lag]
        y = y[lag:]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_lag_correlation_adjacency(
    qobs: np.ndarray,
    train_indices: np.ndarray,
    max_lag: int,
    threshold: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = qobs[train_indices]
    n = q.shape[1]
    best_corr = np.zeros((n, n), dtype=np.float32)
    best_lag = np.zeros((n, n), dtype=np.int16)
    min_count = 30

    log(f"[6/7] Building lag-correlation graph for {n} sites, lags 0..{max_lag}")
    for lag in range(0, max_lag + 1):
        if lag > 0:
            src_series = q[:-lag]
            dst_series = q[lag:]
        else:
            src_series = q
            dst_series = q

        src_mask = np.isfinite(src_series).astype(np.float32)
        dst_mask = np.isfinite(dst_series).astype(np.float32)
        src_filled = np.nan_to_num(src_series, nan=0.0).astype(np.float32)
        dst_filled = np.nan_to_num(dst_series, nan=0.0).astype(np.float32)

        count = src_mask.T @ dst_mask
        sum_src = src_filled.T @ dst_mask
        sum_dst = src_mask.T @ dst_filled
        sum_src2 = (src_filled * src_filled).T @ dst_mask
        sum_dst2 = src_mask.T @ (dst_filled * dst_filled)
        sum_prod = src_filled.T @ dst_filled

        safe_count = np.maximum(count, 1.0)
        cov = sum_prod - (sum_src * sum_dst / safe_count)
        var_src = sum_src2 - (sum_src * sum_src / safe_count)
        var_dst = sum_dst2 - (sum_dst * sum_dst / safe_count)
        denom = np.sqrt(np.maximum(var_src * var_dst, 0.0))
        corr = np.divide(cov, denom, out=np.zeros_like(cov, dtype=np.float32), where=denom > 0)
        corr[count < min_count] = 0.0
        np.fill_diagonal(corr, 0.0)

        improve = np.abs(corr) > np.abs(best_corr)
        best_corr[improve] = corr[improve]
        best_lag[improve] = lag
        log(f"[6/7] Lag {lag}/{max_lag} correlation matrix done")

    adj = np.zeros((n, n), dtype=np.float32)
    lags = np.zeros((n, n), dtype=np.int16)
    for dst in range(n):
        candidates = np.flatnonzero(best_corr[:, dst] >= threshold)
        if candidates.size == 0:
            continue
        order = np.argsort(best_corr[candidates, dst])[::-1][:top_k]
        chosen = candidates[order]
        adj[chosen, dst] = best_corr[chosen, dst]
        lags[chosen, dst] = best_lag[chosen, dst]
    return adj, lags


def save_windows(
    out_dir: Path,
    split_name: str,
    split_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    input_window: int,
    horizons: list[int],
) -> None:
    max_h = max(horizons)
    indices = np.flatnonzero(split_mask)
    starts = indices[(indices >= input_window - 1) & (indices + max_h < len(split_mask))]
    windows_x = []
    windows_y = []
    windows_y_mask = []
    for end_idx in starts:
        target_idx = [end_idx + h for h in horizons]
        if not split_mask[target_idx].all():
            continue
        windows_x.append(x[end_idx - input_window + 1 : end_idx + 1])
        windows_y.append(y[target_idx])
        windows_y_mask.append(y_mask[target_idx])
    if not windows_x:
        return
    np.savez_compressed(
        out_dir / f"windows_{split_name}.npz",
        x=np.asarray(windows_x, dtype=np.float32),
        y=np.asarray(windows_y, dtype=np.float32),
        y_mask=np.asarray(windows_y_mask, dtype=bool),
        horizons=np.asarray(horizons, dtype=np.int16),
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.max_missing_rate <= 1.0:
        raise ValueError("--max-missing-rate must be in [0, 1]")
    if not 0.0 <= args.max_train_missing_rate <= 1.0:
        raise ValueError("--max-train-missing-rate must be in [0, 1]")
    features = split_csv_arg(args.features)
    horizons = [int(h) for h in split_csv_arg(args.horizons)]
    date_range = pd.date_range(args.start_date, args.end_date, freq="D")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[1/7] Indexing archive: {args.archive}")
    with tarfile.open(args.archive, "r:gz") as tf:
        members = set(tf.getnames())
    paths = resolve_member_paths(members, args.basin_root)
    basin_ids = ids_from_members(members, paths.basin_timeseries_prefix)
    gauge_ids = ids_from_members(members, paths.gauge_timeseries_prefix)
    candidate_ids = sorted(basin_ids & gauge_ids)
    if args.max_sites:
        candidate_ids = candidate_ids[: args.max_sites]
    log(f"[2/7] Found {len(candidate_ids)} candidate sites for {args.basin_root}")

    log("[3/7] Reading gauge series and attributes for filtering")
    _, gauge_ts, tables = load_tables(
        args.archive,
        paths,
        set(candidate_ids),
        include_basin_ts=False,
        include_gauge_ts=True,
        progress_label="[3/7]",
        progress_every=args.progress_every,
    )
    site_ids = select_sites(
        candidate_ids,
        tables["gauge_attributes"],
        gauge_ts,
        args.start_date,
        args.end_date,
        args.max_missing_rate,
        args.max_impact,
    )
    if not site_ids:
        raise RuntimeError("No sites passed the filtering rules. Relax --max-missing-rate or --max-impact.")
    log(f"[4/7] {len(site_ids)} sites passed filtering")

    log("[5/7] Reading basin forcing and selected gauge series")
    basin_ts, selected_gauge_ts, _ = load_tables(
        args.archive,
        paths,
        set(site_ids),
        include_basin_ts=True,
        include_gauge_ts=True,
        progress_label="[5/7]",
        progress_every=args.progress_every,
    )
    gauge_ts = selected_gauge_ts

    frames = []
    masks = []
    log("[6/7] Aligning dates, interpolating short gaps, and building masks")
    for pos, site_id in enumerate(site_ids, start=1):
        frame, q_mask = prepare_site_frame(
            site_id, basin_ts[site_id], gauge_ts[site_id], date_range, args.max_short_gap
        )
        frames.append(frame)
        masks.append(q_mask)
        if pos % args.progress_every == 0:
            log(f"[6/7] Aligned {pos}/{len(site_ids)} sites")
    log(f"[6/7] Aligned {len(site_ids)}/{len(site_ids)} sites")

    missing_features = sorted(set(features) - set().union(*(set(f.columns) for f in frames)))
    if missing_features:
        raise KeyError(f"Requested features are absent from the data: {missing_features}")

    log("[6/7] Stacking feature tensors")
    x_raw = np.stack([f.reindex(columns=features).to_numpy(dtype=np.float32) for f in frames], axis=1)
    y_raw = x_raw[:, :, features.index("qobs")] if "qobs" in features else np.stack(
        [f["qobs"].to_numpy(dtype=np.float32) for f in frames], axis=1
    )
    y_mask = np.stack([m.reindex(date_range).to_numpy(dtype=bool) for m in masks], axis=1)
    train_indices, val_indices, test_indices, split_ranges = make_split_indices(
        date_range,
        args.split_mode,
        args.train_end,
        args.val_end,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    train_missing_rates = 1.0 - y_mask[train_indices].mean(axis=0)
    keep_station = train_missing_rates <= args.max_train_missing_rate
    removed_site_ids = [
        int(site_id) for site_id, keep in zip(site_ids, keep_station) if not keep
    ]
    removed_train_missing_rates = [
        float(rate) for rate, keep in zip(train_missing_rates, keep_station) if not keep
    ]
    if not np.any(keep_station):
        raise RuntimeError(
            "No stations remain after applying --max-train-missing-rate="
            f"{args.max_train_missing_rate}."
        )
    if removed_site_ids:
        log(
            "[6/7] Removing "
            f"{len(removed_site_ids)} stations with training qobs missing rate > "
            f"{args.max_train_missing_rate:.2%}: {removed_site_ids}"
        )
        x_raw = x_raw[:, keep_station, :]
        y_raw = y_raw[:, keep_station]
        y_mask = y_mask[:, keep_station]
        site_ids = [site_id for site_id, keep in zip(site_ids, keep_station) if keep]
    else:
        log(
            "[6/7] No stations exceeded the training qobs missing-rate threshold "
            f"of {args.max_train_missing_rate:.2%}"
        )
    split_train = np.zeros(len(date_range), dtype=bool)
    split_train[train_indices] = True

    log("[6/7] Applying train-only normalization")
    x_for_scaling = x_raw.copy()
    q_feature_index = features.index("qobs") if "qobs" in features else None
    if args.use_log_q and q_feature_index is None:
        raise ValueError("--use-log-q requires qobs to be present in --features.")
    if args.use_log_q and q_feature_index is not None:
        q_values = x_for_scaling[:, :, q_feature_index]
        if np.nanmin(q_values) < 0:
            raise ValueError("qobs contains negative values; log1p normalization is not valid.")
        x_for_scaling[:, :, q_feature_index] = np.log1p(q_values)
    scaler_mask = split_train[:, None] & np.isfinite(x_for_scaling).all(axis=2)
    x_scaled, feature_mean, feature_std = standardize_train_only(x_for_scaling, scaler_mask)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    q_mean = float(feature_mean[q_feature_index]) if q_feature_index is not None else None
    q_std = float(feature_std[q_feature_index]) if q_feature_index is not None else None

    log("[6/7] Preparing static attributes")
    static_features = split_csv_arg(args.static_features)
    catchment = tables["catchment_attributes"].set_index("ID").reindex(site_ids)
    existing_static = [c for c in static_features if c in catchment.columns]
    static_scaled, static_stats = standardize_static(catchment[existing_static])

    log("[6/7] Building physical adjacency")
    adj_physical = build_physical_adjacency(
        site_ids,
        tables.get("gauge_hierarchy"),
        tables.get("stream_dist"),
    )
    adj_corr = np.zeros_like(adj_physical, dtype=np.float32)
    corr_lags = np.zeros_like(adj_physical, dtype=np.int16)
    adj = adj_physical.astype(np.float32).copy()
    np.fill_diagonal(adj, 1.0)

    train_split = np.zeros(len(date_range), dtype=bool)
    val_split = np.zeros(len(date_range), dtype=bool)
    test_split = np.zeros(len(date_range), dtype=bool)
    train_split[train_indices] = True
    val_split[val_indices] = True
    test_split[test_indices] = True

    log("[7/7] Saving processed tensors")
    dates_array = date_range.strftime("%Y-%m-%d").to_numpy()
    station_ids_array = np.asarray(site_ids, dtype=np.int32)
    features_array = np.asarray(features)
    static_features_array = np.asarray(existing_static)
    np.savez_compressed(
        args.out_dir / "lamah_ce_daily_tensors.npz",
        x=x_scaled,
        x_raw=x_raw.astype(np.float32),
        y=y_raw.astype(np.float32),
        y_mask=y_mask,
        adj=adj,
        adj_physical=adj_physical,
        adj_lag_corr=adj_corr,
        corr_lags=corr_lags,
        dates=dates_array,
        site_ids=station_ids_array,
        station_ids=station_ids_array,
        features=features_array,
        static=static_scaled,
        static_features=static_features_array,
    )
    np.save(args.out_dir / "X.npy", x_scaled)
    np.save(args.out_dir / "X_raw.npy", x_raw.astype(np.float32))
    np.save(args.out_dir / "Q.npy", y_raw.astype(np.float32))
    np.save(args.out_dir / "S.npy", static_scaled)
    np.save(args.out_dir / "A.npy", adj)
    np.save(args.out_dir / "A_physical.npy", adj_physical)
    np.save(args.out_dir / "A_lag_corr.npy", adj_corr)
    np.save(args.out_dir / "mask.npy", y_mask)
    np.save(args.out_dir / "station_ids.npy", station_ids_array)
    np.save(args.out_dir / "dates.npy", dates_array)
    np.save(args.out_dir / "features.npy", features_array)
    np.save(args.out_dir / "static_features.npy", static_features_array)
    np.savez_compressed(
        args.out_dir / "split_indices.npz",
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    pd.DataFrame({"site_id": site_ids}).to_csv(args.out_dir / "site_ids.csv", index=False)
    metadata = {
        "archive": str(args.archive),
        "basin_root": args.basin_root,
        "date_range": [args.start_date, args.end_date],
        "split_mode": args.split_mode,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "split_date_params": {
            "train_end": args.train_end,
            "val_end": args.val_end,
        },
        "splits": split_ranges,
        "features": features,
        "static_features": existing_static,
        "static_scaler": static_stats,
        "use_log_q": bool(args.use_log_q),
        "q_normalization": {
            "feature": "qobs" if q_feature_index is not None else None,
            "q_mean": q_mean,
            "q_std": q_std,
            "use_log_q": bool(args.use_log_q),
            "inverse": "expm1(z * q_std + q_mean)" if args.use_log_q else "z * q_std + q_mean",
        },
        "feature_scaler": {
            name: {"mean": float(feature_mean[i]), "std": float(feature_std[i])}
            for i, name in enumerate(features)
        },
        "filters": {
            "max_missing_rate": args.max_missing_rate,
            "max_train_missing_rate": args.max_train_missing_rate,
            "train_missing_rate_filter": {
                "comparison": "remove if training qobs missing rate > threshold",
                "removed_site_count": len(removed_site_ids),
                "removed_site_ids": removed_site_ids,
                "removed_train_missing_rates": removed_train_missing_rates,
            },
            "max_short_gap": args.max_short_gap,
            "max_impact": args.max_impact,
        },
        "graph": {
            "type": "physical_only",
            "corr_threshold": None,
            "max_corr_lag": None,
            "corr_top_k": None,
            "physical_edges": int((adj_physical > 0).sum()),
            "lag_corr_edges": 0,
        },
        "num_sites": len(site_ids),
        "num_days": len(date_range),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.save_windows:
        save_windows(args.out_dir, "train", train_split, x_scaled, y_raw, y_mask, args.input_window, horizons)
        save_windows(args.out_dir, "val", val_split, x_scaled, y_raw, y_mask, args.input_window, horizons)
        save_windows(args.out_dir, "test", test_split, x_scaled, y_raw, y_mask, args.input_window, horizons)

    log(f"Saved {len(site_ids)} sites and {len(date_range)} days to {args.out_dir}")
    log(f"Features: {features}")
    log(f"Physical edges: {(adj_physical > 0).sum()}, lag-correlation edges: 0 (disabled)")


if __name__ == "__main__":
    main()
