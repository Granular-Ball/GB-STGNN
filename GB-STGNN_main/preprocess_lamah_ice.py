"""Convert the extracted LamaH-Ice daily dataset to GB-STGNN tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_lamah_ce import (
    DEFAULT_DYNAMIC_FEATURES, build_physical_adjacency, date_index,
    make_split_indices, standardize_static,
)

STATIC_FEATURES = [
    "area_calc", "elev_mean", "slope_mean", "p_mean", "ref_et_mean",
    "frac_snow", "forest_fra", "lake_fra", "urban_fra",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/lamah_ice"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/processed_lamah_ice"))
    parser.add_argument("--basin-root", choices=["A_basins_total_upstrm", "B_basins_intermediate_all", "C_basins_intermediate_lowimp"], default="B_basins_intermediate_all")
    parser.add_argument("--gauge-series", choices=["daily_filtered", "daily"], default="daily_filtered")
    parser.add_argument("--start-date", default="1950-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--split-mode", choices=["ratio", "date"], default=None,
                        help="Default: fixed calendar-year date splits; ratio mode is opt-in.")
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-short-gap", type=int, default=3)
    args = parser.parse_args(argv)
    if args.max_short_gap < 0:
        parser.error("--max-short-gap must be nonnegative")
    args.split_mode = args.split_mode or "date"
    if args.split_mode == "date":
        args.train_end = args.train_end or "2007-12-31"
        args.val_end = args.val_end or "2014-12-31"
    if args.split_mode == "ratio":
        if args.train_end is not None or args.val_end is not None:
            parser.error("Split dates cannot be supplied with --split-mode ratio")
        ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
        if not all(0 < ratio < 1 for ratio in ratios) or not np.isclose(sum(ratios), 1):
            parser.error("Positive train/val/test ratios must sum to 1")
    return args


def read_table(path):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", na_values=["-999", "-9999"]).rename(columns={"id": "ID"})


def read_series(path, dates, require_coverage=False):
    frame = read_table(path)
    frame.index = date_index(frame)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Duplicate or unsorted dates in {path}")
    if require_coverage and (frame.index.min() > dates.min() or frame.index.max() < dates.max()):
        raise ValueError(f"Requested dates exceed meteorological coverage in {path}: {frame.index.min()} to {frame.index.max()}")
    return frame.reindex(dates)


def interpolate_short_gaps(frame, max_gap):
    """Fill only complete interior missing runs no longer than max_gap."""
    result = frame.copy()
    for name in frame:
        values = frame[name]
        missing = values.isna()
        groups = missing.ne(missing.shift()).cumsum()
        lengths = missing.groupby(groups).transform("sum")
        filled = values.interpolate(limit_area="inside")
        result[name] = values.where(~(missing & lengths.le(max_gap)), filled)
    return result


def infer_date_range(candidates, met_dir, q_dir):
    """Use shared forcing coverage intersected with overall observed runoff coverage."""
    met_starts, met_ends, q_starts, q_ends = [], [], [], []
    for site in candidates:
        met = read_table(met_dir / f"ID_{site}.csv")
        forcing_valid = np.isfinite(met[DEFAULT_DYNAMIC_FEATURES[1:]]).all(axis=1)
        met_dates = date_index(met)[forcing_valid]
        if not len(met_dates):
            raise ValueError(f"No valid meteorological coverage for station {site}")
        met_starts.append(met_dates.min())
        met_ends.append(met_dates.max())
        q = read_table(q_dir / f"ID_{site}.csv")
        q_dates = date_index(q)[np.isfinite(q.qobs) & q.qobs.ge(0)]
        if len(q_dates):
            q_starts.append(q_dates.min())
            q_ends.append(q_dates.max())
    if not q_starts:
        raise ValueError("No valid runoff observations found")
    start = max(max(met_starts), min(q_starts))
    end = min(min(met_ends), max(q_ends))
    if start > end:
        raise ValueError("Meteorological and observed runoff coverage do not overlap")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def preprocess(args):
    root = args.data_dir / args.basin_root
    met_dir = root / "2_timeseries/daily/meteorological_data"
    q_dir = args.data_dir / "D_gauges/2_timeseries" / args.gauge_series
    catchment = read_table(root / "1_attributes/Catchment_attributes.csv").set_index("ID")
    gauges = read_table(args.data_dir / "D_gauges/1_attributes/Gauge_attributes.csv").set_index("ID")
    met_ids = {int(p.stem[3:]) for p in met_dir.glob("ID_*.csv")}
    q_ids = {int(p.stem[3:]) for p in q_dir.glob("ID_*.csv")}
    candidates = sorted(met_ids & q_ids)
    if not candidates:
        raise RuntimeError("No matching gauge and meteorological files found")
    missing_attrs = sorted(set(candidates) - (set(catchment.index) & set(gauges.index)))
    if missing_attrs:
        raise ValueError(f"Missing required attributes for matched stations: {missing_attrs}")
    inferred_range = None
    if args.start_date is None or args.end_date is None:
        inferred_range = infer_date_range(candidates, met_dir, q_dir)
    start_date = args.start_date or inferred_range[0]
    end_date = args.end_date or inferred_range[1]
    if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise ValueError("Require start-date < end-date")
    if args.split_mode == "date" and not (pd.Timestamp(start_date) <= pd.Timestamp(args.train_end) < pd.Timestamp(args.val_end) < pd.Timestamp(end_date)):
        raise ValueError("Require start-date <= train-end < val-end < end-date")
    dates = pd.date_range(start_date, end_date, freq="D")
    train, val, test, ranges = make_split_indices(
        dates, args.split_mode, args.train_end, args.val_end,
        args.train_ratio, args.val_ratio, args.test_ratio,
    )
    print(f"Using source dates {start_date} to {end_date}; splits: {ranges}", flush=True)
    features = DEFAULT_DYNAMIC_FEATURES.copy()
    ids, inputs, targets, masks, feature_masks = [], [], [], [], []
    for site in candidates:
        q = read_series(q_dir / f"ID_{site}.csv", dates)["qobs"]
        # daily_filtered already applies the provider's quality screening.
        # Keep its original missing values out of both labels and flow statistics.
        q = q.where(np.isfinite(q) & q.ge(0))
        valid = q.notna()
        met = read_series(met_dir / f"ID_{site}.csv", dates, require_coverage=True)
        frame = met[features[1:]].copy()
        frame.insert(0, "qobs", q)
        frame = frame.where(np.isfinite(frame))
        # Interpolate each split separately so future validation/test observations
        # cannot fill training inputs near a boundary.
        filled = frame.copy()
        for index in [train, val, test]:
            filled.iloc[index] = interpolate_short_gaps(frame.iloc[index], args.max_short_gap)
        ids.append(site)
        inputs.append(filled.to_numpy(dtype=np.float32))
        targets.append(q.to_numpy(dtype=np.float32))
        masks.append(valid.to_numpy())
        feature_masks.append(frame.notna().to_numpy())
        if len(ids) % 20 == 0:
            print(f"Loaded {len(ids)} stations", flush=True)
    raw = np.stack(inputs, axis=1)
    q = np.stack(targets, axis=1)
    mask = np.stack(masks, axis=1)
    observed = np.stack(feature_masks, axis=1)
    train_values = np.where(observed[train], raw[train], np.nan)
    count = np.isfinite(train_values).sum(axis=(0, 1))
    mean = np.divide(np.nansum(train_values, axis=(0, 1)), count,
                     out=np.zeros(len(features), dtype=np.float32), where=count > 0)
    variance = np.divide(np.nansum((train_values - mean) ** 2, axis=(0, 1)), count,
                         out=np.zeros(len(features), dtype=np.float32), where=count > 0)
    std = np.sqrt(variance)
    std = np.where(std > 0, std, 1)
    x = np.nan_to_num((raw - mean) / std, nan=0., posinf=0., neginf=0.).astype(np.float32)
    static, static_stats = standardize_static(catchment.loc[ids, STATIC_FEATURES])
    coords = gauges.loc[ids, ["lon", "lat"]].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not np.isfinite(static).all():
        raise ValueError("Non-finite station coordinates or static attributes")
    hierarchy_path = root / "1_attributes/Gauge_hierarchy.csv"
    if not hierarchy_path.exists():
        # A uses total catchments but shares the same gauge network with B.
        hierarchy_path = args.data_dir / "B_basins_intermediate_all/1_attributes/Gauge_hierarchy.csv"
    physical = build_physical_adjacency(ids, read_table(hierarchy_path), None)
    adjacency = physical.copy()
    np.fill_diagonal(adjacency, 1)
    arrays = {"X": x, "Q": q, "mask": mask, "feature_mask": observed,
              "S": static, "A": adjacency, "A_physical": physical,
              "station_coords": coords, "station_ids": np.asarray(ids, dtype=np.int32),
              "dates": dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
              "features": np.asarray(features), "static_features": np.asarray(STATIC_FEATURES)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(args.out_dir / f"{name}.npy", values)
    np.savez_compressed(args.out_dir / "split_indices.npz", train_indices=train, val_indices=val, test_indices=test)
    metadata = {
        "dataset": "lamah_ice", "source_dir": str(args.data_dir.resolve()),
        "basin_root": args.basin_root, "gauge_series": args.gauge_series,
        "date_range": [start_date, end_date], "split_mode": args.split_mode, "splits": ranges,
        "date_range_selection": {"requested_start": args.start_date, "requested_end": args.end_date,
                                 "inferred_range": inferred_range,
                                 "method": "explicit date range" if inferred_range is None else "shared valid meteorological coverage intersected with overall valid runoff coverage"},
        "split_date_params": {"train_end": args.train_end, "val_end": args.val_end} if args.split_mode == "date" else None,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio} if args.split_mode == "ratio" else None,
        "num_sites": len(ids), "num_days": len(dates), "station_ids": ids,
        "features": features, "static_features": STATIC_FEATURES, "static_scaler": static_stats,
        "feature_scaler": {name: {"mean": float(mean[i]), "std": float(std[i])} for i, name in enumerate(features)},
        "use_log_q": False,
        "q_normalization": {"feature": "qobs", "q_mean": float(mean[0]), "q_std": float(std[0]), "use_log_q": False, "inverse": "z * q_std + q_mean"},
        "coordinates": {"crs": "EPSG:3057", "columns": ["lon", "lat"], "units": "m"},
        "quality_control": "provider daily_filtered; finite nonnegative observed qobs" if args.gauge_series == "daily_filtered" else "raw daily; finite nonnegative observed qobs only, no qc_flag screening",
        "normalization": "per-feature training observations only; missing inputs replaced by training mean (zero after scaling)",
        "interpolation": "interior complete gaps only, independently within each split; target observations unchanged",
        "filters": {"station_selection": "keep every runoff station with a matching basin meteorological file",
                    "max_missing_rate": None, "max_train_missing_rate": None,
                    "max_short_gap": args.max_short_gap, "require_observations_each_split": False,
                    "removed_site_ids": sorted(q_ids - met_ids)},
        "observed_runoff_days": {name: mask[index].sum(axis=0).tolist()
                                 for name, index in [("train", train), ("val", val), ("test", test)]},
        "graph": {"type": "physical_only", "physical_edges": int((physical > 0).sum()), "edge_weight": "1; no Stream_dist table"},
        "variable_conventions": "native LamaH-Ice values; ref_et_mean retained under its original name; no sign or unit conversion",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(ids)} stations, {len(dates)} days, {int((physical > 0).sum())} physical edges to {args.out_dir}")
    return metadata


if __name__ == "__main__":
    preprocess(parse_args())
