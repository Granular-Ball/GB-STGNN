"""Standalone GB-STGNN training for variance-seed topology-penalty balls."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lamah_dataset import LamaHCEDataset
from metrics import masked_kge, masked_mae, masked_nse, masked_rmse
from models.gb_stgnn import GBSTGNN
from models.graph_ops import graph_operator_config, validate_similarity_weight
from streaming_metrics import StreamingMetrics
from node_attributes import BALL_CONSTRUCTION_VERSION, FLOW_FEATURES, TOPOLOGY_FEATURES, EXCLUDED_BALL_STATIC, load_static_attributes, processed_data_signature
from gb_runtime import (
    PROTOCOL as REPORT_PROTOCOL,
    TIMING_KEYS,
    benchmark as benchmark_inference,
    report as write_shared_report,
)
from test_window_policy import select_test_windows


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


SUPPORTED_DATASETS = ("lamah_ce", "lamah_ice")


def quality_threshold_for_dataset(args: argparse.Namespace, dataset: str) -> float:
    if args.theta_q is not None:
        return args.theta_q
    return args.theta_q_ce if dataset == "lamah_ce" else args.theta_q_ice


def parse_args(argv=None, *, seed_count=5) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GB-STGNN.",allow_abbrev=False)
    parser.add_argument("--dataset", choices=["lamah_ce", "lamah_ice", "all"], default="lamah_ice",
                        help="all runs each supported dataset with the same five seeds.")
    parser.add_argument("--processed_dir", type=Path, default=None,
                        help="Processed directory; for all, parent of processed_<dataset> folders.")
    parser.add_argument("--balls_dir", type=Path, default=None,
                        help="Ball cache directory; for all, parent of per-dataset cache folders.")
    parser.add_argument("--input_len", type=int, default=30)
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_gru_layers", type=int, default=2)
    parser.add_argument("--num_diffconv_layers", type=int, default=2)
    parser.add_argument("--theta_q", "--theta-q", type=float, default=None,
                        help="Default: CE=0.8, Ice=0.7; explicit value overrides either/all datasets.")
    parser.add_argument("--theta_q_ce", "--theta-q-ce", type=float, default=0.8,
                        help="CE quality threshold; used unless --theta_q is specified.")
    parser.add_argument("--theta_q_ice", "--theta-q-ice", type=float, default=0.7,
                        help="Ice quality threshold; used unless --theta_q is specified.")
    parser.add_argument("--lambda_sim", type=float, default=0.1)
    parser.add_argument("--topk_sim", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--loss", choices=["mae", "huber"], default="huber")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--benchmark_repeats", "--benchmark-repeats", type=int, default=20,
                        help="Batch-size-one timing repetitions, using batch size one.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warm-up forwards before latency measurement.")
    parser.add_argument("--seeds", type=int, nargs=seed_count,
                        default=[2026, 2027, 2028, 2029, 2030][:seed_count])
    args = parser.parse_args(argv)
    for name in ("theta_q", "theta_q_ce", "theta_q_ice"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            parser.error(f"{name} must be finite and in [0, 1]")
    if args.theta_q is None and args.dataset != "all":
        args.theta_q = quality_threshold_for_dataset(args, args.dataset)
    args.processed_dir = args.processed_dir or (
        Path("dataset") if args.dataset == "all" else Path(f"dataset/processed_{args.dataset}")
    )
    suffix = "_all" if args.dataset == "all" else ("_lamah_ice" if args.dataset == "lamah_ice" else "")
    args.balls_dir = args.balls_dir or Path(f"outputs/gb_stgnn{suffix}")
    args.save_dir = args.save_dir or Path(f"runs/gb_stgnn_diffconv_quality_2hop{suffix}")
    if len(set(args.seeds)) != seed_count or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error(f"Provide {seed_count} distinct seeds in [0, 2**32)")
    if args.benchmark_repeats <= 0 or args.warmup < 0:
        parser.error("benchmark_repeats must be positive and warmup must be nonnegative")
    if args.epochs <= 0 or args.batch_size <= 0 or args.input_len <= 0:
        parser.error("epochs, batch_size and input_len must be positive")
    return args


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"--horizons must contain positive integers, got {value!r}")
    return horizons


def choose_device(requested: str, gpu_id: int | None) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda" and gpu_id is not None:
        count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= count:
            raise ValueError(f"--gpu_id must be in [0, {count - 1}], got {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(requested)


def load_array(directory: Path, name: str, dtype=None) -> np.ndarray:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    values = np.load(path, allow_pickle=True)
    return values.astype(dtype) if dtype is not None else values


def load_quality_definition(balls_dir: Path) -> str:
    """Load the quality definition recorded for the prebuilt balls."""
    metadata_path = balls_dir / "metadata.json"
    if not metadata_path.exists():
        return "synchrony"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(
        metadata.get("quality_definition", "synchrony")
    )


def ensure_balls_match_training_args(args: argparse.Namespace) -> None:
    """Build granular balls when the cached files do not match this run."""
    metadata_path = args.balls_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = None

    expected = {
        "theta_q": float(args.theta_q),
        "min_ball_size": 1,
        "construction_version": BALL_CONSTRUCTION_VERSION,
        "topk_sim": int(args.topk_sim),
    }
    _, static_names = load_static_attributes(args.processed_dir)
    expected_features = [name for name in static_names if name not in EXCLUDED_BALL_STATIC] + FLOW_FEATURES + TOPOLOGY_FEATURES
    required_files = (
        "node_to_ball.npy", "ball_sizes.npy", "ball_quality.npy",
        "hydro_dist_to_ball_center.npy", "A_phy.npy", "A_sim.npy",
    )

    def values_match() -> bool:
        if metadata is None or not all(
            (args.balls_dir / name).exists() for name in required_files
        ):
            return False
        source = metadata.get("processed_dir")
        if source is not None and Path(source).resolve() != args.processed_dir.resolve():
            return False
        if getattr(args, "dataset", "lamah_ce") == "lamah_ice" and source is None:
            return False
        signature = metadata.get("processed_data_signature")
        if signature is not None and signature != processed_data_signature(args.processed_dir):
            return False
        if getattr(args, "dataset", "lamah_ce") == "lamah_ice" and signature is None:
            return False
        try:
            for key, expected_value in expected.items():
                actual = metadata.get(key)
                if isinstance(expected_value, float):
                    if actual is None or not np.isclose(float(actual), expected_value):
                        return False
                elif actual != expected_value:
                    return False
            return (
                metadata.get("quality_definition") == "synchrony"
                and metadata.get("station_features") == expected_features
            )
        except (TypeError, ValueError):
            return False

    if values_match():
        print(
            f"Reusing granular balls from {args.balls_dir} | "
            f"theta_q={args.theta_q:g} | num_balls={metadata.get('num_balls')}"
        )
        return

    if getattr(args, "lock_prebuilt_balls", False):
        raise ValueError("Locked ball cache does not match this dataset/configuration; refusing to rebuild it")

    builder_path = Path(__file__).resolve().parent / (
        "build_bfs_farthest_variance_seed_topology_penalty_balls.py"
    )
    if not builder_path.exists():
        raise FileNotFoundError(f"Missing granular-ball builder: {builder_path}")
    print(
        f"Building granular balls in {args.balls_dir} for theta_q={args.theta_q:g}; "
        "cached metadata is missing or does not match this run."
    )
    command = [
        sys.executable,
        str(builder_path),
        "--processed_dir", str(args.processed_dir.resolve()),
        "--output_dir", str(args.balls_dir.resolve()),
        "--theta_q", str(args.theta_q),
        "--topk_sim", str(args.topk_sim),
    ]
    result = subprocess.run(
        command,
        cwd=builder_path.parent,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    rebuilt = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not np.isclose(float(rebuilt.get("theta_q", -1.0)), args.theta_q):
        raise RuntimeError(
            f"Built granular-ball metadata has theta_q={rebuilt.get('theta_q')}, "
            f"expected {args.theta_q}"
        )


def prepare_prebuilt_balls(args: argparse.Namespace):
    node_to_ball = load_array(args.balls_dir, "node_to_ball.npy", np.int64)
    A_phy = load_array(args.balls_dir, "A_phy.npy", np.float32)
    A_sim = load_array(args.balls_dir, "A_sim.npy", np.float32)
    ball_sizes = load_array(args.balls_dir, "ball_sizes.npy", np.int64)
    ball_quality = load_array(args.balls_dir, "ball_quality.npy", np.float32)
    hydro_dist = load_array(args.balls_dir, "hydro_dist_to_ball_center.npy", np.float32)
    num_nodes = np.load(args.processed_dir / "X.npy", mmap_mode="r").shape[1]
    num_balls = len(ball_sizes)
    if node_to_ball.shape != (num_nodes,) or hydro_dist.shape != (num_nodes,):
        raise ValueError("Station-level granular-ball arrays do not match the processed node count")
    if any(matrix.shape != (num_balls, num_balls) for matrix in (A_phy, A_sim)):
        raise ValueError(f"Ball adjacency matrices must have shape {(num_balls, num_balls)}")
    if ball_quality.shape != (num_balls,):
        raise ValueError("ball_quality.npy does not match the ball count")
    if not np.array_equal(np.bincount(node_to_ball, minlength=num_balls), ball_sizes):
        raise ValueError("ball_sizes.npy does not match node_to_ball.npy")
    return node_to_ball, A_phy, A_sim, ball_sizes, ball_quality, hydro_dist


def masked_loss(y_pred, y_true, mask, loss_name):
    valid = mask.bool() & torch.isfinite(y_pred) & torch.isfinite(y_true)
    if not torch.any(valid):
        raise RuntimeError("Batch has no valid target values after applying y_mask.")
    if loss_name == "mae":
        return torch.mean(torch.abs(y_pred[valid] - y_true[valid]))
    return F.smooth_l1_loss(y_pred[valid], y_true[valid])


def metrics(pred, true, mask):
    return {
        "mae": float(masked_mae(pred, true, mask).item()),
        "rmse": float(masked_rmse(pred, true, mask).item()),
        "nse": float(masked_nse(pred, true, mask).item()),
        "kge": float(masked_kge(pred, true, mask).item()),
    }


@torch.inference_mode()
def evaluate_arrays(
    model, loader, device, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality,
):
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        predictions.append(predict_batch(model, x, batch, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality).cpu())
        targets.append(batch["y"])
        masks.append(batch["y_mask"])
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def predict_batch(model, x, batch, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality):
    if model.graph_is_prepared:
        return model(x, static=batch.get("static"))
    return model(x, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality, static=batch.get("static"))


@torch.inference_mode()
def evaluate_streaming(model, loader, device, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality):
    model.eval()
    accumulator = StreamingMetrics()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        prediction = predict_batch(model, x, batch, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality)
        accumulator.update(prediction, batch["y"].to(device, non_blocking=True),
                           batch["y_mask"].to(device, non_blocking=True))
    return accumulator.compute()


def evaluate(
    model, loader, device, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality,
):
    return evaluate_streaming(
        model, loader, device, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality,
    )[0]


def evaluate_horizons(
    model, loader, device, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality, horizons,
):
    overall, horizon_metrics = evaluate_streaming(
        model, loader, device, node_to_ball, hydro_dist, A_phy, A_sim, ball_quality,
    )
    if len(horizon_metrics) != len(horizons):
        raise ValueError("Prediction horizons do not match requested horizons")
    return overall, dict(zip(horizons, horizon_metrics))


def make_loader(processed_dir, split, input_len, horizons, batch_size, shuffle, num_workers,
                seed=2026, test_dataset_name=None):
    dataset = LamaHCEDataset(
        processed_dir=processed_dir, split=split, input_len=input_len,
        horizons=horizons, target_mode="raw", return_adj=False,
        return_static=False, return_dates=False, memory_map=True,
    )
    if split == "test" and test_dataset_name is not None:
        dataset.test_window_policy = select_test_windows(dataset, test_dataset_name, "GB-STGNN")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=num_workers > 0,
    )


def count_nonzero_offdiag(matrix):
    values = matrix.detach().cpu().numpy() if isinstance(matrix, torch.Tensor) else np.asarray(matrix)
    if values.ndim == 3:
        values = values[0]
    mask = values > 0
    np.fill_diagonal(mask, False)
    return int(mask.sum())


def print_graph_statistics(node_to_ball, A_phy, A_sim):
    print("--------------------------------------------------")
    print("Graph Statistics (Test Phase):")
    print(f"M (granular balls) = {len(np.unique(node_to_ball))}")
    print(f"|E_phy| = {count_nonzero_offdiag(A_phy)}")
    print(f"|E_sim| = {count_nonzero_offdiag(A_sim)}")
    print("--------------------------------------------------")


def prepare_station_graph(processed_dir: Path):
    """Original river graph, expressed through the identity node grouping."""
    graph_file = "A_physical.npy" if (processed_dir / "A_physical.npy").exists() else "A.npy"
    physical = load_array(processed_dir, graph_file, np.float32)
    if (physical.ndim != 2 or physical.shape[0] != physical.shape[1]
            or not np.isfinite(physical).all() or np.any(physical < 0)):
        raise ValueError("Station adjacency must be a finite, nonnegative square matrix")
    count = physical.shape[0]
    return (np.arange(count, dtype=np.int64), physical, np.zeros_like(physical),
            np.ones(count, dtype=np.int64), np.ones(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32), graph_file)


def prepare_physical_only_graph(processed_dir: Path, node_to_ball: np.ndarray, num_balls: int):
    """Aggregate verified physical station edges, never station/ball similarity edges."""
    path = processed_dir / "A_physical.npy"
    if not path.is_file():
        raise FileNotFoundError(f"physical_only requires raw physical edges: {path}; A.npy may contain similarity edges")
    station = load_array(processed_dir, "A_physical.npy", np.float32)
    if (station.shape != (len(node_to_ball), len(node_to_ball))
            or not np.isfinite(station).all() or np.any(station < 0)):
        raise ValueError("Invalid original physical station graph")
    physical = np.zeros((num_balls, num_balls), dtype=np.float32)
    source, target = np.nonzero(station > 0)
    source_ball, target_ball = node_to_ball[source], node_to_ball[target]
    inter_ball = source_ball != target_ball
    np.add.at(physical, (source_ball[inter_ball], target_ball[inter_ball]), station[source[inter_ball], target[inter_ball]])
    return physical, np.zeros_like(physical)


def run(args: argparse.Namespace, seed: int, *, add_self_loops: bool = False,
        decoder_static: bool = False, ablation: str = "full") -> None:
    if ablation not in {"full", "station_graph", "no_ball_feedback", "no_quality", "physical_only"}:
        raise ValueError(f"Unknown ablation: {ablation}")
    run_start = time.perf_counter()
    data_metadata = json.loads((args.processed_dir / "metadata.json").read_text(encoding="utf-8"))
    data_name = data_metadata.get("dataset", "lamah_ce")
    if data_name != getattr(args, "dataset", "lamah_ce"):
        raise ValueError(f"--dataset={args.dataset} does not match processed data {data_name} in {args.processed_dir}")
    if not 0 <= args.theta_q <= 1:
        raise ValueError("--theta_q must be in [0, 1]")
    validate_similarity_weight(args.lambda_sim)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    horizons = parse_horizons(args.horizons)
    device = choose_device(args.device, args.gpu_id)
    if ablation != "station_graph":
        ensure_balls_match_training_args(args)
    features = load_array(args.processed_dir, "features.npy")
    raw_input_dim = len(features)
    input_dim = raw_input_dim
    static_values, static_names = load_static_attributes(args.processed_dir)
    static_dim = static_values.shape[1] if decoder_static else 0
    horizon_dim = len(horizons)
    if ablation == "station_graph":
        node_np, A_phy_np, A_sim_np, sizes, quality_np, hydro_np, graph_file = prepare_station_graph(args.processed_dir)
        quality_definition = "not applicable: station graph uses identity grouping, unit gates"
    else:
        node_np, A_phy_np, A_sim_np, sizes, quality_np, hydro_np = prepare_prebuilt_balls(args)
        quality_definition = load_quality_definition(args.balls_dir)
    if ablation == "physical_only":
        A_phy_np, A_sim_np = prepare_physical_only_graph(args.processed_dir, node_np, len(sizes))
    effective_lambda_sim = 0.0 if ablation == "physical_only" else args.lambda_sim
    num_balls = int(len(sizes))
    single_node_balls = int(np.count_nonzero(sizes == 1))
    single_node_ball_pct = 100.0 * single_node_balls / num_balls if num_balls else 0.0
    print(
        "Adaptive granular balls | "
        f"num_balls={num_balls} | avg_size={sizes.mean():.2f} | "
        f"max_size={sizes.max()} | min_size={sizes.min()} | avg_quality={quality_np.mean():.4f}"
    )
    print(
        "Single-node granular balls | "
        f"count={single_node_balls} | pct_of_balls={single_node_ball_pct:.2f}%"
    )
    node = torch.from_numpy(node_np).long().to(device)
    A_phy = torch.from_numpy(A_phy_np).float().to(device)
    A_sim = torch.from_numpy(A_sim_np).float().to(device)
    hydro = torch.from_numpy(hydro_np).float().to(device)
    quality = torch.from_numpy(quality_np).float().to(device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best_gb_stgnn.pt"
    config = vars(args).copy()
    config.update({
        "protocol": REPORT_PROTOCOL,
        "model": "GB-STGNN",
        "ablation": ablation,
        "node_only_decoder": ablation == "no_ball_feedback",
        "quality_modulation": ablation not in {"no_quality", "station_graph"},
        "graph_resolution": "station" if ablation == "station_graph" else "granular_ball",
        "lambda_sim": effective_lambda_sim,
        "requested_lambda_sim": args.lambda_sim,
        "dataset": data_name,
        "date_range": data_metadata.get("date_range"),
        "splits": data_metadata.get("splits"),
        "num_nodes": int(len(node_np)),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "benchmark_repeats": getattr(args, "benchmark_repeats", 20),
        "warmup": getattr(args, "warmup", 3),
        "timing_definition": {
            "training": "data loading, transfers, forward, backward, optimizer; excludes validation/checkpoint saving",
            "inference": "batch-size-one full model forward with preloaded-device input, synchronized wall time",
            "end_to_end": "batch-size-one CPU input transfer, full model forward, output CPU transfer; excludes disk loading",
            "graph": "sum ball DiffConv-layer CUDA event elapsed times (CPU: region wall time), per batch-size-one forward; includes linear mixing, excludes GRU, pooling, node projection, residual activation and decoder",
            "run": "data/model setup, optional cached-ball construction, training, validation, checkpoint I/O, test evaluation and timing benchmarks; excludes Python process startup",
        },
        "seed": seed,
        "processed_dir": str(args.processed_dir), "balls_dir": str(args.balls_dir),
        "save_dir": str(args.save_dir), "horizons": horizons, "input_dim": input_dim,
        "raw_input_dim": raw_input_dim, "horizon_dim": horizon_dim,
        "node_input_features": features.tolist(),
        "static_dim": static_dim,
        "model_static_features": static_names if decoder_static else [],
        "decoder_static_enabled": decoder_static,
        "station_features": (json.loads((args.balls_dir / "metadata.json").read_text(encoding="utf-8"))["station_features"]
                             if ablation != "station_graph" else []),
        "ball_construction_version": BALL_CONSTRUCTION_VERSION,
        "ball_representation": {
            "aggregation": "mean_pool",
            "description": "Arithmetic mean of time-encoded node hidden states inside each granular ball.",
        },
        "graph_operator": {
            **graph_operator_config(effective_lambda_sim, add_self_loops=add_self_loops),
            "applied_in": "models.graph_ops.build_transition_matrices",
        },
        "execution": {
            "fixed_graph_precomputed": True,
            "fixed_static_cached_on_device": decoder_static,
            "dataset_memory_map": "copy-on-write",
            "streaming_metrics": "float64 sufficient statistics; no full prediction collection",
            "per_batch_loss_cpu_sync": False,
            "precision": "float32 model; float64 metric accumulation",
        },
        "ball_size_summary": {
            "quality_definition": quality_definition,
            "num_balls": num_balls, "avg_size": float(sizes.mean()),
            "max_size": int(sizes.max()), "min_size": int(sizes.min()),
            "single_node_balls": single_node_balls,
            "single_node_ball_pct": single_node_ball_pct,
            "avg_quality": float(quality_np.mean()),
        },
    })
    if ablation in {"no_quality", "station_graph"}:
        config["graph_operator"]["quality_modulation"] = "disabled: G_fwd/rev = P_fwd/rev"
    if ablation == "station_graph":
        config["graph_operator"]["physical_weights"] = f"original station adjacency from {graph_file}"
        config["graph_operator"]["similarity_weights"] = "disabled: no ball similarity edges"
        config["ball_representation"]["description"] = "Identity grouping for station diffusion; no ball construction or compression"
    if ablation == "no_ball_feedback":
        config["timing_definition"]["graph"] = "0: graph branch removed from prediction path"
    if ablation == "physical_only":
        config["graph_operator"].update({
            "physical_weights": "sum original A_physical.npy station edge weights between distinct balls",
            "similarity_weights": "disabled: zero matrix; no station similarity, ball similarity, or isolated-node/ball fallback edges",
            "isolated_policy": "remain physically isolated; no similarity reconnection; zero-hop and residual retained",
        })
    (args.save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_loader = make_loader(args.processed_dir, "train", args.input_len, horizons, args.batch_size, True, args.num_workers, seed)
    val_loader = make_loader(args.processed_dir, "val", args.input_len, horizons, args.batch_size, False, args.num_workers, seed)
    test_loader = make_loader(args.processed_dir, "test", args.input_len, horizons, args.batch_size, False, args.num_workers, seed,
                              test_dataset_name=data_name)
    model = GBSTGNN(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        static_dim=static_dim,
        lambda_sim=effective_lambda_sim,
        add_self_loops=add_self_loops,
        num_gru_layers=args.num_gru_layers,
        num_diffconv_layers=args.num_diffconv_layers,
        horizon_dim=horizon_dim,
        dropout=args.dropout,
        node_only_decoder=ablation == "no_ball_feedback",
        quality_modulation=ablation not in {"no_quality", "station_graph"},
    ).to(device)
    model.prepare_graph(node, A_phy, A_sim, quality,
                        static=torch.from_numpy(static_values) if decoder_static else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" ({torch.cuda.get_device_name(device)})"
    print(
        f"Training GBSTGNN (two-step DiffConv) on {device_label} | input_dim={input_dim}, horizons={horizons}"
    )
    print(f"batch_size={args.batch_size}, hidden_dim={args.hidden_dim}, loss={args.loss}, num_workers={args.num_workers}")
    print(f"theta_q={args.theta_q}, topk_sim={args.topk_sim}, lambda_sim={effective_lambda_sim}")
    best_val_mae = float("inf")
    training_epoch_times = []
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.train()
        epoch_loss = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y, y_mask = batch["y"].to(device, non_blocking=True), batch["y_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = masked_loss(prediction, y, y_mask, args.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = x.shape[0]
            epoch_loss.add_(loss.detach() * batch_count)
            count += batch_count
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_epoch_times.append(time.perf_counter() - start)
        total_loss = epoch_loss.item()
        val = evaluate(
            model, val_loader, device, node, hydro, A_phy, A_sim, quality,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"Epoch {epoch:03d} | train_loss={total_loss/max(count,1):.6f} | "
            f"val_MAE={val['mae']:.6f} | val_RMSE={val['rmse']:.6f} | "
            f"val_NSE={val['nse']:.6f} | val_KGE={val['kge']:.6f} | epoch_time={time.perf_counter()-start:.2f}s"
        )
        if val["mae"] < best_val_mae:
            best_val_mae = val["mae"]
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "val_metrics": val, "epoch": epoch, "node_to_ball": node_np,
                "A_phy": A_phy_np, "A_sim": A_sim_np,
                "ball_sizes": sizes, "ball_quality": quality_np,
                "hydro_dist_to_ball": hydro_np,
            }, best_path)
            print(f"Saved best model to {best_path}")

    print(
        f"Average training time over {len(training_epoch_times)} epochs: "
        f"{sum(training_epoch_times) / max(len(training_epoch_times), 1):.4f}s/epoch "
        f"| total_training_time={sum(training_epoch_times):.4f}s "
        f"| excludes_validation_and_test=True"
    )

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Fixed graph buffers are not checkpointed. Recreate the inference snapshot
    # before metrics/latency evaluation, outside every timed forward.
    model.prepare_graph(node, A_phy, A_sim, quality,
                        static=torch.from_numpy(static_values) if decoder_static else None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_start = time.perf_counter()
    test, test_horizons = evaluate_horizons(
        model, test_loader, device, node, hydro, A_phy, A_sim, quality, horizons,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - test_start
    samples = len(test_loader.dataset)
    print(f"Test Overall | MAE={test['mae']:.6f} | RMSE={test['rmse']:.6f} | NSE={test['nse']:.6f} | KGE={test['kge']:.6f}")
    print(f"Batched test evaluation time (includes loading, transfers and metrics): total={elapsed:.4f}s | per_sample={elapsed/max(samples,1):.6f}s | samples={samples}")
    # Measure batch-size-one inference and graph propagation.
    # Static attributes and graph operators are already resident on the device.
    example = test_loader.dataset[0]["x"].unsqueeze(0)
    timing = benchmark_inference(model, example, device,
                                 getattr(args, "warmup", 3),
                                 getattr(args, "benchmark_repeats", 20))
    if ablation == "no_ball_feedback":
        timing["graph_propagation_seconds"] = 0.0
    elif timing["graph_propagation_seconds"] is None:
        raise RuntimeError("No GB-STGNN DiffConv graph regions were measured")
    timing.update({
        "average_epoch_training_seconds": float(statistics.mean(training_epoch_times)),
        "run_end_to_end_seconds": time.perf_counter() - run_start,
    })
    print(
        f"Single sample model inference: {timing['single_sample_inference_seconds']:.6f}s | "
        f"end_to_end={timing['end_to_end_seconds']:.6f}s | "
        f"graph_propagation={timing['graph_propagation_seconds']:.6f}s | batch_size=1"
    )
    print_graph_statistics(node_np, A_phy_np, A_sim_np)
    report = {
        "complete": True,
        "protocol": REPORT_PROTOCOL,
        "model": "GB-STGNN",
        "ablation": ablation,
        "node_only_decoder": config["node_only_decoder"],
        "quality_modulation": config["quality_modulation"],
        "graph_resolution": config["graph_resolution"],
        "dataset": data_name,
        "date_range": data_metadata.get("date_range"),
        "splits": data_metadata.get("splits"),
        "num_nodes": int(len(node_np)),
        "test_samples": samples,
        "test_window_policy": test_loader.dataset.test_window_policy,
        "timing": timing,
        "epoch_training_seconds": training_epoch_times,
        "test_evaluation": {
            "seconds": elapsed,
            "seconds_per_sample": elapsed / max(samples, 1),
            "includes": "batched data loading, transfers, model forward and metric accumulation; not single-sample latency",
        },
        "seed": seed,
        "decoder_static_enabled": decoder_static,
        "static_dim": static_dim,
        "model_static_features": static_names if decoder_static else [],
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["val_metrics"],
        "test_overall": test,
        "test_horizons": {str(h): test_horizons[h] for h in horizons},
        "num_balls": None if ablation == "station_graph" else num_balls,
        "num_graph_nodes": num_balls,
        "mean_ball_quality": None if ablation == "station_graph" else float(quality_np.mean()),
        "graph_operator": config["graph_operator"],
    }
    (args.save_dir / "test_results.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item()), encoding="utf-8"
    )
    for horizon in horizons:
        result = test_horizons[horizon]
        print(f"Test Horizon h={horizon} | MAE={result['mae']:.6f} | RMSE={result['rmse']:.6f} | NSE={result['nse']:.6f} | KGE={result['kge']:.6f}")


def run_single_seed(args: argparse.Namespace, seed: int, directory: Path,
                    *, add_self_loops: bool = False,
                    decoder_static: bool = False, ablation: str = "full") -> None:
    """Create a fresh model and optimizer for one seed."""
    seed_args = argparse.Namespace(**vars(args))
    seed_args.save_dir = directory
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.log").open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            run(seed_args, seed, add_self_loops=add_self_loops,
                decoder_static=decoder_static, ablation=ablation)


def summarize_seed_results(results: list[dict], seeds: list[int], save_dir: Path) -> dict:
    """Report metrics, latency and ball statistics across seeds."""
    if len(results) != len(seeds) or {result["seed"] for result in results} != set(seeds):
        raise ValueError("A complete five-seed result set is required")
    horizons = sorted(results[0]["test_horizons"], key=int)
    operator = results[0]["graph_operator"]
    decoder_static = results[0].get("decoder_static_enabled", False)
    for result in results:
        if (set(result["test_horizons"]) != set(horizons)
                or result["graph_operator"] != operator
                or result.get("decoder_static_enabled", False) != decoder_static):
            raise ValueError("Seed runs must share graph parameters, attributes and horizons")
    names = ("mae", "rmse", "nse", "kge")
    for horizon in ["overall", *horizons]:
        group = [result["test_overall"] if horizon == "overall" else result["test_horizons"][horizon]
                 for result in results]
        for result, metrics in zip(results, group):
            if any(not math.isfinite(float(metrics[name])) for name in names):
                raise ValueError(f"Non-finite test metrics for seed {result['seed']}, horizon {horizon}")
    # Historical results contain no timing measurements. Keep them explicitly
    # unavailable instead of relabeling batch-evaluation times as latency.
    enriched = []
    for result in results:
        timing = {key: result.get("timing", {}).get(key) for key in TIMING_KEYS}
        for key, value in timing.items():
            if value is not None and (not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"Invalid {key} for seed {result['seed']}")
        enriched.append({**result, "model": "GB-STGNN",
                         "dataset": result.get("dataset", "unknown"),
                         "num_nodes": result.get("num_nodes"), "timing": timing})
    if len({item["dataset"] for item in enriched}) != 1:
        raise ValueError("Seed runs must use the same dataset")
    for key in ("date_range", "splits"):
        if any(item.get(key) != enriched[0].get(key) for item in enriched):
            raise ValueError("Seed runs must share the same date range and split dates")
    summary = write_shared_report(enriched, save_dir, seeds)[0]
    summary.update({"n": len(seeds), "decoder_static_enabled": decoder_static,
                    "graph_operator": operator,
                    "protocol": REPORT_PROTOCOL,
                    "timing_available": all(all(item["timing"][key] is not None for key in TIMING_KEYS)
                                            for item in enriched)})
    save_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the historical single-object GB JSON API; its metric/timing
    # fields and formatted CSV/Markdown use the shared reporting schema.
    (save_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    # Compatibility filename for callers expecting the earlier per-seed CSV.
    (save_dir / "seed_metrics.csv").write_bytes((save_dir / "per_seed_metrics.csv").read_bytes())
    print("\n" + (save_dir / "seed_summary.md").read_text(encoding="utf-8"), flush=True)
    return summary


def run_seeds(args: argparse.Namespace) -> list[dict]:
    if args.dataset == "all":
        raise ValueError("Use run_all_datasets for --dataset all")
    root = args.save_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Do not leave an earlier aggregate looking current if a new run fails.
    for name in ("seed_summary.json", "seed_summary.md", "seed_metrics.csv", "seed_results.json",
                 "summary.csv", "per_seed_metrics.csv"):
        (root / name).unlink(missing_ok=True)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    (root / "multi_seed_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    results = []
    for index, seed in enumerate(args.seeds, 1):
        directory = root / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "test_results.json"
        result_path.unlink(missing_ok=True)
        print(f"\n===== Random seed {seed} ({index}/{len(args.seeds)}) =====", flush=True)
        run_single_seed(args, seed, directory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["seed"] != seed:
            raise ValueError(f"Unexpected seed in {result_path}")
        results.append(result)
        (root / "seed_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summarize_seed_results(results, args.seeds, root)
    print(f"Five-seed reports saved to {root}", flush=True)
    return results


def dataset_run_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    """Resolve an all-dataset run without mutating the shared arguments."""
    if args.dataset != "all" or dataset not in SUPPORTED_DATASETS:
        raise ValueError("Expected --dataset all and a supported dataset")
    current = argparse.Namespace(**vars(args))
    current.dataset = dataset
    current.theta_q = quality_threshold_for_dataset(args, dataset)
    current.seeds = list(args.seeds)
    current.processed_dir = args.processed_dir / f"processed_{dataset}"
    current.balls_dir = args.balls_dir / dataset
    current.save_dir = args.save_dir / dataset
    return current


def run_all_datasets(args: argparse.Namespace) -> None:
    root = args.save_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("seed_summary.json", "seed_summary.md", "seed_results.json",
                 "summary.csv", "per_seed_metrics.csv"):
        (root / name).unlink(missing_ok=True)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings["datasets"] = list(SUPPORTED_DATASETS)
    settings["quality_thresholds"] = {
        dataset: quality_threshold_for_dataset(args, dataset)
        for dataset in SUPPORTED_DATASETS
    }
    (root / "multi_dataset_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    results = []
    for dataset in SUPPORTED_DATASETS:
        current = dataset_run_args(args, dataset)
        current.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== Dataset {dataset} | seeds {current.seeds} =====", flush=True)
        with (current.save_dir / "multi_seed.log").open("w", encoding="utf-8") as log_file:
            with redirect_stdout(Tee(sys.stdout, log_file)):
                results.extend(run_seeds(current))
        (root / "seed_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # The shared writer groups by dataset: never pool CE and Ice statistics.
    write_shared_report(results, root, args.seeds)
    print((root / "seed_summary.md").read_text(encoding="utf-8"), flush=True)
    print(f"All-dataset reports saved to {root}", flush=True)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_name = "multi_seed.log"
    with (args.save_dir / log_name).open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            if args.dataset == "all":
                run_all_datasets(args)
            else:
                run_seeds(args)


if __name__ == "__main__":
    main()
