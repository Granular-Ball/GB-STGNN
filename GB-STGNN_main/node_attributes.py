"""Optional LamaH node attributes for the paired attribute ablation."""
from pathlib import Path
import json
import numpy as np

FLOW_FEATURES = ["flow_mean", "flow_std", "flow_median"]
BALL_CONSTRUCTION_VERSION = "fixed_composite_distance_no_component_penalty_v1"
TOPOLOGY_FEATURES = ["in_degree", "out_degree", "total_degree",
                     "upstream_reachable", "downstream_reachable", "topological_depth",
                     "coordinate_0", "coordinate_1"]
EXCLUDED_BALL_STATIC = {"forest_fra", "urban_fra"}


def processed_data_signature(processed_dir: Path):
    """Invalidate a ball cache when its source tensors are regenerated."""
    names = ["Q.npy", "mask.npy", "S.npy", "A.npy", "A_physical.npy",
             "station_coords.npy", "station_ids.npy", "split_indices.npz", "metadata.json"]
    signature = {}
    for name in names:
        path = processed_dir / name
        if path.exists():
            stat = path.stat()
            signature[name] = [stat.st_size, stat.st_mtime_ns]
    return signature


def load_static_attributes(processed_dir: Path):
    """Keep all static columns for the decoder, seven for ball construction."""
    static = np.load(processed_dir / "S.npy").astype(np.float32)
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    names = metadata.get("static_features")
    if static.ndim != 2 or not isinstance(names, list) or len(names) != static.shape[1]:
        raise ValueError("metadata static_features must match S.npy")
    if not EXCLUDED_BALL_STATIC.issubset(names):
        raise ValueError("The original attribute setting requires forest_fra and urban_fra")
    if not np.isfinite(static).all():
        raise ValueError("S.npy contains non-finite static attributes")
    return static, names


def _reachable_counts(adjacency):
    counts = np.zeros(len(adjacency), dtype=np.float32)
    for source in range(len(adjacency)):
        seen = set()
        stack = list(np.flatnonzero(adjacency[source]))
        while stack:
            node = int(stack.pop())
            if node not in seen:
                seen.add(node)
                stack.extend(np.flatnonzero(adjacency[node]))
        counts[source] = len(seen)
    return counts


def load_ball_attributes(processed_dir: Path, adjacency):
    """Reproduce the original seven static + eight topology/location columns."""
    static, names = load_static_attributes(processed_dir)
    keep = [i for i, name in enumerate(names) if name not in EXCLUDED_BALL_STATIC]
    graph = (adjacency > 0).copy()
    np.fill_diagonal(graph, False)
    in_degree = graph.sum(axis=0).astype(np.float32)
    out_degree = graph.sum(axis=1).astype(np.float32)
    remaining = in_degree.astype(int).copy()
    queue = list(np.flatnonzero(remaining == 0))
    depth = np.zeros(len(graph), dtype=np.float32)
    for node in queue:
        for downstream in np.flatnonzero(graph[node]):
            depth[downstream] = max(depth[downstream], depth[node] + 1)
            remaining[downstream] -= 1
            if remaining[downstream] == 0:
                queue.append(int(downstream))
    if len(queue) != len(graph):
        raise ValueError("Original topological depth requires an acyclic physical graph")
    coordinates = np.load(processed_dir / "station_coords.npy").astype(np.float32)
    if static.shape[0] != len(graph) or coordinates.shape != (len(graph), 2):
        raise ValueError("Static attributes and coordinates must match graph node order/count")
    if not np.isfinite(coordinates).all():
        raise ValueError("station_coords.npy contains non-finite coordinates")
    topology = np.column_stack([in_degree, out_degree, in_degree + out_degree,
                               _reachable_counts(graph.T), _reachable_counts(graph), depth,
                               coordinates])
    # The original feature order was static, flow statistics, topology/location.
    return static[:, keep], topology, [names[i] for i in keep], names
