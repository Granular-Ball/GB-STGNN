"""Build topology-penalty granular balls with graph-BFS initial centers.

Initialization differs from ``build_variance_seed_topology_penalty_balls.py``:

1. choose the maximum river-degree station as the first center;
2. compute every station's nearest undirected BFS distance to existing centers;
3. for the second and every later center, form a pool from the three largest
   nearest-distance levels;
4. choose the highest-degree candidate, breaking ties by larger BFS distance;
5. assign stations to their nearest BFS center.

Stations in a disconnected component with no selected center have infinite BFS
distance to every center.  Every such connected component becomes an
independent centerless initial granular ball.

During recursive splitting, low/high candidates are separated into induced
physical connected components.  A singleton component is then merged into its
most temporally synchronous physically adjacent component before split quality
gain is evaluated.  Truly isolated initial stations are left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from node_attributes import BALL_CONSTRUCTION_VERSION, FLOW_FEATURES, TOPOLOGY_FEATURES, load_ball_attributes, processed_data_signature

MIN_BALL_SIZE = 1


def load_required(directory: Path, name: str) -> np.ndarray:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return np.load(path)


def _nanmean_safe(values: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )


def _nanstd_safe(values: np.ndarray, axis: int) -> np.ndarray:
    mean = _nanmean_safe(values, axis)
    expanded = np.expand_dims(mean, axis)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    squared = np.where(valid, (values - expanded) ** 2, 0.0).sum(axis=axis)
    variance = np.divide(
        squared,
        count,
        out=np.full_like(squared, np.nan, dtype=float),
        where=count > 0,
    )
    return np.sqrt(variance)


def _column_quantiles_safe(
    values: np.ndarray, quantiles: list[float]
) -> np.ndarray:
    output = np.full((values.shape[1], len(quantiles)), np.nan, dtype=float)
    for station in range(values.shape[1]):
        valid = values[:, station][np.isfinite(values[:, station])]
        if valid.size:
            output[station] = np.quantile(valid, quantiles)
    return output


def _connected_components(A: np.ndarray) -> list[np.ndarray]:
    unseen = set(range(len(A)))
    components = []
    undirected = (A > 0) | ((A > 0).T)
    while unseen:
        queue = [unseen.pop()]
        component = []
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in np.flatnonzero(undirected[node]):
                neighbor = int(neighbor)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def _river_hop_distances(A: np.ndarray) -> np.ndarray:
    num_nodes = len(A)
    distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    undirected = (A > 0) | ((A > 0).T)
    for source in range(num_nodes):
        distances[source, source] = 0.0
        queue = [source]
        for node in queue:
            for neighbor in np.flatnonzero(undirected[node]):
                if not np.isfinite(distances[source, neighbor]):
                    distances[source, neighbor] = distances[source, node] + 1.0
                    queue.append(int(neighbor))
    finite = distances[np.isfinite(distances)]
    disconnected = float(finite.max() + 1.0) if finite.size else 1.0
    distances[~np.isfinite(distances)] = disconnected
    return distances


def _build_station_features(
    Q: np.ndarray,
    mask: np.ndarray,
    static: np.ndarray,
    topology: np.ndarray,
) -> np.ndarray:
    observed_flow = np.where(mask & np.isfinite(Q), Q, np.nan)
    flow_quantiles = _column_quantiles_safe(observed_flow, [0.5])
    flow_statistics = np.column_stack(
        [
            _nanmean_safe(observed_flow, 0),
            _nanstd_safe(observed_flow, 0),
            flow_quantiles,
        ]
    )
    parts = [static, flow_statistics, topology]
    features = np.nan_to_num(np.concatenate(parts, axis=1))
    scale = features.std(axis=0).clip(1e-6)
    return ((features - features.mean(axis=0)) / scale).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) < 1e-8 or np.std(b[valid]) < 1e-8:
        return 0.0
    return float(max(np.corrcoef(a[valid], b[valid])[0, 1], 0.0))


def _ball_stats(
    nodes: np.ndarray,
    features: np.ndarray,
    Q: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Calculate ball statistics with quality equal to temporal synchrony."""
    center = features[nodes].mean(axis=0)
    feature_distance = np.linalg.norm(features[nodes] - center, axis=1)
    flow = np.where(mask[:, nodes] & np.isfinite(Q[:, nodes]), Q[:, nodes], np.nan)
    mean_flow = _nanmean_safe(flow, 1)
    synchrony = (
        np.mean([_corr(flow[:, index], mean_flow) for index in range(len(nodes))])
        if len(nodes)
        else 0.0
    )
    return {
        "dist": feature_distance,
        "compactness": float(np.exp(-feature_distance.mean())),
        "synchrony": float(synchrony),
        "quality": float(synchrony),
    }


def _hydrological_distance_matrix(
    features: np.ndarray,
    river_dist: np.ndarray,
) -> np.ndarray:
    num_nodes = len(features)
    feature_dist = np.empty((num_nodes, num_nodes), dtype=np.float32)
    for source in range(num_nodes):
        feature_dist[source] = np.linalg.norm(features - features[source], axis=1)
    feature_scale = max(float(feature_dist.max()), 1e-6)
    river_scale = max(float(river_dist.max()), 1.0)
    return (
        feature_dist / feature_scale + river_dist / river_scale
    ).astype(np.float32)


def _variance_endpoint_split(
    nodes: np.ndarray,
    features: np.ndarray,
    hydro_dist: np.ndarray,
):
    if len(nodes) < 2 * MIN_BALL_SIZE:
        return None
    split_axis = int(np.argmax(np.var(features[nodes], axis=0)))
    ordered = nodes[np.argsort(features[nodes, split_axis])]
    seed_low = int(ordered[0])
    seed_high = int(ordered[-1])
    if seed_low == seed_high:
        return None
    cost_low = hydro_dist[nodes, seed_low]
    cost_high = hydro_dist[nodes, seed_high]
    assign_high = cost_high < cost_low
    assign_high[nodes == seed_low] = False
    assign_high[nodes == seed_high] = True
    child_low = nodes[~assign_high]
    child_high = nodes[assign_high]
    if len(child_low) < MIN_BALL_SIZE or len(child_high) < MIN_BALL_SIZE:
        return None
    return child_low, child_high, split_axis, seed_low, seed_high


def build_adaptive_ball_adjacency(
    A: np.ndarray,
    node_to_ball: np.ndarray,
    num_balls: int,
    Q_train: np.ndarray,
    mask_train: np.ndarray,
    topk_sim: int = 3,
):
    physical = np.zeros((num_balls, num_balls), dtype=np.float32)
    for source, target in zip(*np.nonzero(A > 0)):
        source_ball = node_to_ball[source]
        target_ball = node_to_ball[target]
        if source_ball != target_ball:
            physical[source_ball, target_ball] += A[source, target]

    similarity = np.zeros_like(physical)
    ball_series = []
    for ball_id in range(num_balls):
        in_ball = node_to_ball == ball_id
        series = np.where(mask_train[:, in_ball], Q_train[:, in_ball], np.nan)
        ball_series.append(_nanmean_safe(series, 1))
    for target_ball in range(num_balls):
        scores = sorted(
            [
                (_corr(ball_series[source_ball], ball_series[target_ball]), source_ball)
                for source_ball in range(num_balls)
                if source_ball != target_ball
            ],
            reverse=True,
        )[:topk_sim]
        for score, source_ball in scores:
            if score > 0:
                similarity[source_ball, target_ball] = score

    # Store raw components; normalization and quality modulation belong to the model.
    return physical, similarity


def add_feature_center_fallback_edges(
    A_phy: np.ndarray,
    A_sim: np.ndarray,
    node_to_ball: np.ndarray,
    station_features: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int]], float]:
    num_balls = A_phy.shape[0]
    physical = A_phy.copy()
    similarity = A_sim.copy()
    np.fill_diagonal(physical, 0.0)
    np.fill_diagonal(similarity, 0.0)
    physical_degree = ((physical > 0) | ((physical > 0).T)).sum(axis=1)
    similarity_degree = ((similarity > 0) | ((similarity > 0).T)).sum(axis=1)
    isolated = np.flatnonzero((physical_degree == 0) & (similarity_degree == 0))
    if num_balls < 2 or len(isolated) == 0:
        return similarity.astype(np.float32), [], 1.0

    centers = np.stack(
        [
            station_features[node_to_ball == ball_id].mean(axis=0)
            for ball_id in range(num_balls)
        ],
        axis=0,
    )
    center_dist = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    np.fill_diagonal(center_dist, np.inf)
    nearest_distances = center_dist.min(axis=1)
    positive = nearest_distances[
        np.isfinite(nearest_distances) & (nearest_distances > 0)
    ]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    sigma = max(sigma, 1e-6)

    records: list[dict[str, float | int]] = []
    added_pairs: set[tuple[int, int]] = set()
    for ball_id in isolated.tolist():
        nearest = int(np.argmin(center_dist[ball_id]))
        pair = tuple(sorted((int(ball_id), nearest)))
        if pair in added_pairs:
            continue
        distance = float(center_dist[ball_id, nearest])
        weight = float(np.exp(-distance / sigma))
        similarity[ball_id, nearest] = max(
            float(similarity[ball_id, nearest]), weight
        )
        similarity[nearest, ball_id] = max(
            float(similarity[nearest, ball_id]), weight
        )
        added_pairs.add(pair)
        records.append(
            {
                "isolated_ball": int(ball_id),
                "nearest_ball": nearest,
                "feature_center_distance": distance,
                "weight": weight,
            }
        )
    return similarity.astype(np.float32), records, sigma


def calculate_quality_components(
    node_to_ball: np.ndarray,
    features: np.ndarray,
    q_train: np.ndarray,
    mask_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate quality and temporal synchrony for each ball."""
    num_balls = int(node_to_ball.max()) + 1
    quality = np.zeros(num_balls, dtype=np.float32)
    synchrony = np.zeros(num_balls, dtype=np.float32)
    for ball_id in range(num_balls):
        nodes = np.flatnonzero(node_to_ball == ball_id)
        stats = _ball_stats(
            nodes,
            features,
            q_train,
            mask_train,
        )
        quality[ball_id] = stats["quality"]
        synchrony[ball_id] = stats["synchrony"]
    return quality, synchrony


def undirected_bfs_distance_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Return all-pairs hop distance; disconnected pairs remain infinite."""
    graph = (adjacency > 0) | ((adjacency > 0).T)
    np.fill_diagonal(graph, False)
    num_nodes = len(graph)
    distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    for source in range(num_nodes):
        distances[source, source] = 0.0
        queue = [source]
        for node in queue:
            unvisited_neighbors = np.flatnonzero(
                graph[node] & ~np.isfinite(distances[source])
            )
            for neighbor in unvisited_neighbors:
                distances[source, neighbor] = distances[source, node] + 1.0
                queue.append(int(neighbor))
    return distances


def river_degrees(adjacency: np.ndarray) -> np.ndarray:
    """Use total directed river degree, matching the original initializer."""
    graph = adjacency > 0
    np.fill_diagonal(graph, False)
    return graph.sum(axis=0).astype(np.int64) + graph.sum(axis=1).astype(np.int64)


def _largest_three_distance_levels(values: np.ndarray) -> np.ndarray:
    levels = np.unique(values)
    return levels[-min(3, len(levels)) :]


def select_bfs_farthest_centers(
    adjacency: np.ndarray,
    num_centers: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | float | bool | None]]]:
    """Select every center after the first from three farthest BFS layers."""
    num_nodes = len(adjacency)
    if not 1 <= num_centers <= num_nodes:
        raise ValueError(f"num_centers must be in [1, {num_nodes}]")

    degrees = river_degrees(adjacency)
    bfs_distances = undirected_bfs_distance_matrix(adjacency)
    first = int(np.flatnonzero(degrees == degrees.max())[0])
    centers = [first]
    records: list[dict[str, int | float | bool | None]] = [
        {
            "order": 0,
            "node": first,
            "degree": int(degrees[first]),
            "nearest_bfs_distance_before_selection": None,
            "was_unreachable_from_existing_centers": False,
        }
    ]

    while len(centers) < num_centers:
        nearest = bfs_distances[:, np.asarray(centers)].min(axis=1)
        available = np.ones(num_nodes, dtype=bool)
        available[np.asarray(centers)] = False
        available_distances = nearest[available]
        top_levels = _largest_three_distance_levels(available_distances)
        candidates = np.flatnonzero(available & np.isin(nearest, top_levels))

        highest_degree = int(degrees[candidates].max())
        candidates = candidates[degrees[candidates] == highest_degree]
        farthest = float(nearest[candidates].max())
        candidates = candidates[nearest[candidates] == farthest]
        chosen = int(candidates[0])
        centers.append(chosen)
        records.append(
            {
                "order": len(centers) - 1,
                "node": chosen,
                "degree": int(degrees[chosen]),
                "nearest_bfs_distance_before_selection": (
                    None if not np.isfinite(farthest) else farthest
                ),
                "was_unreachable_from_existing_centers": bool(
                    not np.isfinite(farthest)
                ),
            }
        )
    return np.asarray(centers, dtype=np.int64), bfs_distances, records


def assign_to_nearest_bfs_center(
    centers: np.ndarray,
    bfs_distances: np.ndarray,
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assign by BFS and give each uncovered component its own ball id."""
    distances = bfs_distances[:, centers]
    assignments = np.empty(len(distances), dtype=np.int64)
    used_fallback = ~np.isfinite(distances).any(axis=1)
    center_degrees = degrees[centers]

    # Stations reachable from at least one center retain ordinary nearest-BFS
    # assignment with the existing degree-aware tie break.
    for node in np.flatnonzero(~used_fallback):
        metric = distances[node]
        best = np.flatnonzero(metric == metric.min())
        best_degree = center_degrees[best].max()
        assignments[node] = int(best[center_degrees[best] == best_degree][0])

    # A component without a selected center has infinite BFS distance to every
    # center.  It becomes a separate centerless initial ball instead of being
    # attached to a geographically or hydrologically remote selected center.
    remaining = set(np.flatnonzero(used_fallback).tolist())
    fallback_component_count = 0
    while remaining:
        root = min(remaining)
        component = np.asarray(
            sorted(node for node in remaining if np.isfinite(bfs_distances[root, node])),
            dtype=np.int64,
        )
        assignments[component] = len(centers) + fallback_component_count
        remaining.difference_update(component.tolist())
        fallback_component_count += 1
    return assignments, used_fallback, fallback_component_count


def split_induced_connected_components(
    nodes: np.ndarray, adjacency: np.ndarray
) -> list[np.ndarray]:
    """Split a candidate child into connected components of its induced graph."""
    if len(nodes) == 0:
        return []
    local = (adjacency[np.ix_(nodes, nodes)] > 0)
    local = local | local.T
    np.fill_diagonal(local, False)
    unseen = set(range(len(nodes)))
    components: list[np.ndarray] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = [root]
        while stack:
            current = stack.pop()
            for neighbor in np.flatnonzero(local[current]):
                neighbor = int(neighbor)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(nodes[np.asarray(sorted(component), dtype=np.int64)])
    return components


def merge_singleton_components_by_synchrony(
    components: list[np.ndarray],
    q_train: np.ndarray,
    mask_train: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, int | float | list[int]]]]:
    """Merge recursive singleton components into their most synchronous neighbor.

    Only physically adjacent components are candidates, so every merged result
    remains connected.  Components are updated after each merge; this also lets
    two adjacent singleton components form a valid two-station component.
    """
    merged_components = [np.asarray(nodes, dtype=np.int64).copy() for nodes in components]
    graph = (adjacency > 0) | ((adjacency > 0).T)
    np.fill_diagonal(graph, False)
    records: list[dict[str, int | float | list[int]]] = []

    while True:
        singleton_index = next(
            (index for index, nodes in enumerate(merged_components) if len(nodes) == 1),
            None,
        )
        if singleton_index is None or len(merged_components) < 2:
            break

        node = int(merged_components[singleton_index][0])
        candidate_indices = [
            index
            for index, nodes in enumerate(merged_components)
            if index != singleton_index and np.any(graph[node, nodes])
        ]
        if not candidate_indices:
            # This can occur only for a truly isolated parent component.  Leave
            # it untouched rather than breaking the connectivity invariant.
            break

        station_flow = np.where(
            mask_train[:, node] & np.isfinite(q_train[:, node]),
            q_train[:, node],
            np.nan,
        )
        scored_candidates: list[tuple[int, float]] = []
        for index in candidate_indices:
            target_nodes = merged_components[index]
            target_flow = np.where(
                mask_train[:, target_nodes] & np.isfinite(q_train[:, target_nodes]),
                q_train[:, target_nodes],
                np.nan,
            )
            target_mean_flow = _nanmean_safe(target_flow, axis=1)
            scored_candidates.append(
                (index, float(_corr(station_flow, target_mean_flow)))
            )

        # Synchrony is primary.  Prefer the larger component and then its
        # smallest station id for deterministic ties.
        target_index, synchrony = max(
            scored_candidates,
            key=lambda item: (
                item[1],
                len(merged_components[item[0]]),
                -int(merged_components[item[0]].min()),
            ),
        )
        target_before = merged_components[target_index].copy()
        merged_components[target_index] = np.asarray(
            sorted(np.concatenate([target_before, np.asarray([node])]).tolist()),
            dtype=np.int64,
        )
        del merged_components[singleton_index]
        records.append(
            {
                "singleton_node": node,
                "target_nodes_before_merge": target_before.tolist(),
                "target_size_before_merge": int(len(target_before)),
                "synchrony": float(synchrony),
                "adjacent_candidate_count": int(len(candidate_indices)),
            }
        )

    return merged_components, records


def build_bfs_initialized_balls(
    q_train: np.ndarray,
    mask_train: np.ndarray,
    river_adjacency: np.ndarray,
    theta_q: float,
    static: np.ndarray,
    topology: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    features = _build_station_features(
        q_train, mask_train, static, topology
    )
    river_dist = _river_hop_distances(river_adjacency)
    hydro_dist = _hydrological_distance_matrix(features, river_dist)
    num_centers = max(1, int(np.sqrt(q_train.shape[1])))
    centers, bfs_distances, center_records = select_bfs_farthest_centers(
        river_adjacency, num_centers
    )
    degrees = river_degrees(river_adjacency)
    assignment, fallback_mask, fallback_component_count = assign_to_nearest_bfs_center(
        centers, bfs_distances, degrees
    )
    initial_ball_count = int(assignment.max()) + 1
    pending = [
        np.flatnonzero(assignment == ball_id).astype(np.int64)
        for ball_id in range(initial_ball_count)
        if np.any(assignment == ball_id)
    ]

    accepted: list[np.ndarray] = []
    split_records: list[dict] = []
    while pending:
        nodes = pending.pop(0)
        parent = _ball_stats(
            nodes,
            features,
            q_train,
            mask_train,
        )
        if parent["quality"] >= theta_q or len(nodes) < 2 * MIN_BALL_SIZE:
            accepted.append(nodes)
            continue
        proposal = _variance_endpoint_split(
            nodes,
            features,
            hydro_dist,
        )
        if proposal is None:
            accepted.append(nodes)
            continue
        child_low, child_high, split_axis, seed_low, seed_high = proposal
        low_stats = _ball_stats(
            child_low,
            features,
            q_train,
            mask_train,
        )
        high_stats = _ball_stats(
            child_high,
            features,
            q_train,
            mask_train,
        )
        low_components = split_induced_connected_components(
            child_low, river_adjacency
        )
        high_components = split_induced_connected_components(
            child_high, river_adjacency
        )
        premerge_children = low_components + high_components
        final_children, singleton_merge_records = (
            merge_singleton_components_by_synchrony(
                premerge_children,
                q_train,
                mask_train,
                river_adjacency,
            )
        )
        if any(len(child) < MIN_BALL_SIZE for child in final_children):
            accepted.append(nodes)
            continue
        final_child_stats = [
            _ball_stats(
                child,
                features,
                q_train,
                mask_train,
            )
            for child in final_children
        ]
        # A singleton has no within-ball relationship to evaluate.  Keep its
        # stored synchrony/quality unchanged for downstream model
        # use, but do not let its conventional quality of 1.0 make a split look
        # artificially profitable.
        final_child_gain_qualities = [
            0.0 if len(child) == 1 else float(stats["quality"])
            for child, stats in zip(final_children, final_child_stats)
        ]
        child_quality_sum = float(sum(final_child_gain_qualities))
        quality_gain = float(child_quality_sum - parent["quality"])
        if quality_gain > 0.0:
            split_records.append(
                {
                    "parent_size": int(len(nodes)),
                    "low_size": int(len(child_low)),
                    "high_size": int(len(child_high)),
                    "split_axis": int(split_axis),
                    "seed_low": int(seed_low),
                    "seed_high": int(seed_high),
                    "parent_quality": float(parent["quality"]),
                    "low_quality": float(low_stats["quality"]),
                    "high_quality": float(high_stats["quality"]),
                    "low_component_sizes": [
                        int(len(child)) for child in low_components
                    ],
                    "high_component_sizes": [
                        int(len(child)) for child in high_components
                    ],
                    "premerge_child_sizes": [
                        int(len(child)) for child in premerge_children
                    ],
                    "singleton_merges": singleton_merge_records,
                    "singleton_merge_count": int(len(singleton_merge_records)),
                    "final_child_sizes": [
                        int(len(child)) for child in final_children
                    ],
                    "final_child_qualities": [
                        float(stats["quality"]) for stats in final_child_stats
                    ],
                    "final_child_gain_qualities": final_child_gain_qualities,
                    "singleton_gain_quality": 0.0,
                    "final_child_count": int(len(final_children)),
                    "child_quality_sum": float(child_quality_sum),
                    "quality_gain": quality_gain,
                }
            )
            pending.extend(final_children)
        else:
            accepted.append(nodes)

    node_to_ball = np.empty(q_train.shape[1], dtype=np.int64)
    sizes = np.empty(len(accepted), dtype=np.int64)
    quality = np.empty(len(accepted), dtype=np.float32)
    distance_to_center = np.empty(q_train.shape[1], dtype=np.float32)
    for ball_id, nodes in enumerate(accepted):
        stats = _ball_stats(
            nodes,
            features,
            q_train,
            mask_train,
        )
        center = int(nodes[np.argmin(hydro_dist[np.ix_(nodes, nodes)].sum(axis=1))])
        node_to_ball[nodes] = ball_id
        sizes[ball_id] = len(nodes)
        quality[ball_id] = stats["quality"]
        distance_to_center[nodes] = hydro_dist[nodes, center]

    initialization = {
        "center_count": int(len(centers)),
        "center_nodes": centers.tolist(),
        "center_records": center_records,
        "second_center_candidate_distance_levels": 3,
        "later_center_candidate_distance_levels": 3,
        "selection_priority": "degree_desc_then_nearest_bfs_distance_desc",
        "assignment": "nearest_undirected_bfs_distance",
        "assignment_tie_break": "center_degree_desc_then_selection_order",
        "initial_ball_count": initial_ball_count,
        "disconnected_assignment_fallback": "independent centerless initial ball",
        "fallback_station_count": int(fallback_mask.sum()),
        "fallback_component_count": fallback_component_count,
        "centerless_initial_ball_count": fallback_component_count,
        "initial_ball_sizes": np.bincount(
            assignment, minlength=initial_ball_count
        ).tolist(),
    }
    return (
        node_to_ball,
        sizes,
        quality,
        distance_to_center,
        split_records,
        initialization,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build variance-seed topology balls with BFS-farthest initialization."
    )
    parser.add_argument(
        "--processed_dir", type=Path, default=Path("dataset/processed_lamah_ce")
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/variance_seed_bfs_farthest_topology_penalty_sqrtN"),
    )
    parser.add_argument("--theta_q", type=float, default=0.8)
    parser.add_argument("--topk_sim", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discharge = load_required(args.processed_dir, "Q.npy").astype(np.float32)
    mask = load_required(args.processed_dir, "mask.npy").astype(bool)
    adjacency = load_required(args.processed_dir, "A.npy").astype(np.float32)
    physical_path = args.processed_dir / "A_physical.npy"
    river_adjacency = (
        np.load(physical_path).astype(np.float32)
        if physical_path.exists()
        else adjacency
    )
    train_indices = np.load(args.processed_dir / "split_indices.npz")["train_indices"]
    q_train, mask_train = discharge[train_indices], mask[train_indices]
    static, topology, ball_static_names, model_static_names = load_ball_attributes(
        args.processed_dir, river_adjacency
    )

    (
        node_to_ball,
        sizes,
        _,
        hydro_dist_to_center,
        split_records,
        initialization,
    ) = build_bfs_initialized_balls(
        q_train,
        mask_train,
        river_adjacency,
        args.theta_q,
        static,
        topology,
    )
    station_features = _build_station_features(
        q_train, mask_train, static, topology
    )
    quality, synchrony = calculate_quality_components(
        node_to_ball,
        station_features,
        q_train,
        mask_train,
    )
    A_phy, A_sim = build_adaptive_ball_adjacency(
        A=adjacency,
        node_to_ball=node_to_ball,
        num_balls=len(sizes),
        Q_train=q_train,
        mask_train=mask_train,
        topk_sim=args.topk_sim,
    )
    A_sim, fallback_records, fallback_sigma = add_feature_center_fallback_edges(
        A_phy, A_sim, node_to_ball, station_features
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "node_to_ball.npy": node_to_ball,
        "ball_sizes.npy": sizes,
        "ball_quality.npy": quality,
        "ball_synchrony.npy": synchrony,
        "hydro_dist_to_ball_center.npy": hydro_dist_to_center,
        "A_phy.npy": A_phy,
        "A_sim.npy": A_sim,
    }
    for name, values in arrays.items():
        np.save(args.output_dir / name, values)

    river_edges = np.argwhere(river_adjacency > 0)
    cut_edges = int(
        sum(node_to_ball[source] != node_to_ball[target] for source, target in river_edges)
    )
    singleton_count = int(np.count_nonzero(sizes == 1))
    metadata = {
        "processed_dir": str(args.processed_dir.resolve()),
        "processed_data_signature": processed_data_signature(args.processed_dir),
        "method": "bfs_three_farthest_layers_degree_priority_then_variance_split",
        "quality_definition": "synchrony",
        "split_quality_gain_formula": (
            "sum(Q_gain(final connected child)) - Q_parent, where "
            "Q_gain(singleton) = 0 and Q_gain(non-singleton) = Q"
        ),
        "singleton_quality_policy": (
            "singletons contribute 0 only to split quality gain; stored "
            "synchrony and quality remain unchanged"
        ),
        "split_quality_gain_weighted_by_size": False,
        "candidate_child_connectivity_policy": (
            "split low/high candidates into induced physical connected components "
            "then merge each recursive singleton into its most synchronous "
            "physically adjacent component before quality gain and recursion"
        ),
        "recursive_singleton_merge_policy": (
            "highest training-flow synchrony among physically adjacent connected "
            "components; larger component then smaller station id break ties"
        ),
        "theta_q": args.theta_q,
        "min_ball_size": MIN_BALL_SIZE,
        "construction_version": BALL_CONSTRUCTION_VERSION,
        "distance_definition": "normalized feature distance + normalized undirected river hop distance",
        "split_assignment": "nearest composite distance; no component penalty",
        "adjacency_storage": "raw physical/similarity components without self-loops",
        "topk_sim": args.topk_sim,
        "num_nodes": int(q_train.shape[1]),
        "num_balls": int(len(sizes)),
        "single_node_balls": singleton_count,
        "single_node_ball_pct": 100.0 * singleton_count / len(sizes),
        "cut_river_edges": cut_edges,
        "accepted_recursive_splits": len(split_records),
        "station_features": ball_static_names + FLOW_FEATURES + TOPOLOGY_FEATURES,
        "model_static_features": model_static_names,
        "initialization": initialization,
        "split_records": split_records,
        "feature_center_fallback": {
            "sigma": fallback_sigma,
            "edges_added": len(fallback_records),
            "records": fallback_records,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved BFS-initialized balls to {args.output_dir}")
    print(
        f"nodes={q_train.shape[1]} | initial_centers={initialization['center_count']} | "
        f"balls={len(sizes)} | size min/mean/max="
        f"{sizes.min()}/{sizes.mean():.2f}/{sizes.max()}"
    )
    print(
        f"single-node balls={singleton_count} ({100.0 * singleton_count / len(sizes):.2f}%) | "
        f"cut river edges={cut_edges} | "
        f"centerless initial balls={initialization['centerless_initial_ball_count']} "
        f"({initialization['fallback_station_count']} stations)"
    )


if __name__ == "__main__":
    main()
