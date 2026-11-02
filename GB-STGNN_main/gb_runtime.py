"""GB-STGNN inference timing and multi-seed reporting."""
from __future__ import annotations
import csv
import json
import time
import numpy as np
import torch
from graph_timing import graph_region, measure_graph

PROTOCOL = "gb_stgnn_cuda_events_v1"

TIMING_KEYS = ["average_epoch_training_seconds", "single_sample_inference_seconds",
               "end_to_end_seconds", "graph_propagation_seconds", "run_end_to_end_seconds"]

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

@torch.inference_mode()
def benchmark(model, x_cpu, device, warmup, repeats):
    """Real batch-size-one latency; timings are separate from metric evaluation."""
    model.eval()
    x_cpu = x_cpu[:1].contiguous()
    x = x_cpu.to(device)
    for _ in range(warmup):
        model(x)
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    sync(device)
    inference = (time.perf_counter()-start)/repeats
    start = time.perf_counter()
    for _ in range(repeats):
        model(x_cpu.to(device)).cpu()
    sync(device)
    end_to_end = (time.perf_counter()-start)/repeats
    # Instrument graph layers only during this independent timing run.
    handles, contexts = [], {}
    def before(module, inputs):
        context = graph_region()
        context.__enter__()
        contexts.setdefault(id(module), []).append(context)
    def after(module, inputs, output):
        contexts[id(module)].pop().__exit__(None, None, None)
    classes = {"DiffConv"}
    for module in model.modules():
        if module.__class__.__name__ in classes:
            handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        with measure_graph(device) as collector:
            for _ in range(repeats):
                model(x)
            sync(device)
        total = collector.seconds()
        graph = total/repeats if total is not None else None
    finally:
        for handle in handles:
            handle.remove()
    return {"single_sample_inference_seconds": inference, "end_to_end_seconds": end_to_end,
            "graph_propagation_seconds": graph, "benchmark_batch_size": 1}

def mean_std(values):
    # Never hide failed/undefined seeds by dropping NaNs.
    if any(v is None for v in values):
        return {"mean": None, "std": None, "n": len(values)}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values)>1 else None, "n": len(values)}

def format_stat(stat):
    if stat["mean"] is None:
        return "N/A"
    return f"{stat['mean']:.6f} ± {stat['std']:.6f}" if stat["std"] is not None else f"{stat['mean']:.6f} ± N/A"

def report(results, root, expected_seeds):
    groups = {}
    for result in results:
        groups.setdefault((result["dataset"], result["model"]), []).append(result)
    summaries, rows, raw = [], [], []
    metrics = ["mae", "rmse", "nse", "kge"]
    for (dataset, model), group in groups.items():
        completed = [r["seed"] for r in group]
        if len(completed) != len(set(completed)):
            raise ValueError("Duplicate seed results")
        summary = {"dataset": dataset, "model": model, "seeds": completed, "expected_seeds": expected_seeds,
                   "complete": set(completed)==set(expected_seeds), "std_ddof": 1, "test_metrics": {},
                   "timing": {key: mean_std([r["timing"][key] for r in group]) for key in TIMING_KEYS},
                   "num_balls": mean_std([r["num_balls"] for r in group]),
                   "mean_ball_quality": mean_std([r["mean_ball_quality"] for r in group])}
        summary["num_nodes"] = mean_std([r["num_nodes"] for r in group])
        for horizon in ["overall", *group[0]["test_horizons"]]:
            values = [r["test_overall"] if horizon=="overall" else r["test_horizons"][horizon] for r in group]
            summary["test_metrics"][horizon] = {m: mean_std([v[m] for v in values]) for m in metrics}
            rows.append({"dataset": dataset, "model": model, "horizon": horizon,
                **{m: format_stat(summary["test_metrics"][horizon][m]) for m in metrics},
                **{k: format_stat(v) for k,v in summary["timing"].items()},
                "num_balls": format_stat(summary["num_balls"]), "mean_ball_quality": format_stat(summary["mean_ball_quality"]),
                "num_nodes": format_stat(summary["num_nodes"]),
                "completed_seeds": len(group), "complete": summary["complete"]})
            for r, value in zip(group, values):
                raw.append({"dataset": dataset, "model": model, "seed": r["seed"], "horizon": horizon,
                    **value, **{k:r["timing"][k] for k in TIMING_KEYS}, "num_nodes": r.get("num_nodes"),
                    "best_epoch": r.get("best_epoch"), "num_balls": r["num_balls"], "mean_ball_quality": r["mean_ball_quality"]})
        summaries.append(summary)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "seed_summary.json", summaries)
    save_json(root / "seed_results.json", results)
    for name, table in [("summary.csv", rows), ("per_seed_metrics.csv", raw)]:
        if table:
            with (root / name).open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines = ["# GB-STGNN: mean ± sample std (ddof=1)", "",
             "MAE/RMSE lower is better; NSE/KGE higher is better. Times are seconds.",
             "Single-sample timings use batch size 1 after warm-up; graph regions are separately measured with CUDA events or CPU wall time.",
             "End-to-end is CPU input to CPU prediction. Run end-to-end includes the whole experiment.",
             ""]
    for heading, columns in [("Prediction metrics", ["dataset","model","horizon",*metrics]),
                             ("Timing and ball metadata (overall)", ["dataset","model",*TIMING_KEYS,"num_nodes","num_balls","mean_ball_quality","completed_seeds","complete"])]:
        lines += [f"## {heading}", "", "| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
        for row in rows:
            if heading.startswith("Timing") and row["horizon"] != "overall":
                continue
            lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
        lines.append("")
    (root / "seed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries
