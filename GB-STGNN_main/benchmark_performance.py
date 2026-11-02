"""Short benchmark of cached vs uncached execution; never trains a full model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from models.gb_stgnn import GBSTGNNDiffusion
import train_gb_stgnn_variance_seed_topology_penalty as training

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/performance_benchmark.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.iterations < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error('Batch size, iterations and repeats must be positive; warmup must be nonnegative')
    device = torch.device(args.device)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    balls = ROOT / 'outputs/variance_seed_bfs_farthest_topology_penalty_sqrtN'
    dataset = ROOT / 'dataset/processed_lamah_ce'
    load = lambda name: torch.from_numpy(np.load(balls / name)).to(device)
    node, physical, similarity, quality = [load(name) for name in
        ('node_to_ball.npy', 'A_phy.npy', 'A_sim.npy', 'ball_quality.npy')]
    node = node.long()
    static = torch.from_numpy(np.load(dataset / 'S.npy')).to(device)
    hydro = torch.zeros(node.numel(), device=device)
    reference = GBSTGNNDiffusion(5, 64, 2, 2, 3, 0., static_dim=static.shape[1]).to(device)
    prepared = copy.deepcopy(reference)
    reference.gru.flatten_parameters()
    prepared.prepare_graph(node, physical, similarity, quality, static)
    x = torch.randn(args.batch_size, 30, node.numel(), 5, device=device)
    with torch.inference_mode():
        expected = reference(x, node, hydro, physical, similarity, quality, static)
        actual = prepared(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measure(function):
        for _ in range(args.warmup):
            function()
        sync()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _ in range(args.iterations):
                function()
            sync()
            times.append(1000 * (time.perf_counter() - start) / args.iterations)
        return statistics.median(times)

    results = {}
    for phase in ('forward_backward', 'inference'):
        def step(model, cached):
            if phase == 'forward_backward':
                model.train()
                model.zero_grad(set_to_none=True)
                prediction = model(x) if cached else model(x, node, hydro, physical, similarity, quality, static)
                prediction.square().mean().backward()
            else:
                model.eval()
                with torch.inference_mode():
                    model(x) if cached else model(x, node, hydro, physical, similarity, quality, static)
        uncached_ms = measure(lambda: step(reference, False))
        cached_ms = measure(lambda: step(prepared, True))
        results[phase] = {'uncached_ms_per_batch': uncached_ms, 'optimized_ms_per_batch': cached_ms,
                          'speedup': uncached_ms / cached_ms}
        print(phase, results[phase], flush=True)

    # Four synthetic batches, including CPU->GPU transfer and all test metrics.
    batches = [{'x': x.detach().cpu(), 'static': static.unsqueeze(0).expand(args.batch_size, -1, -1).cpu(),
                'y': torch.rand(args.batch_size, node.numel(), 3) * 100,
                'y_mask': torch.ones(args.batch_size, node.numel(), 3, dtype=torch.bool)} for _ in range(4)]
    fast_batches = [{k: v for k, v in batch.items() if k != 'static'} for batch in batches]

    def collected():
        pred, true, mask = training.evaluate_arrays(reference, batches, device, node, hydro, physical, similarity, quality)
        return training.metrics(pred, true, mask), {
            h: training.metrics(pred[:, :, i:i+1], true[:, :, i:i+1], mask[:, :, i:i+1])
            for i, h in enumerate((1, 3, 7))}

    def streaming():
        return training.evaluate_horizons(prepared, fast_batches, device, node, hydro, physical, similarity, quality, [1, 3, 7])

    expected, actual = collected(), streaming()
    for expected_group, actual_group in [(expected[0], actual[0])] + [(expected[1][h], actual[1][h]) for h in (1, 3, 7)]:
        for name in expected_group:
            np.testing.assert_allclose(actual_group[name], expected_group[name], rtol=2e-5, atol=2e-6)
    collected_ms, streaming_ms = measure(collected), measure(streaming)
    results['evaluation_four_batches'] = {'collected_ms': collected_ms, 'optimized_ms': streaming_ms,
                                          'speedup': collected_ms / streaming_ms}
    print('evaluation_four_batches', results['evaluation_four_batches'], flush=True)
    report = {'device': str(device), 'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
              'torch_version': torch.__version__, 'batch_size': args.batch_size,
              'nodes': node.numel(), 'balls': quality.numel(), 'sequence_length': 30,
              'hidden_dim': 64, 'iterations': args.iterations, 'repeats': args.repeats,
              'model_precision': 'float32', 'metric_accumulation': 'float64',
              'scope': 'synthetic inputs, real fixed graph; no optimizer update or full training; setup excluded',
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved benchmark to {args.output}', flush=True)


if __name__ == '__main__':
    main()
