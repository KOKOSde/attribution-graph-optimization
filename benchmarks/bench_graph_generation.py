from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch

from attribution_graph_optimization import (
    generate_attribution_graph,
    get_native_extension_status,
)
from attribution_graph_optimization.graph_generation import canonicalize_nodes

BENCHMARK_RESULTS_PATH = Path('results/benchmark_results.json')



def _make_inputs(
    batch_size: int,
    num_layers: int,
    seq_len: int,
    hidden_dim: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(0)
    hidden_states = {}
    transcoders = {}
    for layer_offset in range(num_layers):
        layer_idx = 40 + layer_offset
        hidden_states[layer_idx] = torch.randn(
            batch_size,
            seq_len,
            hidden_dim,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        transcoders[layer_idx] = {
            'W_enc': torch.randn(
                feature_dim,
                hidden_dim,
                generator=generator,
                dtype=dtype,
                device=device,
            ),
            'b_enc': torch.zeros(feature_dim, dtype=dtype, device=device),
        }
    return hidden_states, transcoders



def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)



def _run_once(fn: Callable[[], object], device: torch.device) -> tuple[float, object]:
    _synchronize(device)
    start = time.perf_counter()
    result = fn()
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result



def _summarize(latencies_ms: List[float]) -> Dict[str, float]:
    mean_ms = statistics.fmean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    throughput_graphs_min = 60000.0 / mean_ms
    return {
        'mean_ms': mean_ms,
        'p50_ms': p50_ms,
        'throughput_graphs_min': throughput_graphs_min,
    }



def _compare_nodes(reference_nodes, candidate_nodes, atol: float) -> Dict[str, object]:
    ref = canonicalize_nodes(reference_nodes)
    cand = canonicalize_nodes(candidate_nodes)
    if len(ref) != len(cand):
        return {
            'matches': False,
            'reason': f'node count mismatch: {len(ref)} != {len(cand)}',
        }

    for index, (ref_node, cand_node) in enumerate(zip(ref, cand)):
        if ref_node[:3] != cand_node[:3]:
            return {
                'matches': False,
                'reason': f'node structure mismatch at index {index}: {ref_node[:3]} != {cand_node[:3]}',
            }
        if abs(ref_node[3] - cand_node[3]) > atol:
            return {
                'matches': False,
                'reason': f'node value mismatch at index {index}: {ref_node[3]} != {cand_node[3]}',
            }

    return {
        'matches': True,
        'reason': 'ok',
        'num_nodes': len(ref),
    }



def benchmark(
    batch_size: int,
    num_layers: int,
    seq_len: int,
    hidden_dim: int,
    feature_dim: int,
    top_k: int,
    threshold: float,
    iterations: int,
    warmup: int,
    device_name: str,
    dtype_name: str,
    atol: float,
    num_threads: int,
) -> Dict[str, object]:
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)
    if device.type == 'cpu' and num_threads > 0:
        torch.set_num_threads(num_threads)
    hidden_states, transcoders = _make_inputs(
        batch_size=batch_size,
        num_layers=num_layers,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        device=device,
        dtype=dtype,
    )

    implementations = {
        'baseline': lambda: generate_attribution_graph(
            hidden_states=hidden_states,
            transcoders=transcoders,
            top_k=top_k,
            threshold=threshold,
            implementation='baseline',
        ),
        'optimized': lambda: generate_attribution_graph(
            hidden_states=hidden_states,
            transcoders=transcoders,
            top_k=top_k,
            threshold=threshold,
            implementation='optimized',
        ),
        'extension': lambda: generate_attribution_graph(
            hidden_states=hidden_states,
            transcoders=transcoders,
            top_k=top_k,
            threshold=threshold,
            implementation='extension',
        ),
    }

    print('Configuration:')
    print(f'  device: {device}')
    print(f'  dtype: {dtype}')
    print(f'  batch_size: {batch_size}')
    print(f'  num_layers: {num_layers}')
    print(f'  seq_len: {seq_len}')
    print(f'  hidden_dim: {hidden_dim}')
    print(f'  feature_dim: {feature_dim}')
    print(f'  top_k: {top_k}')
    print(f'  threshold: {threshold}')
    print(f'  warmup: {warmup}')
    print(f'  iterations: {iterations}')
    print(f'  num_threads: {torch.get_num_threads()}')
    print(f'  native_extension: {get_native_extension_status()}')

    for _ in range(warmup):
        for fn in implementations.values():
            fn()

    timings = {}
    last_outputs = {}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for name, fn in implementations.items():
            latencies = []
            output = None
            for _ in range(iterations):
                elapsed_ms, output = _run_once(fn, device)
                latencies.append(elapsed_ms)
            timings[name] = _summarize(latencies)
            timings[name]['latencies_ms'] = latencies
            last_outputs[name] = output
    finally:
        if gc_was_enabled:
            gc.enable()

    correctness = {
        'optimized_vs_baseline': _compare_nodes(
            last_outputs['baseline']['nodes'],
            last_outputs['optimized']['nodes'],
            atol=atol,
        ),
        'extension_vs_optimized': _compare_nodes(
            last_outputs['optimized']['nodes'],
            last_outputs['extension']['nodes'],
            atol=atol,
        ),
        'summary_match': {
            'baseline_num_nodes': int(last_outputs['baseline']['summary']['num_nodes']),
            'optimized_num_nodes': int(last_outputs['optimized']['summary']['num_nodes']),
            'extension_num_nodes': int(last_outputs['extension']['summary']['num_nodes']),
        },
    }

    results = {
        'config': {
            'batch_size': batch_size,
            'num_layers': num_layers,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'feature_dim': feature_dim,
            'top_k': top_k,
            'threshold': threshold,
            'iterations': iterations,
            'warmup': warmup,
            'device': str(device),
            'dtype': dtype_name,
            'num_threads': torch.get_num_threads(),
        },
        'extension_status': get_native_extension_status(),
        'timings': timings,
        'correctness': correctness,
    }

    print('\nLatency summary (ms):')
    for name in ('baseline', 'optimized', 'extension'):
        summary = timings[name]
        print(
            f"  {name:<10} mean={summary['mean_ms']:.3f}  p50={summary['p50_ms']:.3f}  "
            f"throughput={summary['throughput_graphs_min']:.2f} graphs/min"
        )

    print('\nCorrectness:')
    for name, report in correctness.items():
        print(f'  {name}: {report}')

    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nWrote benchmark results to {BENCHMARK_RESULTS_PATH}')

    return results



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark attribution graph feature extraction paths.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--feature-dim', type=int, default=2048)
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--num-threads', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    benchmark(
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        top_k=args.top_k,
        threshold=args.threshold,
        iterations=args.iterations,
        warmup=args.warmup,
        device_name=args.device,
        dtype_name=args.dtype,
        atol=args.atol,
        num_threads=args.num_threads,
    )
