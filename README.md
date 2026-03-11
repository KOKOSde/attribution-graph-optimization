# Attribution Graph Optimization

Minimal attribution-graph generation code with three execution paths:

- `baseline`: per-position Python loops
- `optimized`: pure PyTorch vectorization
- `extension`: the same graph-generation pipeline, but with a custom native op for top-k threshold compaction

The repository now contains a real custom C++/CUDA extension:

- C++ source: [csrc/compact_topk.cpp](csrc/compact_topk.cpp)
- CUDA source: [csrc/compact_topk_cuda.cu](csrc/compact_topk_cuda.cu)
- Python fallback: [attribution_graph_optimization/graph_generation.py](attribution_graph_optimization/graph_generation.py)

## What Is Fused

After `torch.topk`, the graph pipeline still has to:

1. apply an activation threshold
2. compact the surviving entries
3. return `(batch_idx, position_idx, feature_idx, value)` tuples for node construction

The native op `compact_topk_threshold(...)` replaces the pure-PyTorch sequence:

- `valid_mask = top_vals >= threshold`
- `torch.where(valid_mask)`
- indexed gathers back into `top_vals` and `top_idx`

That is the smallest high-impact bottleneck in this repo that can be moved into a real compiled component without rewriting the whole pipeline.

## CUDA Work

The CUDA path added in this repo is narrow on purpose: it accelerates the post-`topk` threshold-compaction step without changing the surrounding Python graph-generation API.

What was added:

- [csrc/compact_topk_cuda.cu](csrc/compact_topk_cuda.cu): CUDA kernel and launcher for thresholding and compacting `topk` outputs directly on device
- [csrc/compact_topk.cpp](csrc/compact_topk.cpp): PyTorch operator registration and CPU/CUDA dispatch
- [setup.py](setup.py): conditional `CUDAExtension` build when a CUDA toolkit is available
- [attribution_graph_optimization/native.py](attribution_graph_optimization/native.py): runtime loading, status reporting, and safe fallback behavior
- [attribution_graph_optimization/graph_generation.py](attribution_graph_optimization/graph_generation.py): `implementation="extension"` integration with automatic fallback to pure PyTorch

What the CUDA kernel does:

1. reads `top_vals` and `top_idx` produced by `torch.topk`
2. applies the activation threshold on GPU
3. compacts surviving entries into a dense tuple representation
4. returns tensors used to build `(batch_idx, position_idx, feature_idx, value)` graph nodes

This keeps the custom native surface area small, makes the fallback path easy to verify, and leaves the rest of the graph pipeline readable in Python.

## Installation

Create a virtual environment, then use a single editable-install command:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Notes:

- If a CUDA toolkit is available during build, `setup.py` uses `CUDAExtension` and compiles the CUDA kernel.
- If CUDA is not available, the repo still works: it will build the CPU native extension when possible.
- If native compilation fails entirely, import and graph generation still work through the pure PyTorch fallback.

Optional build flags:

```bash
ATTR_GRAPH_FORCE_CPU=1 pip install -e ".[dev]"
ATTR_GRAPH_FORCE_CUDA=1 pip install -e ".[dev]"
```

## Quick Check

Inspect which path is active:

```python
from attribution_graph_optimization import get_native_extension_status
print(get_native_extension_status())
```

Example output from this workspace:

```python
{'loaded': True, 'enabled': True, 'disabled_by_env': False, 'build_variant': 'cpu', 'has_cuda_kernels': False, 'load_error': None}
```

Disable the native path explicitly:

```bash
ATTR_GRAPH_DISABLE_EXTENSION=1 python -m pytest -q
```

## Usage

```python
import torch
from attribution_graph_optimization import generate_attribution_graph

hidden_states = {
    40: torch.randn(1, 32, 256),
    41: torch.randn(1, 32, 256),
}
transcoders = {
    40: {'W_enc': torch.randn(1024, 256), 'b_enc': torch.zeros(1024)},
    41: {'W_enc': torch.randn(1024, 256), 'b_enc': torch.zeros(1024)},
}

graph = generate_attribution_graph(
    hidden_states=hidden_states,
    transcoders=transcoders,
    top_k=16,
    threshold=0.05,
    implementation='extension',
)
```

Valid `implementation` values:

- `baseline`
- `optimized`
- `extension`
- `auto` (`extension` when available, otherwise fallback)

## Benchmark

Run the reproducible benchmark:

```bash
python -m benchmarks.bench_graph_generation
```

Default settings are:

```bash
python -m benchmarks.bench_graph_generation \
  --device cpu \
  --batch-size 1 \
  --num-layers 4 \
  --seq-len 128 \
  --hidden-dim 256 \
  --feature-dim 2048 \
  --top-k 32 \
  --threshold 0.05 \
  --warmup 5 \
  --iterations 15 \
  --num-threads 1
```

The benchmark:

- runs `baseline`, `optimized`, and `extension`
- prints mean latency, p50 latency, and throughput in graphs/min
- checks node count and numerical equivalence within tolerance
- writes JSON output to [results/benchmark_results.json](results/benchmark_results.json)

### Example Output From This Run

This exact output was produced from a fresh Python 3.9 editable install in this workspace on CPU, with the native extension built in `cpu` mode because `nvcc` was not available:

```text
Configuration:
  device: cpu
  dtype: torch.float32
  batch_size: 1
  num_layers: 4
  seq_len: 128
  hidden_dim: 256
  feature_dim: 2048
  top_k: 32
  threshold: 0.05
  warmup: 5
  iterations: 15
  num_threads: 1
  native_extension: {'loaded': True, 'enabled': True, 'disabled_by_env': False, 'build_variant': 'cpu', 'has_cuda_kernels': False, 'load_error': None}

Latency summary (ms):
  baseline   mean=53.745  p50=53.903  throughput=1116.38 graphs/min
  optimized  mean=49.074  p50=48.835  throughput=1222.65 graphs/min
  extension  mean=48.456  p50=48.224  throughput=1238.23 graphs/min

Correctness:
  optimized_vs_baseline: {'matches': True, 'reason': 'ok', 'num_nodes': 16384}
  extension_vs_optimized: {'matches': True, 'reason': 'ok', 'num_nodes': 16384}
  summary_match: {'baseline_num_nodes': 16384, 'optimized_num_nodes': 16384, 'extension_num_nodes': 16384}
```

### Benchmark Table

Measured CPU results from this workspace are included below. CUDA/A100 values are placeholders until the GPU benchmark path is validated end-to-end.

| Environment | Variant | Mean latency (ms) | p50 latency (ms) | Throughput (graphs/min) | Status |
| --- | --- | ---: | ---: | ---: | --- |
| CPU local | `baseline` | 53.745 | 53.903 | 1116.38 | measured |
| CPU local | `optimized` | 49.074 | 48.835 | 1222.65 | measured |
| CPU local | `extension` | 48.456 | 48.224 | 1238.23 | measured |
| ASU Sol A100 | `optimized` | TBD | TBD | TBD | placeholder |
| ASU Sol A100 | `extension` | TBD | TBD | TBD | placeholder |

Current note for the A100 placeholders: the latest ASU Sol smoke run reached dependency install, local model preload, and vLLM startup, but failed on a CLI flag mismatch between the notebook runner and the pinned `vllm==0.11.0` server.

## Tests

CPU smoke test:

```bash
python -m pytest -q
```

The smoke test validates:

- package import
- explicit fallback via `ATTR_GRAPH_DISABLE_EXTENSION=1`
- graph-node shape/structure on CPU

## Project Layout

```text
attribution_graph_optimization/
  __init__.py
  graph_generation.py
  native.py
benchmarks/
  bench_graph_generation.py
csrc/
  compact_topk.cpp
  compact_topk_cuda.cu
tests/
  test_smoke.py
```

## Troubleshooting

### `nvcc` not found

The CUDA source is present, but the build will fall back to CPU native code unless a CUDA toolkit is installed.

### Native build fails

The install is intentionally non-fatal for the extension build. The Python fallback remains usable.

### CPU-only machines

CPU-only is supported. The benchmark and tests both run on CPU.

## What Is Now Verifiably True

This repository now truthfully contains:

- Python graph-generation code
- a custom C++ extension
- a CUDA implementation for the same native op
- a documented fallback path when native compilation is unavailable
- a benchmark script that reports actual measured results
- a CPU smoke test for CI

That makes the precise resume claim "custom C++/CUDA extension" accurate for this repo.
