# ⚡ 4.76x Faster Attribution Graph Generation for 32B Vision-Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Fahad Alghanim  
**Performance:** 79% faster (1034ms → 217ms per graph)  
**Time Saved:** 13.4 minutes per 1000 graphs

---

## 🎯 The Problem

Attribution graph generation for large vision-language models (32B parameters) was bottlenecked by **Python loops** over sequence positions during feature extraction:

```python
# ❌ BASELINE: 1034 ms per graph
for pos in range(T):  # Python loop for each position
    pos_feats = feat_acts[0, pos, :]
    top_vals, top_idx = torch.topk(pos_feats, k=50)
    
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):  # CPU transfer per loop!
        if val > 0.01:
            nodes.append({...})
```

**Issues:**
- 🐌 Python loop over 256-512 positions
- 🔄 GPU→CPU transfer **per position** via `.tolist()`
- ❌ No GPU parallelization
- 💾 Multiple small memory transfers

---

## ✨ The Solution

**Vectorized GPU operations** eliminating Python loops:

```python
# ✅ OPTIMIZED: 217 ms per graph (4.76x faster!)
# Single vectorized top-k across ALL positions at once
top_vals, top_idx = torch.topk(feat_acts, k=50, dim=2)  # [B, T, 50]

# Vectorized threshold filtering
valid_mask = top_vals >= 0.01
batch_idx, pos_idx, k_idx = torch.where(valid_mask)

# Single GPU→CPU transfer (not per-position!)
vals_list = top_vals[batch_idx, pos_idx, k_idx].tolist()
feats_list = top_idx[batch_idx, pos_idx, k_idx].tolist()
pos_list = pos_idx.tolist()

# Build nodes
for val, feat, pos in zip(vals_list, feats_list, pos_list):
    nodes.append({...})
```

**Key Optimizations:**
1. ⚡ **Vectorized `torch.topk`** across sequence dimension
2. 🔥 **Single GPU→CPU transfer** instead of 256 transfers
3. 🎯 **`torch.where` mask filtering** entirely on GPU
4. 🚀 **Eliminated Python loop** over sequence positions

---

## 📊 Performance Results

### Configuration
- **Model:** Qwen 32B Vision-Language Model
- **Layers:** 23 transformer layers (L40-62)
- **Features:** 12,288 dimensions
- **Sequence:** 256 tokens
- **Hardware:** NVIDIA A100 GPU

### Benchmark Results

| Configuration | Baseline | Optimized | **Speedup** | Time Saved (100 graphs) |
|--------------|----------|-----------|-------------|-------------------------|
| Qwen 32B (256 seq) | 1034 ms | 217 ms | **4.76x** | 81.7s |
| Qwen 32B (512 seq) | 2104 ms | 435 ms | **4.83x** | 166.9s |
| Qwen 7B (all layers) | 1074 ms | 253 ms | **4.25x** | 82.1s |

### Real-World Impact

- **100 graphs:** 1.7 minutes → 0.4 minutes ⚡
- **1000 graphs:** 17 minutes → 3.6 minutes ⚡
- **Saves 13.4 minutes per 1000 graphs** 🎉

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YourUsername/attribution-graph-optimization.git
cd attribution-graph-optimization
pip install torch numpy
```

### Usage

```python
from optimized_graph_generation import generate_attribution_graph_optimized

# Generate graph (4.76x faster than baseline!)
graph = generate_attribution_graph_optimized(
    hidden_states={40: hidden_tensor, 41: hidden_tensor, ...},
    transcoders={40: transcoder_weights, 41: transcoder_weights, ...},
    top_k=50,
    threshold=0.01
)

print(f"Generated {graph['summary']['num_nodes']} nodes")
```

### Run Benchmarks

```bash
python benchmark_graph_generation.py
```

**Expected output:**
```
Qwen 32B - 100 graphs (typical)
  Baseline:    1034.76 ms
  Optimized:    217.51 ms
  Speedup:        4.76x
  Improvement:   79.0% faster
```

---

## 🔬 Technical Deep Dive

### Why Python Loops are Slow

```python
# SLOW: 256 separate GPU operations
for i in range(256):
    result = torch.topk(tensor[:, i, :], k=50)  # 256 kernel launches!
    data = result.tolist()  # 256 CPU transfers!
```

**Problems:**
- Each iteration launches a separate GPU kernel
- `.tolist()` forces GPU synchronization and memory copy
- Python interpreter overhead per iteration
- No opportunity for GPU parallelism

### The Vectorization Advantage

```python
# FAST: Single vectorized operation
result = torch.topk(tensor, k=50, dim=2)  # 1 kernel launch!
data = result[mask].tolist()  # 1 CPU transfer!
```

**Benefits:**
- Single GPU kernel processes all positions in parallel
- One memory transfer after all GPU work completes
- PyTorch can optimize the entire operation
- Leverages GPU's massive parallelism

### Performance Breakdown

| Operation | Baseline (ms) | Optimized (ms) | Improvement |
|-----------|--------------|----------------|-------------|
| top-k extraction | 450 | 120 | 3.75x |
| GPU→CPU transfer | 380 | 35 | 10.9x |
| Threshold filtering | 180 | 42 | 4.3x |
| Node creation | 24 | 20 | 1.2x |
| **Total** | **1034** | **217** | **4.76x** |

---

## 📁 Project Structure

```
attribution-graph-optimization/
├── README.md                           # This file
├── benchmark_graph_generation.py       # Reproducible benchmarks
├── optimized_graph_generation.py       # Optimized implementation
├── results/
│   └── benchmark_results.json          # Performance data
└── examples/
    └── usage_example.py                # Example usage
```

---

## 🎓 Key Learnings

### 1. GPU ↔ CPU Transfers are Expensive

The baseline called `.tolist()` **256 times per graph**, transferring small amounts of data repeatedly. The optimized version calls it **once**, transferring all data in bulk.

**Result:** 10.9x faster data transfer

### 2. Vectorization Beats Loops

PyTorch is highly optimized for batch operations. A single `torch.topk(tensor, dim=2)` is much faster than looping and calling `torch.topk` 256 times.

**Result:** 3.75x faster computation

### 3. Profile Before Optimizing

Initial hypothesis: Matrix multiplication was the bottleneck.  
Reality: Python loops and memory transfers were the bottleneck.

**Lesson:** Always measure before optimizing!

---

## 💡 Applications

This optimization is useful for:

- **Attribution Analysis** - Understanding which features influence model outputs
- **Model Interpretability** - Generating feature activation graphs at scale
- **AI Safety Research** - Analyzing model behavior patterns
- **Neural Network Debugging** - Identifying problematic features quickly

---

## 🛠️ Technical Skills Demonstrated

- ✅ **Performance Profiling** - Identified true bottleneck through systematic measurement
- ✅ **GPU Programming** - Understanding memory transfers and kernel launches
- ✅ **PyTorch Optimization** - Leveraging vectorized operations effectively
- ✅ **Benchmarking** - Reproducible, statistically sound measurements
- ✅ **Production ML** - Optimizing 32B parameter model pipelines

---

## 📈 Reproducing Results

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (tested on A100)

### Run Full Benchmark Suite

```bash
python benchmark_graph_generation.py
```

This will:
1. Test on Qwen 32B (256 & 512 sequence lengths)
2. Test on Qwen 7B (all layers)
3. Generate `results/benchmark_results.json`
4. Print summary table with speedups

---

## 🤝 Contributing

Found a way to make it even faster? Contributions welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/even-faster`)
3. Commit your changes (`git commit -m 'Add some optimization'`)
4. Push to the branch (`git push origin feature/even-faster`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

**Fahad Alghanim**  
Email: fkalghan@email.sc.edu  
GitHub: [@YourGitHubUsername]

---

## 🌟 Acknowledgments

This optimization work supports:
- **AI Safety Research** - Trap detection in vision-language models
- **Model Interpretability** - Understanding feature attribution at scale
- **Production ML Systems** - Efficient processing of large-scale models

---

## 📚 Citation

If you use this optimization in your research, please cite:

```bibtex
@software{alghanim2025attribution,
  author = {Alghanim, Fahad},
  title = {4.76x Faster Attribution Graph Generation for Vision-Language Models},
  year = {2025},
  url = {https://github.com/YourUsername/attribution-graph-optimization}
}
```

---

<div align="center">

**⚡ 4.76x Faster | 🎯 79% Improvement | 🚀 Production-Ready**

Made with ❤️ by Fahad Alghanim

</div>

