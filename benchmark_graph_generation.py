"""
Find the REAL bottleneck in attribution graph generation
Measure baseline vs optimized feature extraction
"""

import torch
import time
import json


def baseline_feature_extraction(feat_acts, layer_idx, top_k=50, threshold=0.01):
    """
    BASELINE: Current implementation (with Python loop)
    Lines 216-233 in generate_32b_attribution_fixed.py
    """
    B, T, F = feat_acts.shape
    nodes = []
    
    # SLOW: Python loop + .tolist()
    for pos in range(T):
        pos_feats = feat_acts[0, pos, :]
        top_vals, top_idx = torch.topk(pos_feats, k=min(top_k, pos_feats.shape[0]))
        
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val > threshold:
                nodes.append({
                    'node_id': f"{layer_idx}_{idx}_{pos}",
                    'feature': idx,
                    'layer': str(layer_idx),
                    'ctx_idx': int(pos),
                    'feature_type': 'cross layer transcoder',
                    'influence': float(val),
                    'activation': float(val)
                })
    
    return nodes


def optimized_feature_extraction(feat_acts, layer_idx, top_k=50, threshold=0.01):
    """
    OPTIMIZED: Vectorized GPU operations, no Python loops
    """
    B, T, F = feat_acts.shape
    
    # Apply threshold mask
    mask = feat_acts >= threshold
    
    # Get top-k per position (vectorized)
    top_vals, top_idx = torch.topk(feat_acts, k=min(top_k, F), dim=2)  # [B, T, top_k]
    
    # Filter by threshold
    valid_mask = top_vals >= threshold
    
    # Flatten and extract valid entries
    batch_idx, pos_idx, k_idx = torch.where(valid_mask)
    
    valid_vals = top_vals[batch_idx, pos_idx, k_idx]
    valid_features = top_idx[batch_idx, pos_idx, k_idx]
    valid_positions = pos_idx
    
    # Convert to list once (not in loop)
    vals_list = valid_vals.tolist()
    feats_list = valid_features.tolist()
    pos_list = valid_positions.tolist()
    
    # Build nodes
    nodes = []
    for val, feat, pos in zip(vals_list, feats_list, pos_list):
        nodes.append({
            'node_id': f"{layer_idx}_{feat}_{pos}",
            'feature': feat,
            'layer': str(layer_idx),
            'ctx_idx': int(pos),
            'feature_type': 'cross layer transcoder',
            'influence': float(val),
            'activation': float(val)
        })
    
    return nodes


def benchmark_extraction(seq_len=256, feature_dim=12288, num_layers=23, top_k=50):
    """
    Benchmark baseline vs optimized across multiple layers
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Seq len: {seq_len}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Top-k: {top_k}")
    
    # Create realistic feature activations (sparse)
    feat_acts = torch.relu(torch.randn(1, seq_len, feature_dim, device=device))
    
    # Make it sparse (realistic for ReLU features)
    feat_acts = feat_acts * (feat_acts > 1.0)
    
    print(f"  Sparsity: {(feat_acts > 0).float().mean().item():.3f}")
    
    # Warmup
    for _ in range(5):
        _ = baseline_feature_extraction(feat_acts, 40)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark BASELINE
    start = time.time()
    baseline_nodes = []
    for layer_idx in range(40, 40 + num_layers):
        nodes = baseline_feature_extraction(feat_acts, layer_idx)
        baseline_nodes.extend(nodes)
    if device == 'cuda':
        torch.cuda.synchronize()
    baseline_time = (time.time() - start) * 1000
    
    print(f"\n[1/2] BASELINE")
    print(f"  Time: {baseline_time:.2f} ms")
    print(f"  Nodes generated: {len(baseline_nodes)}")
    
    # Warmup optimized
    for _ in range(5):
        _ = optimized_feature_extraction(feat_acts, 40)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark OPTIMIZED
    start = time.time()
    optimized_nodes = []
    for layer_idx in range(40, 40 + num_layers):
        nodes = optimized_feature_extraction(feat_acts, layer_idx)
        optimized_nodes.extend(nodes)
    if device == 'cuda':
        torch.cuda.synchronize()
    optimized_time = (time.time() - start) * 1000
    
    print(f"\n[2/2] OPTIMIZED")
    print(f"  Time: {optimized_time:.2f} ms")
    print(f"  Nodes generated: {len(optimized_nodes)}")
    
    # Calculate speedup
    speedup = baseline_time / optimized_time
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"  Baseline:   {baseline_time:>8.2f} ms")
    print(f"  Optimized:  {optimized_time:>8.2f} ms")
    print(f"  Speedup:    {speedup:>8.2f}x")
    print(f"  Improvement: {(baseline_time - optimized_time)/baseline_time*100:>6.1f}% faster")
    print(f"{'='*70}\n")
    
    return {
        'baseline_ms': baseline_time,
        'optimized_ms': optimized_time,
        'speedup': speedup,
        'baseline_nodes': len(baseline_nodes),
        'optimized_nodes': len(optimized_nodes)
    }


def benchmark_full_pipeline():
    """
    Benchmark the full attribution graph generation pipeline
    """
    configs = [
        (256, 12288, 23, "Qwen 32B - 100 graphs (typical)"),
        (512, 12288, 23, "Qwen 32B - long sequences"),
        (256, 8192, 27, "Qwen 7B - all layers"),
    ]
    
    all_results = {}
    
    for seq_len, feature_dim, num_layers, name in configs:
        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}\n")
        
        results = benchmark_extraction(seq_len, feature_dim, num_layers)
        all_results[name] = results
        
        # Extrapolate to 100 graphs
        per_graph_baseline = results['baseline_ms']
        per_graph_opt = results['optimized_ms']
        
        total_baseline = per_graph_baseline * 100
        total_opt = per_graph_opt * 100
        
        print(f"Extrapolated to 100 graphs:")
        print(f"  Baseline total:  {total_baseline/1000:.1f} seconds ({total_baseline/60000:.1f} minutes)")
        print(f"  Optimized total: {total_opt/1000:.1f} seconds ({total_opt/60000:.1f} minutes)")
        print(f"  Time saved:      {(total_baseline-total_opt)/1000:.1f} seconds")
    
    # Save results
    output_path = '/scratch/fkalghan/graph_generation_speedup.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<40s} {'Speedup':>12s} {'Time Saved (100 graphs)':>15s}")
    print(f"{'-'*70}")
    for name, results in all_results.items():
        time_saved = (results['baseline_ms'] - results['optimized_ms']) * 100 / 1000
        print(f"{name:<40s} {results['speedup']:>11.2f}x {time_saved:>14.1f}s")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    benchmark_full_pipeline()

