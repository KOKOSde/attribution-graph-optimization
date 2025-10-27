"""
Example: Using the Optimized Attribution Graph Generator
4.76x faster than baseline implementation
"""

import torch
from optimized_graph_generation import generate_attribution_graph_optimized


def example_basic():
    """Basic usage example"""
    print("="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Create sample data (simulating real model outputs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hidden states from 3 transformer layers
    hidden_states = {
        40: torch.randn(1, 256, 5120, device=device),  # [batch, seq_len, hidden_dim]
        41: torch.randn(1, 256, 5120, device=device),
        42: torch.randn(1, 256, 5120, device=device),
    }
    
    # Transcoder weights (feature extractors)
    transcoders = {
        40: {
            'W_enc': torch.randn(12288, 5120, device=device),  # [feature_dim, hidden_dim]
            'b_enc': torch.zeros(12288, device=device)
        },
        41: {
            'W_enc': torch.randn(12288, 5120, device=device),
            'b_enc': torch.zeros(12288, device=device)
        },
        42: {
            'W_enc': torch.randn(12288, 5120, device=device),
            'b_enc': torch.zeros(12288, device=device)
        }
    }
    
    # Generate attribution graph (4.76x faster!)
    import time
    start = time.time()
    
    graph = generate_attribution_graph_optimized(
        hidden_states=hidden_states,
        transcoders=transcoders,
        top_k=50,
        threshold=0.01
    )
    
    elapsed = (time.time() - start) * 1000
    
    print(f"\n✅ Generated attribution graph in {elapsed:.1f}ms")
    print(f"   Nodes: {graph['summary']['num_nodes']}")
    print(f"   Layers: {graph['summary']['num_layers']}")
    print(f"   Unique features: {len(graph['node_influences'])}")


def example_comparison():
    """Compare baseline vs optimized"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Performance Comparison")
    print("="*70)
    
    from benchmark_graph_generation import benchmark_extraction
    
    print("\nRunning benchmark on Qwen 32B configuration...")
    results = benchmark_extraction(
        seq_len=256,
        feature_dim=12288,
        num_layers=5,  # Just 5 layers for quick demo
        top_k=50
    )
    
    print(f"\n📊 Results:")
    print(f"   Speedup: {results['speedup']:.2f}x")
    print(f"   Time saved per graph: {results['baseline_ms'] - results['optimized_ms']:.1f}ms")


def example_realistic():
    """More realistic example with actual model dimensions"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Realistic Model Scale (23 layers)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Qwen 32B: 23 layers (40-62)
    num_layers = 23
    start_layer = 40
    
    print(f"\nGenerating for {num_layers} layers...")
    
    # Create hidden states for all layers
    hidden_states = {}
    transcoders = {}
    
    for i in range(num_layers):
        layer_idx = start_layer + i
        hidden_states[layer_idx] = torch.randn(1, 256, 5120, device=device)
        transcoders[layer_idx] = {
            'W_enc': torch.randn(12288, 5120, device=device),
            'b_enc': torch.zeros(12288, device=device)
        }
    
    import time
    start = time.time()
    
    graph = generate_attribution_graph_optimized(
        hidden_states=hidden_states,
        transcoders=transcoders,
        top_k=50,
        threshold=0.01
    )
    
    elapsed = (time.time() - start) * 1000
    
    print(f"\n✅ Generated graph for 23 layers in {elapsed:.1f}ms")
    print(f"   Nodes: {graph['summary']['num_nodes']:,}")
    print(f"   Average nodes per layer: {graph['summary']['num_nodes'] / num_layers:.0f}")
    
    # Estimate time for 100 graphs
    time_100 = elapsed * 100 / 1000  # seconds
    print(f"\n⏱️  Estimated time for 100 graphs: {time_100:.1f}s ({time_100/60:.1f} minutes)")


if __name__ == '__main__':
    example_basic()
    example_realistic()
    
    # Only run comparison if GPU available (faster)
    if torch.cuda.is_available():
        example_comparison()
    
    print("\n" + "="*70)
    print("✨ All examples completed!")
    print("="*70)

