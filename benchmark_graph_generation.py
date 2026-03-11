from benchmarks.bench_graph_generation import benchmark


if __name__ == '__main__':
    benchmark(
        batch_size=1,
        num_layers=4,
        seq_len=128,
        hidden_dim=256,
        feature_dim=2048,
        top_k=32,
        threshold=0.05,
        iterations=15,
        warmup=5,
        device_name='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        dtype_name='float32',
        atol=1e-5,
        num_threads=1,
    )
