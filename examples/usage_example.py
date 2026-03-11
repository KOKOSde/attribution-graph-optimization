import torch

from attribution_graph_optimization import generate_attribution_graph_optimized, get_native_extension_status


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_states = {
        40: torch.randn(1, 32, 256, device=device),
        41: torch.randn(1, 32, 256, device=device),
    }
    transcoders = {
        40: {
            'W_enc': torch.randn(1024, 256, device=device),
            'b_enc': torch.zeros(1024, device=device),
        },
        41: {
            'W_enc': torch.randn(1024, 256, device=device),
            'b_enc': torch.zeros(1024, device=device),
        },
    }

    graph = generate_attribution_graph_optimized(hidden_states, transcoders, top_k=16, threshold=0.05)
    print('native extension status:', get_native_extension_status())
    print('graph summary:', graph['summary'])
