from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Tuple

import torch

from .native import compact_topk_threshold as compact_topk_threshold_native
from .native import get_native_extension_status as _get_native_extension_status
from .native import is_native_extension_available

Node = Dict[str, object]
Implementation = Literal['baseline', 'optimized', 'extension', 'auto']


def _validate_feat_acts(feat_acts: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, int]:
    if feat_acts.dim() != 3:
        raise ValueError('feat_acts must have shape [batch, seq_len, feature_dim].')
    if top_k <= 0:
        raise ValueError('top_k must be positive.')
    return feat_acts.contiguous(), min(int(top_k), int(feat_acts.shape[-1]))



def _topk_threshold_torch(top_vals: torch.Tensor, top_idx: torch.Tensor, threshold: float):
    valid_mask = top_vals >= threshold
    batch_idx, pos_idx, k_idx = torch.where(valid_mask)
    values = top_vals[batch_idx, pos_idx, k_idx]
    features = top_idx[batch_idx, pos_idx, k_idx]
    return batch_idx, pos_idx, features, values



def _build_nodes(
    batch_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    feature_idx: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int,
    include_batch: bool,
) -> List[Node]:
    batch_list = batch_idx.tolist()
    pos_list = pos_idx.tolist()
    feature_list = feature_idx.tolist()
    value_list = values.tolist()

    nodes: List[Node] = []
    for batch, pos, feature, value in zip(batch_list, pos_list, feature_list, value_list):
        node_id = (
            f'{layer_idx}_{batch}_{feature}_{pos}'
            if include_batch
            else f'{layer_idx}_{feature}_{pos}'
        )
        node = {
            'node_id': node_id,
            'feature': int(feature),
            'layer': str(layer_idx),
            'ctx_idx': int(pos),
            'feature_type': 'cross layer transcoder',
            'influence': float(value),
            'activation': float(value),
        }
        if include_batch:
            node['batch_idx'] = int(batch)
        nodes.append(node)
    return nodes



def extract_features_baseline(
    feat_acts: torch.Tensor,
    layer_idx: int,
    top_k: int = 50,
    threshold: float = 0.01,
) -> List[Node]:
    feat_acts, top_k = _validate_feat_acts(feat_acts, top_k)
    batch_size, seq_len, _ = feat_acts.shape
    include_batch = batch_size > 1

    nodes: List[Node] = []
    for batch in range(batch_size):
        for pos in range(seq_len):
            pos_feats = feat_acts[batch, pos, :]
            top_vals, top_idx = torch.topk(pos_feats, k=top_k)
            for value, feature in zip(top_vals.tolist(), top_idx.tolist()):
                if value < threshold:
                    continue
                node_id = (
                    f'{layer_idx}_{batch}_{feature}_{pos}'
                    if include_batch
                    else f'{layer_idx}_{feature}_{pos}'
                )
                node = {
                    'node_id': node_id,
                    'feature': int(feature),
                    'layer': str(layer_idx),
                    'ctx_idx': int(pos),
                    'feature_type': 'cross layer transcoder',
                    'influence': float(value),
                    'activation': float(value),
                }
                if include_batch:
                    node['batch_idx'] = int(batch)
                nodes.append(node)
    return nodes



def extract_features_optimized(
    feat_acts: torch.Tensor,
    layer_idx: int,
    top_k: int = 50,
    threshold: float = 0.01,
) -> List[Node]:
    feat_acts, top_k = _validate_feat_acts(feat_acts, top_k)
    top_vals, top_idx = torch.topk(feat_acts, k=top_k, dim=-1)
    batch_idx, pos_idx, feature_idx, values = _topk_threshold_torch(top_vals, top_idx, threshold)
    return _build_nodes(
        batch_idx=batch_idx,
        pos_idx=pos_idx,
        feature_idx=feature_idx,
        values=values,
        layer_idx=layer_idx,
        include_batch=feat_acts.shape[0] > 1,
    )



def extract_features_extension(
    feat_acts: torch.Tensor,
    layer_idx: int,
    top_k: int = 50,
    threshold: float = 0.01,
) -> List[Node]:
    feat_acts, top_k = _validate_feat_acts(feat_acts, top_k)
    top_vals, top_idx = torch.topk(feat_acts, k=top_k, dim=-1)

    if is_native_extension_available():
        batch_idx, pos_idx, feature_idx, values = compact_topk_threshold_native(
            top_vals.contiguous(),
            top_idx.contiguous(),
            float(threshold),
        )
    else:
        batch_idx, pos_idx, feature_idx, values = _topk_threshold_torch(top_vals, top_idx, threshold)

    return _build_nodes(
        batch_idx=batch_idx,
        pos_idx=pos_idx,
        feature_idx=feature_idx,
        values=values,
        layer_idx=layer_idx,
        include_batch=feat_acts.shape[0] > 1,
    )



def extract_features(
    feat_acts: torch.Tensor,
    layer_idx: int,
    top_k: int = 50,
    threshold: float = 0.01,
    implementation: Implementation = 'auto',
) -> List[Node]:
    if implementation == 'baseline':
        return extract_features_baseline(feat_acts, layer_idx, top_k=top_k, threshold=threshold)
    if implementation == 'optimized':
        return extract_features_optimized(feat_acts, layer_idx, top_k=top_k, threshold=threshold)
    if implementation == 'extension':
        return extract_features_extension(feat_acts, layer_idx, top_k=top_k, threshold=threshold)
    if implementation == 'auto':
        return extract_features_extension(feat_acts, layer_idx, top_k=top_k, threshold=threshold)
    raise ValueError('implementation must be one of: baseline, optimized, extension, auto.')



def _encode_feature_activations(hidden: torch.Tensor, transcoder: Dict[str, torch.Tensor]) -> torch.Tensor:
    if hidden.dim() != 3:
        raise ValueError('hidden must have shape [batch, seq_len, hidden_dim].')

    weight = transcoder['W_enc']
    bias = transcoder['b_enc']
    batch_size, seq_len, hidden_dim = hidden.shape

    if weight.shape[1] != hidden_dim:
        if weight.shape[0] == hidden_dim:
            weight = weight.t()
        else:
            raise ValueError('W_enc shape does not match hidden_dim.')

    hidden_flat = hidden.reshape(batch_size * seq_len, hidden_dim)
    feat_acts = torch.relu(hidden_flat.matmul(weight.t()) + bias)
    return feat_acts.reshape(batch_size, seq_len, -1)



def generate_attribution_graph(
    hidden_states: Dict[int, torch.Tensor],
    transcoders: Dict[int, Dict[str, torch.Tensor]],
    top_k: int = 50,
    threshold: float = 0.01,
    implementation: Implementation = 'auto',
) -> Dict[str, object]:
    nodes: List[Node] = []
    node_influences = defaultdict(float)

    for layer_idx, hidden in hidden_states.items():
        if layer_idx not in transcoders:
            continue
        feat_acts = _encode_feature_activations(hidden, transcoders[layer_idx])
        layer_nodes = extract_features(
            feat_acts=feat_acts,
            layer_idx=layer_idx,
            top_k=top_k,
            threshold=threshold,
            implementation=implementation,
        )
        nodes.extend(layer_nodes)
        for node in layer_nodes:
            key = (layer_idx, int(node['feature']))
            node_influences[key] += float(node['influence'])

    return {
        'nodes': nodes,
        'node_influences': dict(node_influences),
        'summary': {
            'num_nodes': len(nodes),
            'num_layers': len(hidden_states),
            'implementation': implementation if implementation != 'auto' else 'extension_or_fallback',
        },
    }



def generate_attribution_graph_optimized(
    hidden_states: Dict[int, torch.Tensor],
    transcoders: Dict[int, Dict[str, torch.Tensor]],
    top_k: int = 50,
    threshold: float = 0.01,
) -> Dict[str, object]:
    return generate_attribution_graph(
        hidden_states=hidden_states,
        transcoders=transcoders,
        top_k=top_k,
        threshold=threshold,
        implementation='auto',
    )



def canonicalize_nodes(nodes: Iterable[Node]) -> List[Tuple[int, int, int, float]]:
    canonical = []
    for node in nodes:
        canonical.append(
            (
                int(node.get('batch_idx', 0)),
                int(node['ctx_idx']),
                int(node['feature']),
                float(node['activation']),
            )
        )
    return canonical



def get_native_extension_status():
    return _get_native_extension_status()
