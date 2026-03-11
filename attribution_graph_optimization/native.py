import importlib
import os
from typing import Any, Dict

_LOAD_ERROR = None
_NATIVE_MODULE = None

try:
    _NATIVE_MODULE = importlib.import_module('attribution_graph_optimization._native')
except Exception as exc:
    _LOAD_ERROR = exc


def _extension_disabled() -> bool:
    return os.environ.get('ATTR_GRAPH_DISABLE_EXTENSION', '0') == '1'


def is_native_extension_available() -> bool:
    return (not _extension_disabled()) and (_NATIVE_MODULE is not None)



def compact_topk_threshold(top_vals, top_idx, threshold):
    if not is_native_extension_available():
        raise RuntimeError('Native extension is unavailable; use the pure PyTorch fallback.')
    return _NATIVE_MODULE.compact_topk_threshold(top_vals, top_idx, float(threshold))



def get_native_extension_status() -> Dict[str, Any]:
    has_module = _NATIVE_MODULE is not None
    enabled = is_native_extension_available()
    has_cuda = False
    build_variant = 'python-fallback'

    if has_module:
        has_cuda = bool(getattr(_NATIVE_MODULE, 'has_cuda', lambda: False)())
        build_variant = str(getattr(_NATIVE_MODULE, 'build_variant', lambda: 'cpu')())

    return {
        'loaded': has_module,
        'enabled': enabled,
        'disabled_by_env': _extension_disabled(),
        'build_variant': build_variant,
        'has_cuda_kernels': has_cuda,
        'load_error': None if _LOAD_ERROR is None else str(_LOAD_ERROR),
    }
