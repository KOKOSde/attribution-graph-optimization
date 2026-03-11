import os

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension, CppExtension

def should_build_with_cuda() -> bool:
    if os.environ.get('ATTR_GRAPH_FORCE_CPU', '0') == '1':
        return False
    if os.environ.get('ATTR_GRAPH_FORCE_CUDA', '0') == '1':
        return True
    return CUDA_HOME is not None and bool(torch.cuda.is_available())


class OptionalBuildExtension(BuildExtension):
    def run(self):
        try:
            super().run()
        except Exception as exc:
            self._warn(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._warn(exc)

    @staticmethod
    def _warn(exc):
        banner = '=' * 79
        print(banner)
        print('WARNING: optional native extension build failed; pure PyTorch fallback remains available.')
        print(str(exc))
        print(banner)


extra_compile_args = {'cxx': ['-O3']}
sources = ['csrc/compact_topk.cpp']
define_macros = []
extension_cls = CppExtension

if should_build_with_cuda():
    extension_cls = CUDAExtension
    define_macros.append(('WITH_CUDA', None))
    sources.append('csrc/compact_topk_cuda.cu')
    extra_compile_args['nvcc'] = ['-O3']

ext_modules = [
    extension_cls(
        name='attribution_graph_optimization._native',
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name='attribution-graph-optimization',
    version='0.1.0',
    description='Attribution graph generation with an optional custom C++/CUDA extension',
    packages=find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    python_requires='>=3.9,<3.11',
    install_requires=['numpy>=1.24,<2.0', 'torch>=1.13,<2.0'],
    extras_require={'dev': ['pytest>=8.0']},
    ext_modules=ext_modules,
    cmdclass={'build_ext': OptionalBuildExtension},
)
