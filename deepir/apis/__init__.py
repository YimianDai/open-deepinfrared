from .train import init_random_seed
from .test import single_gpu_test, multi_gpu_test
__all__ = [
    'init_random_seed', 'single_gpu_test', 'multi_gpu_test'
]