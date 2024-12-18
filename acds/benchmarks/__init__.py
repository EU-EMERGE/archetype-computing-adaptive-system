from .adiac.getter import get_adiac_data
from .mackey_glass import get_mackey_glass, get_mackey_glass_windows
from .mnist import get_mnist_data
from .mallat import get_mallat_data
from .trace import get_trace_data
from .libras import get_libras_data
from .cifar10 import get_cifar10_data

__all__ = ["get_adiac_data", "get_mackey_glass", "get_mnist_data", "get_mackey_glass_windows", "get_mallat_data", "get_trace_data", "get_libras_data", "get_cifar10_data"]
