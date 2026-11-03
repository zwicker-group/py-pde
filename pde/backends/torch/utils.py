"""Defines utilities for the torch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import DTypeLike

from .typing import NUMPY_TO_TORCH_DTYPE


class TorchOperatorBase(torch.nn.Module):
    """Base class for operators implemented in torch."""

    def __init__(self, *, dtype: DTypeLike):
        """Initialize the torch operator.

        Args:
            dtype:
                The data type of the field using the numpy convention
        """
        super().__init__()
        self.dtype = np.dtype(dtype)

    def register_array(self, name: str, arr: np.ndarray | torch.Tensor) -> None:
        """Register an array as a buffer in the torch module.

        Args:
            name (str):
                The name under which the buffer is registered
            arr (:class:`numpy.ndarray` or :class:`torch.Tensor`):
                The array to register. If a numpy array is provided, it will be
                converted to a torch tensor with the appropriate dtype.
        """
        if isinstance(arr, np.ndarray):
            tensor = torch.from_numpy(np.asarray(arr, dtype=self.dtype))
        elif isinstance(arr, torch.Tensor):
            tensor = arr
        else:
            msg = f"Cannot convert {arr}"
            raise TypeError(msg)

        self.register_buffer(name, tensor)


class TorchGaussianNoise(TorchOperatorBase):
    """Operator that returns uncorrelated Gaussian random field."""

    def __init__(self, data_shape, *, dtype, generator: torch.Generator | None = None):
        """
        Args:
            data_shape (tuple of ints):
                Shape of the output array
            dtype:
                Torch dtype of the returned data
            generator (:class:`torch.Generator` or None):
                Torch random number generator, which also allows setting the device on
                which the data is stored.
        """
        super().__init__(dtype=dtype)
        self.data_shape = data_shape
        self.generator = generator
        self.torch_dtype = NUMPY_TO_TORCH_DTYPE[self.dtype]

    def forward(self):
        return torch.randn(
            self.data_shape,
            dtype=self.torch_dtype,
            device=self.generator.device,
            generator=self.generator,
        )


def torch_heaviside(x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
    """Return the Heaviside step function using torch.

    This does not use :func:`torch.heaviside` since this is not implemented for the MPS
    device.

    Args:
        x1 (:class:`torch.Tensor`):
            Input values at which the Heaviside function is evaluated.
        x2 (:class:`torch.Tensor`, optional):
            Value used where `x1 == 0`. If omitted, `0.5` is used.

    Returns:
        :class:`torch.Tensor`:
            Tensor containing the Heaviside values of `x1`.
    """
    x1 = torch.as_tensor(x1)
    if x2 is None:
        x2 = 0.5  # type: ignore
    x2_t = torch.as_tensor(x2, dtype=x1.dtype, device=x1.device)
    return torch.where(
        x1 > 0,
        torch.ones_like(x1),
        torch.where(x1 < 0, torch.zeros_like(x1), x2_t),
    )


def torch_hypot(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean norm ``sqrt(x1**2 + x2**2)`` using torch.

    This wraps :func:`torch.hypot` and ensures that both inputs are converted to
    tensors before evaluation.

    Args:
        x1 (:class:`torch.Tensor`):
            First input values.
        x2 (:class:`torch.Tensor`):
            Second input values.

    Returns:
        :class:`torch.Tensor`:
            Tensor containing the element-wise hypotenuse of `x1` and `x2`.
    """
    return torch.hypot(torch.as_tensor(x1), torch.as_tensor(x2))


SPECIAL_FUNCTIONS_TORCH: dict[str, Callable] = {
    "Heaviside": torch_heaviside,
    "hypot": torch_hypot,
}
