"""Defines types specific to the torch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Protocol

import numpy as np
import torch
from numpy.typing import DTypeLike

AnyDType = DTypeLike | torch.dtype

NUMPY_TO_TORCH_DTYPE: dict[DTypeLike, torch.dtype] = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.double: torch.double,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
# also define inverse mapping to proper numpy dtypes
TORCH_TO_NUMPY_DTYPE = {v: np.dtype(k) for k, v in NUMPY_TO_TORCH_DTYPE.items()}
# add the proper numpy dtype as an alternative
NUMPY_TO_TORCH_DTYPE |= {np.dtype(k): v for k, v in NUMPY_TO_TORCH_DTYPE.items()}


class TorchRHSType(Protocol):
    """General stepper type working with torch tensors."""

    def __call__(self, state_data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluates right hand side of the PDE in a torch backend.

        Args:
            state_data (:class:`~torch.Tensor`):
                The current state
            t (float):
                Current time point

        Returns:
            :class:`~torch.Tensor`: Evolution rate
        """


class TorchInnerStepperType(Protocol):
    """General stepper type working with torch tensors."""

    def __call__(
        self, state_data: torch.Tensor, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, float]:
        """General stepper that advances the state given as a numpy array.

        Args:
            state_data (:class:`~torch.Tensor`):
                The current state
            t_start (float):
                Initial time point
            t_end (float):
                Desired final time point

        Returns:
            tuple of the state and time at the final point
        """
