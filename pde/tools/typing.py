"""
Provides support for mypy type checking of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

Real = Union[int, float]
Number = Union[Real, complex]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]
ArrayLike = Union[NumberOrArray, Sequence[NumberOrArray], Sequence[Sequence[Any]]]

OperatorType = Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
