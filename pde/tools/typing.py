"""
Provides support for mypy type checking of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Sequence, Union

import numpy as np

Real = Union[int, float]
Number = Union[Real, complex, np.generic]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]
ArrayLike = Union[NumberOrArray, Sequence[NumberOrArray], Sequence[Sequence[Any]]]
