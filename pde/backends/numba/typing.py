"""Provides support for mypy type checking of the module.

.. autosummary::
   :nosignatures:

   NumbaOperatorImplType

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TypeAlias

from ...tools.typing import _OperatorImplUpdateType

NumbaOperatorImplType: TypeAlias = _OperatorImplUpdateType
