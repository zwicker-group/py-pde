"""Defines fields, which contain the actual data stored on a discrete grid.

.. autosummary::
   :nosignatures:

   ~scalar.ScalarField
   ~vectorial.VectorField
   ~tensorial.Tensor2Field
   ~collection.FieldCollection


Inheritance structure of the classes:


.. inheritance-diagram:: scalar.ScalarField vectorial.VectorField tensorial.Tensor2Field
        collection.FieldCollection
   :parts: 1

The details of the classes are explained below:

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .base import FieldBase
from .collection import FieldCollection
from .datafield_base import DataFieldBase
from .scalar import ScalarField
from .tensorial import Tensor2Field
from .vectorial import VectorField

# DataFieldBase has been moved to its own module on 2024-04-18.
# Add it back to `base` for the time being, so dependent code doesn't break
from . import base  # isort:skip

base.DataFieldBase = DataFieldBase  # type: ignore
del base  # clean namespaces
