"""
Defines fields, which contain the actual data stored on a discrete grid.

.. autosummary::
   :nosignatures:

   ~scalar.ScalarField
   ~vectorial.VectorField
   ~tensorial.Tensor2Field
   ~collection.FieldCollection


Inheritance structure of the classes:


.. inheritance-diagram:: pde.fields.base pde.fields.scalar pde.fields.vectorial
        pde.fields.tensorial pde.fields.collection
   :parts: 1

The details of the classes are explained below:

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .collection import FieldCollection
from .scalar import ScalarField
from .tensorial import Tensor2Field
from .vectorial import VectorField
