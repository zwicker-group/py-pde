"""Establishes hooks for the interplay between :mod:`pde` and :mod:`modelrunner`

This package is usually loaded automatically during import if :mod:`modelrunner` is
available. In this case, grids and fields of :mod:`pde` can be directly written to
storages from :mod:`modelrunner.storage`.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from collections.abc import Sequence

from modelrunner.storage import StorageBase, storage_actions
from modelrunner.storage.utils import decode_class

from ..fields.base import FieldBase
from ..grids.base import GridBase


# these actions are inherited by all subclasses by default
def load_grid(storage: StorageBase, loc: Sequence[str]) -> GridBase:
    """Function loading a grid from a modelrunner storage.

    Args:
        storage (:class:`~modelrunner.storage.group.StorageGroup`):
            Storage to load data from
        loc (Location):
            Location in the storage

    Returns:
        :class:`~pde.grids.base.GridBase`: the loaded grid
    """
    # get grid class that was stored
    stored_cls = decode_class(storage._read_attrs(loc).get("__class__"))
    state = storage.read_attrs(loc)
    return stored_cls.from_state(state)  # type: ignore


storage_actions.register("read_item", GridBase, load_grid)


def save_grid(storage: StorageBase, loc: Sequence[str], grid: GridBase) -> None:
    """Function saving a grid to a modelrunner storage.

    Args:
        storage (:class:`~modelrunner.storage.group.StorageGroup`):
            Storage to save data to
        loc (Location):
            Location in the storage
        grid (:class:`~pde.grids.base.GridBase`):
            the grid to store
    """
    storage.write_object(loc, None, attrs=grid.state, cls=grid.__class__)


storage_actions.register("write_item", GridBase, save_grid)


# these actions are inherited by all subclasses by default
def load_field(storage: StorageBase, loc: Sequence[str]) -> FieldBase:
    """Function loading a field from a modelrunner storage.

    Args:
        storage (:class:`~modelrunner.storage.group.StorageGroup`):
            Storage to load data from
        loc (Location):
            Location in the storage

    Returns:
        :class:`~pde.fields.base.FieldBase`: the loaded field
    """
    # get field class that was stored
    stored_cls = decode_class(storage._read_attrs(loc).get("__class__"))
    attributes = stored_cls.unserialize_attributes(storage.read_attrs(loc))  # type: ignore
    return stored_cls.from_state(attributes, data=storage.read_array(loc))  # type: ignore


storage_actions.register("read_item", FieldBase, load_field)


def save_field(storage: StorageBase, loc: Sequence[str], field: FieldBase) -> None:
    """Function saving a field to a modelrunner storage.

    Args:
        storage (:class:`~modelrunner.storage.group.StorageGroup`):
            Storage to save data to
        loc (Location):
            Location in the storage
        field (:class:`~pde.fields.base.FieldBase`):
            the field to store
    """
    storage.write_array(
        loc, field.data, attrs=field.attributes_serialized, cls=field.__class__
    )


storage_actions.register("write_item", FieldBase, save_field)


__all__: list[str] = []  # module only registers hooks and does not export any functions
