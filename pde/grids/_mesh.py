"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from ..fields import FieldCollection
from ..fields.base import DataFieldBase, FieldBase
from ..tools import mpi
from ..tools.cache import cached_method
from ..tools.plotting import plot_on_axes
from .base import GridBase
from .boundaries.axes import Boundaries
from .boundaries.axis import BoundaryPair
from .boundaries.local import _MPIBC


class MPIFlags(IntEnum):
    """enum that contains flags for MPI communication"""

    field_split = 1  # split full field onto nodes
    field_combine = 2  # combine full field from subfields on nodes
    _boundary_lower = 8  # exchange with lower boundary of node with lower id
    _boundary_upper = 9  # exchange with upper boundary of node with lower id

    @classmethod
    def boundary_lower(cls, my_id: int, other_id: int) -> int:
        """flag for connection between my lower boundary and `other_id`"""
        if my_id <= other_id:
            return 2 * my_id + cls._boundary_lower
        else:
            return 2 * other_id + cls._boundary_upper

    @classmethod
    def boundary_upper(cls, my_id: int, other_id: int) -> int:
        """flag for connection between my upper boundary and `other_id`"""
        if my_id <= other_id:
            return 2 * my_id + cls._boundary_upper
        else:
            return 2 * other_id + cls._boundary_lower


def _subdivide(num: int, chunks: int) -> np.ndarray:
    r"""subdivide `num` intervals in `chunk` chunks

    Args:
        num (int): Number of intervals
        chunks (int): Number of chunks

    Returns:
        list: The number of intervals per chunk
    """
    if chunks > num:
        raise RuntimeError("Cannot divide in more chunks than support points")
    return np.diff(np.linspace(0, num, chunks + 1).astype(int))


def _subdivide_along_axis(grid: GridBase, axis: int, chunks: int) -> List[GridBase]:
    """subdivide the grid along a given axis

    Args:
        axis (int): The axis along which the subdivision will happen
        chunks (int): The number of chunks along this axis

    Returns:
        list: A list of subgrids
    """
    if chunks <= 0:
        raise ValueError("Chunks must be a positive Integer")
    elif chunks == 1:
        return [grid]  # no subdivision necessary

    def replace_in_axis(arr, value):
        if isinstance(arr, tuple):
            return arr[:axis] + (value,) + arr[axis + 1 :]
        else:
            res = arr.copy()
            res[axis] = value
            return res

    subgrids = []
    start = 0
    for size in _subdivide(grid.shape[axis], chunks):
        # determine new sub region
        end = start + size
        shape = replace_in_axis(grid.shape, size)
        periodic = replace_in_axis(grid.periodic, False)

        # determine bounds of the new grid
        axis_bounds = grid.axes_bounds[axis]
        cell_bounds = np.linspace(*axis_bounds, grid.shape[axis] + 1)
        bounds = replace_in_axis(
            grid.axes_bounds, (cell_bounds[start], cell_bounds[end])
        )

        # create new subgrid
        subgrid = grid.__class__.from_bounds(bounds, shape, periodic)
        subgrids.append(subgrid)
        start = end  # for next iteration

    return subgrids


TField = TypeVar("TField", bound=FieldBase)
TData = TypeVar("TData")


class GridMesh:
    """handles a collection of subgrids arranged in a regular mesh

    This class provides methods for managing MPI simulations of multiple connected
    subgrids. Each subgrid is also called a cell and identified with a unique number.
    """

    def __init__(self, basegrid: GridBase, subgrids: Sequence):
        """
        Args:
            basegrid (:class:`~pde.grids.base.GridBase`):
                The grid of the entire domain
            subgrids (nested lists of :class:`~pde.grids.base.GridBase`):
                The nested grids representing the subdivision
        """
        self.basegrid = basegrid
        self.subgrids = np.asarray(subgrids)
        for subgrid in self.subgrids.flat:  # , flags=["refs_ok"]):
            subgrid._mesh = self

        assert basegrid.num_axes == self.subgrids.ndim

    @classmethod
    def from_grid(
        cls, grid: GridBase, decomposition: Union[int, List[int]] = -1
    ) -> GridMesh:
        """subdivide the grid into subgrids

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid that will be subdivided
            decomposition (list of ints):
                Number of subdivision in each direction. Should be a list of length
                `grid.num_axes` specifying the number of nodes for this axis. If one
                value is `-1`, its value will be determined from the number of available
                nodes. The default value decomposed the first axis using all available
                nodes.

        Returns:
            :class:`GridMesh`
        """
        # parse `decomposition`
        try:
            decomposition = [int(d) for d in decomposition]  # type: ignore
        except TypeError:
            decomposition = [int(decomposition)]  # type: ignore

        size, var_index = 1, None
        for i, num in enumerate(decomposition):
            if num == -1:
                if var_index is None:
                    var_index = i
                else:
                    raise ValueError("can only specify one unknown dimension")
            elif num > 0:
                size *= num
            else:
                raise RuntimeError(f"Unknown size `{num}`")

        # replace potential variable index with correct value
        if var_index is not None:
            dim = mpi.size // size
            if dim > 0:
                decomposition[var_index] = dim
            else:
                raise RuntimeError("Not enough nodes to satisfy decomposition")

        # fill up with 1s until the grid size is met
        decomposition += [1] * (grid.num_axes - len(decomposition))

        # check compatibility with number of nodes
        if mpi.size > 1 and math.prod(decomposition) != mpi.size:
            raise RuntimeError(
                f"Node count ({mpi.size}) incompatible with decomposition "
                f"({decomposition})"
            )

        # subdivide the base grid according to the decomposition
        subgrids = np.empty(decomposition, dtype=object)
        subgrids.flat[0] = grid  # seed the initial grid at the top-left
        idx_set: List[Any] = [0] * subgrids.ndim  # indices to extract all grids
        for axis, chunks in enumerate(decomposition):
            # iterate over all grids that have been determined already
            for idx, subgrid in np.ndenumerate(subgrids[tuple(idx_set)]):
                i = idx + (slice(None, None),) + (0,) * (subgrids.ndim - axis - 1)  # type: ignore
                divison = _subdivide_along_axis(subgrid, axis=axis, chunks=chunks)
                subgrids[i] = divison

            # mark this axis as set
            idx_set[axis] = slice(None, None)

        return cls(basegrid=grid, subgrids=subgrids)  # type: ignore

    @property
    def num_axes(self) -> int:
        """int: the number of axes that the grids possess"""
        return self.subgrids.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """tuple: the number of subgrids along each axis"""
        return self.subgrids.shape

    def __len__(self) -> int:
        """total number of subgrids"""
        return self.subgrids.size

    @property
    def current_node(self) -> int:
        """int: current MPI node"""
        return mpi.rank

    def __getitem__(self, node_id: Optional[int]) -> GridBase:
        """extract one subgrid from the mesh

        Args:
            node_id (int):
                Index identifying the subgrid

        Returns:
            :class:`~pde.grids.base.GridBase`: The sub grid of the given cell
        """
        if node_id is None:
            node_id = self.current_node
        return self.subgrids[self._id2idx(node_id)]  # type: ignore

    @property
    def current_grid(self) -> GridBase:
        """:class:`~pde.grids.base.GridBase`:subgrid of current MPI node"""
        return self[self.current_node]

    def _id2idx(self, node_id: int) -> Tuple[int, ...]:
        """convert linear id into node index

        Args:
            node_id (int):
                Linear index identifying the node

        Returns:
            tuple: Full index with `num_axes` entries.
        """
        return np.unravel_index(node_id, self.shape)  # type: ignore

    def _idx2id(self, node_idx: Sequence[int]) -> int:
        """convert node index to linear index

        Args:
            node_idx (tuple):
                Full index with `num_axes` entries.

        Returns:
            int: Linear index identifying the node
        """
        return np.ravel_multi_index(node_idx, self.shape)  # type: ignore

    @cached_method()
    def _get_data_indices_1d(self, with_ghost_cells: bool = False) -> List[List[slice]]:
        """indices to extract valid field data for each subgrid

        Args:
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in `field_data`

        Returns:
            A list of slices for each axis
        """
        # create indices for each axis
        i_add = 2 if with_ghost_cells else 0
        indices_1d = []
        for axis in range(self.num_axes):
            grid_ids: List[Any] = [0] * self.num_axes
            grid_ids[axis] = slice(None, None)
            data, last = [], 0
            for grid in self.subgrids[tuple(grid_ids)]:
                n = grid.shape[axis]
                data.append(slice(last, last + n + i_add))
                last += n
            indices_1d.append(data)
        return indices_1d

    @cached_method()
    def _get_data_indices(self, with_ghost_cells: bool = False) -> np.ndarray:
        """indices to extract valid field data for each subgrid

        Args:
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in `field_data`

        Returns:
            :class:`~numpy.ndarray` an array of indices to access data of the sub grids
        """
        indices_1d = self._get_data_indices_1d(with_ghost_cells)

        # combine everything into a full indices
        indices = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            indices[idx] = tuple(indices_1d[n][i] for n, i in enumerate(idx))

        return indices

    def get_boundary_flag(self, neighbor: int, upper: bool) -> int:
        """get MPI flag indicating the boundary between this node and its neighbor

        Args:
            node_id (int):
                The ID of the neighboring node
            upper (bool):
                Flag indicating the boundary of the current node

        Returns:
            int: the unique flag associated with this boundary
        """
        if upper:
            # my upper boundary (lower boundary of right cell)
            return MPIFlags.boundary_upper(self.current_node, neighbor)
        else:
            # my lower boundary (upper boundary of left cell)
            return MPIFlags.boundary_lower(self.current_node, neighbor)

    def get_neighbor(
        self, axis: int, upper: bool, *, node_id: int = None
    ) -> Optional[int]:
        """get node id of the neighbor along the given axis and direction

        Args:
            axis (int):
                The axis of the grid
            upper (bool):
                Flag deciding which boundary to consider
            node_id (int):
                The ID of the considered node. Derived from MPI.rank() if omitted.

        Returns:
            int: The id of the neighboring node
        """
        size = self.shape[axis]
        if size == 1:
            return None  # there are no other nodes along this axis

        if node_id is None:
            node_id = self.current_node

        idx = list(self._id2idx(node_id))

        if upper:
            # my upper boundary
            if idx[axis] < size - 1:
                idx[axis] = idx[axis] + 1  # proper cell neighbor on upper side
            elif self.basegrid.periodic[axis]:
                idx[axis] = 0  # last upper cell, but periodic conditions
            else:
                return None  # no neighbor

        else:
            # my lower boundary
            if idx[axis] > 0:
                idx[axis] = idx[axis] - 1  # proper cell neighbor on lower side
            elif self.basegrid.periodic[axis]:
                idx[axis] = size - 1  # last lower cell, but periodic conditions
            else:
                return None  # no neighbor

        return self._idx2id(idx)

    def extract_field_data(
        self,
        field_data: np.ndarray,
        node_id: int = None,
        *,
        with_ghost_cells: bool = False,
    ) -> np.ndarray:
        """extract one subfield from a global one

        Args:
            field_data (:class:`~numpy.ndarray`):
                The field data that will be split
            node_id (int):
                Index identifying the subgrid
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in `field_data`

        Returns:
            :class:`~numpy.ndarray`: Field data defined on the subgrid. The field
            contains ghost cells if `with_ghost_cells == True`.
        """
        # consistency check
        if with_ghost_cells:
            assert field_data.shape[-self.num_axes :] == self.basegrid._shape_full
        else:
            assert field_data.shape[-self.num_axes :] == self.basegrid.shape

        if node_id is None:
            node_id = self.current_node

        node_idx = self._id2idx(node_id)
        idx = self._get_data_indices_1d(with_ghost_cells)
        i = (...,) + tuple(idx[n][j] for n, j in enumerate(node_idx))
        return field_data[i]

    def extract_subfield(
        self, field: TField, node_id: int = None, *, with_ghost_cells: bool = False
    ) -> TField:
        """extract one subfield from a global field

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split
            node_id (int):
                Index identifying the subgrid
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data

        Returns:
            :class:`~pde.fields.base.DataFieldBase`: The sub field of the node
        """
        if node_id is None:
            node_id = self.current_node

        if isinstance(field, DataFieldBase):
            # extract data from a single field
            data = self.extract_field_data(
                field._data_full if with_ghost_cells else field.data,
                node_id,
                with_ghost_cells=with_ghost_cells,
            )

            return field.__class__(  # type: ignore
                grid=self[node_id],
                data=data,
                label=field.label,
                dtype=field.dtype,
                with_ghost_cells=with_ghost_cells,
            )

        elif isinstance(field, FieldCollection):
            # extract data from a field collection

            # extract individual fields
            fields = [
                self.extract_subfield(f, node_id, with_ghost_cells=with_ghost_cells)
                for f in field
            ]

            # combine everything to a field collection
            return field.__class__(fields, label=field.label)  # type: ignore

        else:
            raise TypeError(f"Field type {field.__class__.__name__} unsupported")

    def extract_boundary_conditions(self, bcs_base: Boundaries) -> Boundaries:
        """extract boundary conditions for current subgrid from global conditions

        Args:
            bcs_base (:class:`~pde.grids.boundaries.axes.Boundaries`):
                The boundary conditions that will be split

        Returns:
            :class:`~pde.grids.boundaries.axes.Boundaries`: Boundary conditions of the
            sub grid
        """
        bcs = []
        for axis in range(self.num_axes):
            bcs_axis = []
            for upper in [False, True]:
                bc = bcs_base[axis][upper]
                if self.get_neighbor(axis, upper=upper) is None:
                    bc = bc.to_subgrid(self.current_grid)  # extract BC for subgrid
                else:
                    bc = _MPIBC(self, axis, upper)  # set an MPI boundary condition
                bcs_axis.append(bc)
            bcs.append(BoundaryPair(*bcs_axis))

        return Boundaries(bcs)

    def split_field_data_mpi(
        self, field_data: np.ndarray, *, with_ghost_cells: bool = False
    ) -> np.ndarray:
        """extract one subfield from a global field

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split. An array with the correct shape and dtype
                also needs to be passed to the receiving nodes.
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data. If `True`,
                `field_data` must be the full data field.

        Returns:
            :class:`~pde.fields.base.DataFieldBase`: The sub field of the current node
        """
        import numba_mpi

        assert len(self) == mpi.size

        if mpi.is_main:
            # send fields to all client processes
            for i in range(1, len(self)):
                subfield_data = self.extract_field_data(
                    field_data, i, with_ghost_cells=with_ghost_cells
                )
                numba_mpi.send(subfield_data, i, MPIFlags.field_split)

            # extract field for the current process
            return self.extract_field_data(
                field_data, 0, with_ghost_cells=with_ghost_cells
            )

        else:
            # receive subfield from main process
            subgrid = self.current_grid

            # determine shape of resulting data
            shape = field_data.shape[: -self.num_axes]
            shape += subgrid._shape_full if with_ghost_cells else subgrid.shape

            subfield_data = np.empty(shape, dtype=field_data.dtype)
            numba_mpi.recv(subfield_data, 0, MPIFlags.field_split)
            return subfield_data

    def split_field_mpi(self: GridMesh, field: TField) -> TField:
        """split a field onto the subgrids by communicating data via MPI

        The ghost cells of the returned fields will be set according to the values of
        the original field.

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split

        Results:
            :class:`~pde.fields.base.DataFieldBase`: The field defined on the subgrid
        """
        assert field.grid == self.basegrid

        if isinstance(field, DataFieldBase):
            # split individual field
            return field.__class__(  # type: ignore
                self.current_grid,
                data=self.split_field_data_mpi(field._data_full, with_ghost_cells=True),
                label=field.label,
                dtype=field.dtype,
                with_ghost_cells=True,
            )

        elif isinstance(field, FieldCollection):
            # split field collection
            field_classes = [f.__class__ for f in field]
            data = self.split_field_data_mpi(field._data_full, with_ghost_cells=True)
            return field.__class__.from_data(  # type: ignore
                field_classes,
                self.current_grid,
                data,
                label=field.label,
                labels=field.labels,
            )

        else:
            raise TypeError(f"Field type {field.__class__.__name__} unsupported")

    def combine_field_data(
        self,
        subfields: Sequence[np.ndarray],
        out: np.ndarray = None,
        *,
        with_ghost_cells: bool = False,
    ) -> np.ndarray:
        """combine data of multiple fields defined on subgrids

        Args:
            subfields (:class:`~numpy.ndarray`):
                The data of the fields that will be combined
            out (:class:`~numpy.ndarray`):
                Full field to which the combined data is written. If `None`, a new array
                is allocated.
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data. If `True`,
                `fields` must be the full data fields.

        Returns:
            :class:`~numpy.ndarray`: Combined field. This is `out` if out is not `None`
        """
        assert len(subfields) == len(self)

        # allocate memory to collect the data
        field0 = subfields[0]
        shape = field0.shape[: -self.num_axes]
        shape += self.basegrid._shape_full if with_ghost_cells else self.basegrid.shape
        if out is None:
            out = np.empty(shape, dtype=field0.dtype)
        else:
            assert out.shape == shape

        # collect data from all fields
        data_idx = self._get_data_indices(with_ghost_cells=with_ghost_cells)
        for i in range(len(self)):
            idx = self._id2idx(i)
            out[(...,) + data_idx[idx]] = subfields[i]

        return out

    def combine_field_data_mpi(
        self,
        subfield: np.ndarray,
        out: np.ndarray = None,
        *,
        with_ghost_cells: bool = False,
    ) -> Optional[np.ndarray]:
        """combine data of all subfields using MPI

        Args:
            subfield (:class:`~numpy.ndarray`):
                The data of the field defined on the current subgrid
            out (:class:`~numpy.ndarray`):
                Full field to which the combined data is written
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data. If `True`,
                `subfield` must be the full data fields.

        Returns:
            :class:`~numpy.ndarray`: Combined field if we are on the main node. For all
            other cases, this function returns `None`. On the main node, this array is
            `out` if `out` was supplied.
        """
        import numba_mpi

        assert len(self) == mpi.size

        if mpi.is_main:
            # simply copy the subfield that is on the main node
            field_data = [subfield]
            element_shape = subfield.shape[: -self.num_axes]

            # collect all subfields from all nodes
            for i in range(1, len(self)):
                if with_ghost_cells:
                    shape = element_shape + self[i]._shape_full
                else:
                    shape = element_shape + self[i].shape
                subfield_data = np.empty(shape, dtype=subfield.dtype)
                numba_mpi.recv(subfield_data, i, MPIFlags.field_combine)
                field_data.append(subfield_data)

            # combine the data into a single field
            return self.combine_field_data(
                field_data, out=out, with_ghost_cells=with_ghost_cells
            )

        else:
            # send our subfield to the main node
            numba_mpi.send(subfield, 0, MPIFlags.field_combine)
            return None

    def broadcast(self, data: TData) -> TData:
        """distribute a value from the main node to all nodes

        Args:
            data: The data that will be broadcasted from the main node

        Returns:
            The same data, but on all nodes
        """
        from mpi4py.MPI import COMM_WORLD

        return COMM_WORLD.bcast(data, root=0)  # type: ignore

    def gather(self, data: TData) -> Optional[List[TData]]:
        """gather a value from all nodes

        Args:
            data: The data that will be sent to the main node

        Returns:
            None on all nodes, except the main node, which receives an ordered list with
            the data from all nodes.
        """
        from mpi4py.MPI import COMM_WORLD

        return COMM_WORLD.gather(data, root=0)

    def allgather(self, data: TData) -> List[TData]:
        """gather a value from reach node and sends them to all nodes

        Args:
            data: The data that will be sent to the main node

        Returns:
            list: data from all nodes.
        """
        from mpi4py.MPI import COMM_WORLD

        return COMM_WORLD.allgather(data)

    @plot_on_axes()
    def plot(self, ax, **kwargs):
        r"""visualize the grid mesh

        Args:
            {PLOT_ARGS}
            \**kwargs: Extra arguments are passed on the to the matplotlib
                plotting routines, e.g., to set the color of the lines
        """
        if self.num_axes not in {1, 2}:
            raise NotImplementedError(
                f"Plotting is not implemented for grids of dimension {self.dim}"
            )

        kwargs.setdefault("color", "k")
        for x in np.arange(self.shape[0] + 1) - 0.5:
            ax.axvline(x, **kwargs)
        ax.set_xlim(-0.5, self.shape[0] - 0.5)
        ax.set_xlabel(self.subgrids.flat[0].axes[0])

        if self.num_axes == 2:
            for y in np.arange(self.shape[1] + 1) - 0.5:
                ax.axhline(y, **kwargs)
            ax.set_ylim(-0.5, self.shape[1] - 0.5)
            ax.set_ylabel(self.subgrids.flat[0].axes[1])

            ax.set_aspect(1)

        for num, idx in enumerate(np.ndindex(self.shape)):
            pos = (idx, 0) if self.num_axes == 1 else idx
            ax.text(
                *pos, str(num), horizontalalignment="center", verticalalignment="center"
            )
