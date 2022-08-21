"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from enum import IntEnum
from typing import Any, List, Tuple, TypeVar, Optional, Sequence, NamedTuple

import numpy as np

from ..fields.base import DataFieldBase, FieldBase
from ..tools.cache import cached_method
from .base import GridBase
from ..tools.plotting import plot_on_axes


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


class MPIBoundaryInfo(NamedTuple):
    """contains information about a boundary between two nodes"""

    cell: int  # id of the neighboring cell
    flag: int  # MPI flag describing the connection


def _subdivide(num: int, chunks: int) -> np.ndarray:
    r"""subdivide `num` intervals in `chunk` chunks

    Args:
        num (int): Number of intervals
        chunks (int): Number of chunks

    Returns:
        list: The number of intervals per chunk
    """
    if chunks > num:
        raise RuntimeError("Cannot divide in more chunks than support points ")
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
    def from_grid(cls, grid: GridBase, decomposition: List[int]):
        """subdivide the grid into subgrids

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid that will be subdivided
            decomposition (list of ints):
                Number of subdivision in each direction. Must be a list of
                length `grid.num_axes` or a single integer. In the latter case,
                the same subdivision is assumed for each axes.

        Returns:
            :class:`GridMesh`
        """
        decomp_arr = np.broadcast_to(decomposition, (grid.num_axes,)).astype(int)

        subgrids = np.empty(decomp_arr, dtype=object)
        subgrids.flat[0] = grid  # seed the initial grid at the top-left
        idx_set: List[Any] = [0] * subgrids.ndim  # indices to extract all grids
        for axis, chunks in enumerate(decomp_arr):
            # iterate over all grids that have been determined already
            for idx, subgrid in np.ndenumerate(subgrids[tuple(idx_set)]):
                i = idx + (slice(None, None),) + (0,) * (subgrids.ndim - axis - 1)
                divison = _subdivide_along_axis(subgrid, axis=axis, chunks=chunks)
                subgrids[i] = divison

            # mark this axis as set
            idx_set[axis] = slice(None, None)

        return cls(basegrid=grid, subgrids=subgrids)

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

    def __getitem__(self, node_id: int) -> GridBase:
        """extract one subgrid from the mesh

        Args:
            node_id (int):
                Index identifying the subgrid

        Returns:
            :class:`~pde.grids.base.GridBase`: The sub grid of the given cell
        """
        return self.subgrids[self._id2idx(node_id)]

    @property
    def current_node(self) -> int:
        """int: current MPI node"""
        import numba_mpi

        return numba_mpi.rank()

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

    def _idx2id(self, node_idx: Tuple[int, ...]) -> int:
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

    def _get_cell_boundary_info(
        self, node_id: int = None, axis: int = 0, upper: bool = False
    ) -> Optional[MPIBoundaryInfo]:
        """get boundary cells that need to be sent to other nodes along given axis

        Args:
            node_id (int):
                The ID of the considered node
            axis (int):
                The axis of the grid
            upper (bool):
                Flag deciding which boundary to consider

        Returns:
            :class:`MPIBoundaryInfo`: Information about the given boundary or `None` if
            the boundary is not subdivided in the mesh
        """
        # TODO: Move this to the boundary class
        size = self.shape[axis]
        if size == 1:
            return None

        if node_id is None:
            node_id = self.current_node

        other_idx = list(self._id2idx(node_id))

        if upper:
            # my upper boundary (lower boundary of right cell)
            other_idx[axis] = (other_idx[axis] + 1) % size
            other_id = self._idx2id(other_idx)
            link_flag = MPIFlags.boundary_upper(node_id, other_id)

        else:
            # my lower boundary (upper boundary of left cell)
            other_idx[axis] = (other_idx[axis] - 1) % size
            other_id = self._idx2id(other_idx)
            link_flag = MPIFlags.boundary_lower(node_id, other_id)

        return MPIBoundaryInfo(cell=other_id, flag=link_flag)

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
            data = self.extract_field_data(
                field._data_full if with_ghost_cells else field.data,
                node_id,
                with_ghost_cells=with_ghost_cells,
            )

            return field.__class__(
                grid=self[node_id],
                data=data,
                dtype=field.dtype,
                with_ghost_cells=with_ghost_cells,
            )
        else:
            # TODO: support FieldCollections, too
            raise NotImplementedError

    def split_field_data_mpi(
        self, field_data: np.ndarray = None, *, with_ghost_cells: bool = False
    ) -> np.ndarray:
        """extract one subfield from a global field

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data. If `True`,
                `field_data` must be the full data field.

        Returns:
            :class:`~pde.fields.base.DataFieldBase`: The sub field of the current node
        """
        import numba_mpi

        assert len(self) == numba_mpi.size()

        if numba_mpi.rank() == 0:
            # send fields to all client processes
            for i in range(1, len(self)):
                subfield = self.extract_field_data(
                    field_data, i, with_ghost_cells=with_ghost_cells
                )
                numba_mpi.send(subfield, i, MPIFlags.field_split)
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

    def split_field_mpi(self, field: TField) -> TField:
        """split a field onto the subgrids by communicating data via MPI

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split

        Results:
            :class:`~pde.fields.base.DataFieldBase`: The field defined on the subgrid
        """
        assert field.grid == self.basegrid
        return field.__class__(
            self.current_grid,
            data=self.split_field_data_mpi(field._data_full, with_ghost_cells=True),
            label=field.label,
            dtype=field.dtype,
            with_ghost_cells=True,
        )

    # def combine_fields(self, fields: np.ndarray) -> FieldBase:
    #     """combine multiple fields defined on subgrids
    #
    #     Args:
    #         fields (:class:`~numpy.ndarray` of :class:`~pde.fields.base.FieldBase`):
    #             The fields that will be combined
    #     """
    #     # prepare to collect data
    #     field0 = fields.flat[0]
    #     shape = (self.basegrid.dim,) * field0.rank + self.basegrid.shape
    #     data = np.empty(shape, dtype=field0.dtype)
    #     data_idx = self._get_data_indices(with_ghost_cells=False)
    #
    #     for idx in np.ndindex(self.shape):
    #         assert self.subgrids[idx] == fields[idx].grid
    #         data[(...,) + data_idx[idx]] = fields[idx].data
    #
    #     return field0.__class__(self.basegrid, data=data, dtype=field0.dtype)

    def combine_field_data(
        self, fields: Sequence[np.ndarray], out: np.ndarray = None
    ) -> np.ndarray:
        """combine data of multiple fields defined on subgrids

        Args:
            fields (:class:`~numpy.ndarray`):
                The data of the fields that will be combined
            out (:class:`~numpy.ndarray`):
                Full field to which the combined data is written

        Returns:
            :class:`~numpy.ndarray`: Combined field. This is `out` if out is not `None`
        """
        # prepare to collect data
        field0 = fields[0]
        rank = field0.ndim - self.basegrid.num_axes
        shape = (self.basegrid.dim,) * rank + self.basegrid.shape
        if out is None:
            out = np.empty(shape, dtype=field0.dtype)

        data_idx = self._get_data_indices(with_ghost_cells=False)
        for i in range(len(self)):
            idx = self._id2idx(i)
            out[(...,) + data_idx[idx]] = fields[i]

        return out

    def combine_field_data_mpi(
        self, subfield: np.ndarray, out: np.ndarray = None
    ) -> Optional[np.ndarray]:
        """combine data of all subfields using MPI

        Args:
            subfield (:class:`~numpy.ndarray`):
                The data of the field defined on the current subgrid
            out (:class:`~numpy.ndarray`):
                Full field to which the combined data is written

        Returns:
            :class:`~numpy.ndarray`: Combined field if we are on the main node. For all
            other cases, this function returns `None`. On the main node, this array is
            `out` if `out` was supplied.
        """
        import numba_mpi

        assert len(self) == numba_mpi.size()

        if numba_mpi.rank() == 0:
            # main node that receives all data from all other nodes
            fields = [subfield]
            for i in range(1, len(self)):
                subfield = np.empty(self[i].shape, dtype=subfield.dtype)
                numba_mpi.recv(subfield, i, MPIFlags.field_combine)
                fields.append(subfield)
            return self.combine_field_data(fields, out=out)

        else:
            # send our subfield to the main node
            numba_mpi.send(subfield, 0, MPIFlags.field_combine)
            return None

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
