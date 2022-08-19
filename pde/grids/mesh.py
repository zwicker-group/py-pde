"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from enum import IntEnum
from typing import Any, List, Tuple, TypeVar, Optional, Sequence, NamedTuple

import numpy as np

from ..fields.base import DataFieldBase, FieldBase
from ..tools.cache import cached_property
from .base import GridBase


class MPIFlags(IntEnum):
    field_split = 1
    field_combine = 2
    boundary_lower = 3
    boundary_upper = 4


class BoundaryInfo(NamedTuple):
    cell: int
    idx: Tuple
    flag: int


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
    """handles a collection of subgrids arranged in a regular mesh"""

    def __init__(self, basegrid: GridBase, subgrids):
        """
        Args:
            basegrid (:class:`~pde.grids.base.GridBase`):
                The grid of the entire domain
            subgrids (nested lists of :class:`~pde.grids.base.GridBase`):
                The nested grids representing the subdivision
        """
        self.basegrid = basegrid
        self.subgrids = np.asarray(subgrids)

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

    def __getitem__(self, mesh_id: int) -> GridBase:
        """extract one subgrid from the mesh

        Args:
            mesh_id (int):
                Index identifying the subgrid
        """
        return self.subgrids[self._id2idx(mesh_id)]

    def _id2idx(self, mesh_id: int) -> Tuple[int, ...]:
        """convert linear id into grid index

        Args:
            mesh_id (int):
                Linear index identifying the subgrid

        Returns:
            tuple: Full index with `num_axes` entries.
        """
        return np.unravel_index(mesh_id, self.shape)  # type: ignore

    def _idx2id(self, mesh_idx: Tuple[int, ...]) -> int:
        """convert grid index to linear index

        Args:
            mesh_idx (tuple):
                Full index with `num_axes` entries.


        Returns:
            int: Linear index identifying the subgrid
        """
        return np.ravel_multi_index(mesh_idx, self.shape)  # type: ignore

    @cached_property()
    def _data_indices_1d(self) -> List[List[slice]]:
        """indices to extract valid field data for each subgrid"""
        # create indices for each axis
        indices_1d = []
        for axis in range(self.num_axes):
            grid_ids: List[Any] = [0] * self.num_axes
            grid_ids[axis] = slice(None, None)
            data, last = [], 0
            for grid in self.subgrids[tuple(grid_ids)]:
                n = grid.shape[axis]
                data.append(slice(last, last + n))
                last += n
            indices_1d.append(data)
        return indices_1d

    @cached_property()
    def _data_indices_valid(self) -> np.ndarray:
        """indices to extract valid field data for each subgrid"""
        indices_1d = self._data_indices_1d
        # combine everything into a full indices
        indices = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            indices[idx] = tuple(indices_1d[n][i] for n, i in enumerate(idx))

        return indices

    def get_boundaries_send(self, mesh_id: int) -> List[BoundaryInfo]:
        """return information about boundary cells that need to be sent to other cells"""
        cell_idx = self._id2idx(mesh_id)
        grid = self[mesh_id]
        result = []
        for axis in range(self.num_axes):
            size = self.shape[axis]
            if size > 1:
                # my lower boundary (upper boundary of left cell)
                idx_left = list(cell_idx)
                idx_left[axis] = (cell_idx[axis] - 1) % size
                my_idx = [...] + [slice(None)] * grid.num_axes
                my_idx[1 + axis] = 0
                b = BoundaryInfo(
                    cell=self._idx2id(idx_left),
                    idx=tuple(my_idx),
                    flag=MPIFlags.boundary_lower + 16 * axis,
                )
                result.append(b)

                # my upper boundary (lower boundary of right cell)
                idx_right = list(cell_idx)
                idx_right[axis] = (cell_idx[axis] + 1) % size
                my_idx = [...] + [slice(None)] * grid.num_axes
                my_idx[1 + axis] = -1
                b = BoundaryInfo(
                    cell=self._idx2id(idx_right),
                    idx=tuple(my_idx),
                    flag=MPIFlags.boundary_upper + 16 * axis,
                )
                result.append(b)

        return result

    def get_boundaries_receive(self, mesh_id: int) -> List[BoundaryInfo]:
        """return information about boundary cells that need to be sent to other cells"""
        cell_idx = self._id2idx(mesh_id)
        grid = self[mesh_id]
        result = []
        for axis in range(self.num_axes):
            size = self.shape[axis]
            if size > 1:
                # my lower boundary (upper boundary of left cell)
                idx_left = list(cell_idx)
                idx_left[axis] = (cell_idx[axis] - 1) % size
                my_idx = [...] + [slice(1, -1)] * grid.num_axes
                my_idx[1 + axis] = 0
                b = BoundaryInfo(
                    cell=self._idx2id(idx_left),
                    idx=tuple(my_idx),
                    flag=MPIFlags.boundary_upper + 16 * axis,
                )
                result.append(b)

                # my upper boundary (lower boundary of right cell)
                idx_right = list(cell_idx)
                idx_right[axis] = (cell_idx[axis] + 1) % size
                my_idx = [...] + [slice(1, -1)] * grid.num_axes
                my_idx[1 + axis] = -1
                b = BoundaryInfo(
                    cell=self._idx2id(idx_right),
                    idx=tuple(my_idx),
                    flag=MPIFlags.boundary_lower + 16 * axis,
                )
                result.append(b)

        return result

    def extract_subfield(self, field: TField, mesh_id: int) -> TField:
        """extract one subfield from a global one

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split
            mesh_id (int):
                Index identifying the subgrid
        """
        mesh_idx = self._id2idx(mesh_id)
        grid = self.subgrids[mesh_idx]
        i = (...,) + tuple(self._data_indices_1d[n][j] for n, j in enumerate(mesh_idx))
        if isinstance(field, DataFieldBase):
            return field.__class__(grid, data=field.data[i], dtype=field.dtype)
        else:
            raise NotImplementedError

    def extract_field_data(self, field: np.ndarray, mesh_id: int) -> np.ndarray:
        """extract one subfield from a global one

        Args:
            field (:class:`~numpy.ndarray`):
                The field that will be split
            mesh_id (int):
                Index identifying the subgrid
        """
        mesh_idx = self._id2idx(mesh_id)
        i = (...,) + tuple(self._data_indices_1d[n][j] for n, j in enumerate(mesh_idx))
        return np.ascontiguousarray(field.data)[i]

    def split_field(self, field: TField, with_ghost_cells: bool = False) -> np.ndarray:
        """split a field onto the subgrids

        Args:
            field (:class:`~pde.fields.base.DataFieldBase`):
                The field that will be split
        """
        assert field.grid == self.basegrid
        result = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            grid = self.subgrids[idx]
            data = field.data[(...,) + self._data_indices_valid[idx]]
            result[idx] = field.__class__(grid, data=data, dtype=field.dtype)  # type: ignore
        return result

    def split_field_data_mpi(self, field: np.ndarray = None) -> np.ndarray:

        import numba_mpi

        assert len(self) == numba_mpi.size()

        if numba_mpi.rank() == 0:
            # send fields to all client processes
            for i in range(1, len(self)):
                subfield = self.extract_field_data(field, i)
                numba_mpi.send(subfield, i, MPIFlags.field_split)
            return self.extract_field_data(field, 0)

        else:
            # receive subfield from main process
            subgrid = self[numba_mpi.rank()]
            # TODO: support different ranks
            subfield = np.empty(subgrid.shape, dtype=field.dtype)
            numba_mpi.recv(subfield, 0, MPIFlags.field_split)
            return subfield

    def combine_fields(self, fields: np.ndarray) -> FieldBase:
        """combine multiple fields defined on subgrids

        Args:
            fields (:class:`~numpy.ndarray` of :class:`~pde.fields.base.FieldBase`):
                The fields that will be combined
        """
        # prepare to collect data
        field0 = fields.flat[0]
        shape = (self.basegrid.dim,) * field0.rank + self.basegrid.shape
        data = np.empty(shape, dtype=field0.dtype)

        for idx in np.ndindex(self.shape):
            assert self.subgrids[idx] == fields[idx].grid
            data[(...,) + self._data_indices_valid[idx]] = fields[idx].data

        return field0.__class__(self.basegrid, data=data, dtype=field0.dtype)

    def combine_field_data(self, fields: Sequence[np.ndarray]) -> np.ndarray:
        """combine data of multiple fields defined on subgrids

        Args:
            fields (:class:`~numpy.ndarray`):
                The data of the fields that will be combined
        """
        # prepare to collect data
        field0 = fields[0]
        rank = field0.ndim - self.basegrid.num_axes
        shape = (self.basegrid.dim,) * rank + self.basegrid.shape
        data = np.empty(shape, dtype=field0.dtype)

        for i in range(len(self)):
            idx = self._id2idx(i)
            data[(...,) + self._data_indices_valid[idx]] = fields[i]

        return data

    def combine_field_data_mpi(self, subfield: np.ndarray) -> Optional[np.ndarray]:

        import numba_mpi

        assert len(self) == numba_mpi.size()

        if numba_mpi.rank() == 0:
            fields = [subfield]
            for i in range(1, len(self)):
                subfield = np.empty(self[i].shape, dtype=subfield.dtype)
                numba_mpi.recv(subfield, i, MPIFlags.field_combine)
                fields.append(subfield)
            return self.combine_field_data(fields)

        else:
            numba_mpi.send(np.ascontiguousarray(subfield), 0, MPIFlags.field_combine)
            return None
