"""Analytic test cases shared by the storage resource generators and the tests.

The cases defined here are written to disk by the scripts
`scripts/create_file_storage_test_resources.py` and
`scripts/create_movie_storage_test_resources.py` and are checked against the same
analytic expectations in `test_backward_compatibility.py`.

Note:
    Changing existing cases invalidates the stored resource files of previous versions.
    Add new cases instead of modifying existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import pde

if TYPE_CHECKING:
    from pde.fields.base import FieldBase

RESOURCES_ROOT = Path(__file__).resolve().parents[1] / "resources"


@dataclass(frozen=True)
class Case:
    """Specification of a single stored trajectory."""

    name: str
    storage_class: str  # either "FileStorage" or "MovieStorage"
    grid: Any  # callable returning the grid
    kind: str  # "scalar", "vector", "tensor", or "collection"
    times: tuple[float, ...]
    dtype: Any = np.double
    equidistant: bool = False  # whether times are reconstructed from `t_start` and `dt`
    storage_kwargs: dict[str, Any] = field(default_factory=dict)
    atol: float = 0  # tolerance when comparing stored data to expectation
    suffix: str | None = None  # file extension; determined by storage class if None

    @property
    def filename(self) -> str:
        """str: name of the file in the storage directory"""
        suffix = self.suffix
        if suffix is None:
            suffix = "hdf5" if self.storage_class == "FileStorage" else "avi"
        return f"{self.name}.{suffix}"

    @property
    def info(self) -> dict[str, Any]:
        """dict: extra information stored along with the data"""
        return {"payload": self.name}

    def make_grid(self) -> pde.GridBase:
        """Create the grid of this case."""
        return self.grid()

    def make_field(self, t: float) -> FieldBase:
        """Create the analytic field at time `t`.

        Args:
            t (float): The time point

        Returns:
            :class:`~pde.fields.base.FieldBase`: The field with analytic data
        """
        grid = self.make_grid()
        # all values are in [0, 1] so they can be stored by the movie storage
        factor = (1 + t) / (1 + max(self.times))

        def ramp(rank: int, offset: float = 0) -> np.ndarray:
            shape = (grid.dim,) * rank + grid.shape
            size = int(np.prod(shape))
            base = (np.arange(size) + 1 + offset) / (size + 1 + offset)
            data = (base * factor).reshape(shape)
            if np.issubdtype(self.dtype, np.complexfloating):
                data = data * (1 + 0.5j) / 1.5
            return data.astype(self.dtype)

        if self.kind == "scalar":
            return pde.ScalarField(grid, ramp(0), label=self.name, dtype=self.dtype)
        if self.kind == "vector":
            return pde.VectorField(grid, ramp(1), label=self.name, dtype=self.dtype)
        if self.kind == "tensor":
            return pde.Tensor2Field(grid, ramp(2), label=self.name, dtype=self.dtype)
        if self.kind == "collection":
            fields = [
                pde.ScalarField(grid, ramp(0), label="a", dtype=self.dtype),
                pde.VectorField(grid, ramp(1, 3), label="b", dtype=self.dtype),
            ]
            return pde.FieldCollection(fields, label=self.name)
        raise NotImplementedError(self.kind)

    def write(self, directory: Path) -> Path:
        """Write the case to a storage file.

        Args:
            directory (:class:`~pathlib.Path`): Directory in which the file is created

        Returns:
            :class:`~pathlib.Path`: The path of the created file
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.filename
        for suffix in ("", ".times"):  # remove stale files
            Path(f"{path}{suffix}").unlink(missing_ok=True)

        info = {**self.info, "pde_version": pde.__version__}
        if self.storage_class == "FileStorage":
            storage: pde.StorageBase = pde.FileStorage(path, info=info)
        else:
            storage = pde.MovieStorage(path, info=info, **self.storage_kwargs)
            if self.equidistant:
                # mimic what the tracker stores for constant intervals
                storage.info["t_start"] = self.times[0]
                storage.info["dt"] = self.times[1] - self.times[0]

        storage.start_writing(self.make_field(self.times[0]))
        for t in self.times:
            storage.append(self.make_field(t), t)
        storage.end_writing()
        return path


def _unit_2d() -> pde.UnitGrid:
    return pde.UnitGrid([4, 3], periodic=[True, False])


def _unit_1d() -> pde.UnitGrid:
    return pde.UnitGrid([8])


def _cylindrical() -> pde.CylindricalSymGrid:
    return pde.CylindricalSymGrid(3, [1, 2], [3, 2], periodic_z=True)


FILE_CASES = [
    Case(
        "scalar_2d", "FileStorage", _unit_2d, "scalar", (0, 1, 2, 3), equidistant=True
    ),
    Case(
        "scalar_1d_float32",
        "FileStorage",
        _unit_1d,
        "scalar",
        (0.1, 0.7, 2.9),
        dtype=np.float32,
    ),
    Case(
        "scalar_2d_complex",
        "FileStorage",
        _unit_2d,
        "scalar",
        (0, 0.5, 1),
        dtype=np.complex128,
    ),
    Case("vector_2d", "FileStorage", _unit_2d, "vector", (0, 1, 2), equidistant=True),
    Case("tensor_cyl", "FileStorage", _cylindrical, "tensor", (0, 0.3, 2.9)),
    Case("collection_2d", "FileStorage", _unit_2d, "collection", (0, 0.2, 1.5)),
    Case("single_frame", "FileStorage", _unit_1d, "scalar", (0.5,)),
]

MOVIE_CASES = [
    Case(
        "scalar_2d",
        "MovieStorage",
        _unit_2d,
        "scalar",
        (0, 1, 2, 3),
        equidistant=True,
        storage_kwargs={"bits_per_channel": 16},
        atol=2e-5,
    ),
    Case(
        "scalar_1d_8bit",
        "MovieStorage",
        _unit_1d,
        "scalar",
        (0.1, 0.7, 2.9),
        storage_kwargs={"bits_per_channel": 8, "write_times": True},
        atol=1 / 255,
    ),
    Case(
        "tensor_2d",
        "MovieStorage",
        _unit_2d,
        "tensor",
        (0, 0.5, 1),
        equidistant=True,
        storage_kwargs={"bits_per_channel": 16},
        atol=2e-5,
    ),
    Case(
        "vector_cyl",
        "MovieStorage",
        _cylindrical,
        "vector",
        (0, 0.3, 2.9),
        storage_kwargs={"bits_per_channel": 16, "write_times": True},
        atol=2e-5,
    ),
    Case(
        "collection_2d",
        "MovieStorage",
        _unit_2d,
        "collection",
        (0, 0.2, 1.5),
        storage_kwargs={
            "bits_per_channel": 16,
            "write_times": True,
            "vmin": [0, 0],
            "vmax": [1, 2],
        },
        atol=5e-5,
    ),
    Case(
        "scalar_2d_mkv",
        "MovieStorage",
        _unit_2d,
        "scalar",
        (0, 0.5, 2),
        suffix="mkv",
        storage_kwargs={"bits_per_channel": 16, "write_times": True},
        atol=2e-5,
    ),
]

CASES = {"FileStorage": FILE_CASES, "MovieStorage": MOVIE_CASES}


def write_cases(storage_class: str, directory: Path) -> list[Path]:
    """Write all cases of a storage class into a directory.

    Args:
        storage_class (str): Either "FileStorage" or "MovieStorage"
        directory (:class:`~pathlib.Path`): Target directory

    Returns:
        list of :class:`~pathlib.Path`: The created files
    """
    return [case.write(directory) for case in CASES[storage_class]]
