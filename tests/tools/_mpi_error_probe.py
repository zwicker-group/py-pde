"""Exercise a failing solver from an isolated MPI test process."""

from pde import PDE, ScalarField, UnitGrid
from pde.tools import mpi

if __name__ == "__main__":
    assert mpi.size == 2
    PDE({"c": "undefined_rhs"}).solve(
        ScalarField(UnitGrid([2])), solver="explicit_mpi", t_range=1
    )
