"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDEBase, ScalarField, UnitGrid
from pde.solvers import Controller


def test_controller_abort():
    """Test how controller deals with errors."""

    class ErrorPDEException(RuntimeError): ...

    class ErrorPDE(PDEBase):
        def evolution_rate(self, state, t):
            if t < 1:
                return 0 * state
            else:
                raise ErrorPDEException

    field = ScalarField(UnitGrid([16]), 1)
    eq = ErrorPDE()

    with pytest.raises(ErrorPDEException):
        eq.solve(field, t_range=2, dt=0.2, backend="numpy")

    assert eq.diagnostics["last_tracker_time"] >= 0
    assert eq.diagnostics["last_state"] == field


def test_controller_foreign_solver():
    """Test whether the Controller can deal with a minimal foreign solver."""

    class MySolver:
        def make_stepper(self, state, dt):
            def stepper(state, t, t_break):
                return t_break

            return stepper

    c = Controller(MySolver(), t_range=1)
    res = c.run(np.arange(3))
    np.testing.assert_allclose(res, np.arange(3))
