"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde import PDEBase, ScalarField, UnitGrid


def test_controller_abort():
    """test how controller deals with errors"""

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
