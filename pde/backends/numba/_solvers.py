"""Implements numba-accelerated solvers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np

from ...solvers import AdamsBashforthSolver, EulerSolver
from ...solvers.base import AdaptiveSolverBase, SolverBase, _make_dt_adjuster
from .overloads import OnlineStatistics, OnlineStatistics_np
from .utils import jit

if TYPE_CHECKING:
    from ...tools.typing import InnerStepperType, NumericArray, StepperHook, TField


def _make_post_step_hook(solver: SolverBase, state: TField) -> StepperHook:
    """Create a callable that executes the PDE's post-step hook.

    The returned function calls the post-step hook provided by the PDE (if any)
    after each completed time step. If the PDE implements make_post_step_hook,
    this method attempts to obtain both the hook function and an initial value
    for the hook's mutable data by calling
    `post_step_hook, post_step_data_init = self.pde.make_post_step_hook(state, backend)`
    The initial data is stored on the solver instance as `solver.info["post_step_data"]`
    (copied to ensure mutability) and will be passed to the hook when the stepper
    is executed.

    If no hook is provided by the PDE (i.e., `make_post_step_hook` raises
    :class:`NotImplementedError`) or if the solver's `_use_post_step_hook` flag
    is `False`, a no-op hook is returned and `solver.info["post_step_data"]` is set
    to `None`.

    The hook returned by this method always conforms to the signature
    `(state_data: numpy.ndarray, t: float, post_step_data: numpy.ndarray) -> None`
    and is suitable for JIT compilation where supported.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the hook is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            Example field providing the array shape and grid information required
            by the PDE when constructing the post-step hook.

    Returns:
        callable:
            A function that invokes the PDE's post-step hook (or a no-op) with the
            signature described above.
    """
    # get uncompiled post_step_hook
    post_step_hook = solver._make_post_step_hook(state)

    # compile post_step_hook
    post_step_data_type = nb.typeof(solver.info["post_step_data"])
    signature_hook = (nb.typeof(state.data), nb.float64, post_step_data_type)
    post_step_hook = jit(signature_hook)(post_step_hook)

    solver._logger.debug("Compiled post-step hook")
    return post_step_hook


def _make_fixed_stepper(solver: SolverBase, state: TField) -> InnerStepperType:
    """Return a stepper function using an explicit scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted
        dt (float):
            Time step of the explicit stepping.
    """
    dt = float(solver.info["dt"])

    # get compiled version of a single step
    single_step = solver._make_single_step_fixed_dt(state, dt)
    single_step_signature = (nb.typeof(state.data), nb.double)
    single_step = jit(single_step_signature)(single_step)
    post_step_hook = _make_post_step_hook(solver, state)

    # provide compiled function for innermost loop
    compiled_stepper_signature = (
        nb.typeof(state.data),
        nb.double,
        nb.int_,
        nb.typeof(solver.info["post_step_data"]),
    )

    @jit(compiled_stepper_signature)
    def compiled_stepper(
        state_data: NumericArray, t_start: float, steps: int, post_step_data: Any
    ) -> tuple[float, Any]:
        """Perform `steps` steps with fixed time steps."""
        for i in range(steps):
            # calculate the right hand side
            t = t_start + i * dt
            state_data = single_step(state_data, t)
            state_data, post_step_data = post_step_hook(state_data, t, post_step_data)

        return t + dt, post_step_data

    def fixed_stepper(state_data: NumericArray, t_start: float, t_end: float) -> float:
        """Advance `state` from `t_start` to `t_end` using fixed steps."""
        steps = max(1, round((t_end - t_start) / dt))
        post_step_data = solver.info["post_step_data"]
        # call the stepper with fixed time steps
        t_last: float
        t_last, solver.info["post_step_data"] = compiled_stepper(
            state_data, t_start, steps, post_step_data
        )
        solver.info["steps"] += steps
        return t_last

    return fixed_stepper


def _make_adams_bashforth_stepper(
    solver: AdamsBashforthSolver, state: TField
) -> InnerStepperType:
    """Return a stepper function using an explicit scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.adams_bashforth.AdamsBashforthSolver`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted
        dt (float):
            Time step of the explicit stepping.
    """
    if solver.pde.is_sde:
        msg = "Deterministic Adams-Bashforth does not support stochastic equations"
        raise RuntimeError(msg)

    dt = float(solver.info["dt"])
    rhs_pde = solver.pde.make_pde_rhs(state, backend=solver.backend)
    post_step_hook = _make_post_step_hook(solver, state)

    # provide compiled function for innermost loop
    compiled_stepper_signature = (
        nb.typeof(state.data),
        nb.typeof(state.data),
        nb.double,
        nb.int_,
        nb.typeof(solver.info["post_step_data"]),
    )

    @jit(compiled_stepper_signature)
    def compiled_stepper(
        state_data: NumericArray,
        state_prev: NumericArray,
        t_start: float,
        steps: int,
        post_step_data: Any,
    ) -> tuple[float, Any]:
        """Perform `steps` steps with fixed time steps."""
        for i in range(steps):
            # calculate the right hand side
            t = t_start + i * dt
            rhs_prev = rhs_pde(state_prev, t - dt).copy()
            rhs_cur = rhs_pde(state_data, t)
            state_prev[:] = state_data  # save the previous state
            state_data += dt * (1.5 * rhs_cur - 0.5 * rhs_prev)
            state_data, post_step_data = post_step_hook(state_data, t, post_step_data)

        return t + dt, post_step_data

    # allocate memory to store the state of the previous time step; this memory will be
    # initialized at the first call
    state_prev = np.empty_like(state.data)
    init_state_prev = True

    def fixed_stepper(state_data: NumericArray, t_start: float, t_end: float) -> float:
        """Perform `steps` steps with fixed time steps."""
        nonlocal state_prev, init_state_prev

        if init_state_prev:
            # initialize the state_prev with an estimate of the previous step
            state_prev[:] = state_data - dt * rhs_pde(state_data, t_start)
            init_state_prev = False

        steps = max(1, round((t_end - t_start) / dt))
        t_final: float
        t_final, solver.info["post_step_data"] = compiled_stepper(
            state_data, state_prev, t_start, steps, solver.info["post_step_data"]
        )

        solver.info["steps"] += steps
        return t_final

    solver._logger.info("Initialize explicit Adams-Bashforth stepper with dt=%g", dt)
    return fixed_stepper


def _make_adaptive_stepper_general(
    solver: AdaptiveSolverBase, state: TField
) -> InnerStepperType:
    """Return a stepper function using an explicit scheme.

    Args:
        solver (:class:`~pde.solvers.base.AdaptiveSolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: numpy.ndarray,
        t_start: float, t_end: float)`
    """
    # add extra information
    solver.info["dt_adaptive"] = solver.adaptive
    solver.info["dt_statistics"] = OnlineStatistics()

    # obtain functions determining how the PDE is evolved
    single_step_error = solver._make_single_step_error_estimate(state)
    signature_single_step = (nb.typeof(state.data), nb.double, nb.double)
    single_step_error = jit(signature_single_step)(single_step_error)
    post_step_hook = _make_post_step_hook(solver, state)
    sync_errors = solver.backend.make_mpi_synchronizer(operator="MAX")

    # obtain auxiliary functions
    signature = (nb.double, nb.double)
    adjust_dt = jit(signature)(_make_dt_adjuster(solver.dt_min, solver.dt_max))
    tolerance = solver.tolerance
    dt_min = solver.dt_min

    # provide compiled function for innermost loop
    def compiled_stepper(
        state_data: NumericArray,
        t_start: float,
        t_end: float,
        dt_init: float,
        dt_stats: OnlineStatistics_np | None = None,
        post_step_data=None,
    ) -> tuple[float, float, int, Any]:
        """Adaptive stepper that advances the state in time."""
        dt_opt = dt_init
        t = t_start
        steps = 0
        while True:
            # use a smaller (but not too small) time step if close to t_end
            dt_step = max(min(dt_opt, t_end - t), dt_min)

            # try two different step sizes to estimate errors
            new_state, error = single_step_error(state_data, t, dt_step)

            error_rel = error / tolerance  # normalize error to given tolerance
            # synchronize the error between all processes (necessary for MPI)
            error_rel = sync_errors(error_rel)

            # do the step if the error is sufficiently small
            if error_rel <= 1:
                steps += 1
                t += dt_step
                state_data[...] = new_state
                state_data, post_step_data = post_step_hook(
                    state_data, t, post_step_data
                )

                if dt_stats is not None:
                    dt_stats.add(dt_step)

            if t < t_end:
                # adjust the time step and continue (happens in every MPI process)
                dt_opt = adjust_dt(dt_step, error_rel)
            else:
                break  # return to the controller

        return t, dt_opt, steps, post_step_data

    if not nb.config.DISABLE_JIT:
        # compile the stepper if enabled for numba
        compiled_stepper_signature = (
            nb.typeof(state.data),
            nb.double,
            nb.double,
            nb.double,
            nb.typeof(solver.info["dt_statistics"]),
            nb.typeof(solver.info["post_step_data"]),
        )
        compiled_stepper = jit(compiled_stepper_signature)(compiled_stepper)

    def adaptive_stepper(
        state_data: NumericArray, t_start: float, t_end: float
    ) -> float:
        """Adaptive stepper that advances the state in time."""
        dt_opt = solver.info["dt"]  # time step from last step

        # call compiled stepper
        t_final: float
        t_final, dt_opt, steps, solver.info["post_step_data"] = compiled_stepper(
            state_data,
            t_start,
            t_end,
            dt_init=dt_opt,
            dt_stats=solver.info["dt_statistics"],
            post_step_data=solver.info["post_step_data"],
        )

        solver.info["dt"] = dt_opt  # save last optimal time step
        solver.info["steps"] += steps
        return t_final

    solver._logger.info("Initialized adaptive stepper")
    return adaptive_stepper


def _make_adaptive_stepper_euler(
    solver: AdaptiveSolverBase, state: TField
) -> InnerStepperType:
    """Return a stepper function using an explicit scheme.

    Args:
        solver (:class:`~pde.solvers.explicit.EulerSolver`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: numpy.ndarray,
        t_start: float, t_end: float)`
    """
    # add extra information
    solver.info["dt_adaptive"] = solver.adaptive
    solver.info["dt_statistics"] = OnlineStatistics()

    # obtain functions determining how the PDE is evolved
    rhs_pde = solver.backend.make_pde_rhs(solver.pde, state)
    single_step_error = solver._make_single_step_error_estimate(state)
    signature_single_step = (nb.typeof(state.data), nb.double, nb.double)
    single_step_error = jit(signature_single_step)(single_step_error)
    post_step_hook = _make_post_step_hook(solver, state)
    sync_errors = solver.backend.make_mpi_synchronizer(operator="MAX")

    # obtain auxiliary functions
    signature = (nb.double, nb.double)
    adjust_dt = jit(signature)(_make_dt_adjuster(solver.dt_min, solver.dt_max))
    tolerance = solver.tolerance
    dt_min = solver.dt_min

    def compiled_stepper(
        state_data: NumericArray,
        t_start: float,
        t_end: float,
        dt_init: float,
        dt_stats: OnlineStatistics_np | None = None,
        post_step_data=None,
    ) -> tuple[float, float, int, Any]:
        """Adaptive stepper that advances the state in time."""
        state_cur = state_data
        dt_opt = dt_init
        rate = rhs_pde(state_data, t_start)  # calculate initial rate

        t = t_start
        steps = 0
        while True:
            # use a smaller (but not too small) time step if close to t_end
            dt_step = max(min(dt_opt, t_end - t), dt_min)

            # do single step with dt
            step_large = state_cur + dt_step * rate
            # do double step with half the time step
            step_small = state_cur + 0.5 * dt_step * rate

            try:
                # calculate rate at the midpoint of the double step
                rate_midpoint = rhs_pde(step_small, t + 0.5 * dt_step)
            except Exception:
                # an exception likely signals that rate could not be calculated
                error_rel = np.nan
            else:
                # advance to end of double step
                step_small += 0.5 * dt_step * rate_midpoint

                # calculate maximal error
                error = np.abs(step_large - step_small).max()
                error_rel = error / tolerance  # normalize error to given tolerance

            # synchronize the error between all processes (necessary for MPI)
            error_rel = sync_errors(error_rel)

            if error_rel <= 1:  # error is sufficiently small
                try:
                    # calculating the rate at putative new step
                    rate = rhs_pde(step_small, t)
                except Exception:
                    # calculating the rate failed => retry with smaller dt
                    error_rel = np.nan
                else:
                    # everything worked => do the step
                    steps += 1
                    t += dt_step
                    state_cur, post_step_data = post_step_hook(
                        step_small, t, post_step_data
                    )
                    if dt_stats is not None:
                        dt_stats.add(dt_step)

            if t < t_end:
                # adjust the time step and continue
                dt_opt = adjust_dt(dt_step, error_rel)
            else:
                break  # return to the controller

        state_data[:] = state_cur
        return t, dt_opt, steps, post_step_data

    if not nb.config.DISABLE_JIT:
        # compile the stepper if enabled for numba
        compiled_stepper_signature = (
            nb.typeof(state.data),
            nb.double,
            nb.double,
            nb.double,
            nb.typeof(solver.info["dt_statistics"]),
            nb.typeof(solver.info["post_step_data"]),
        )
        compiled_stepper = jit(compiled_stepper_signature)(compiled_stepper)

    def adaptive_stepper(
        state_data: NumericArray, t_start: float, t_end: float
    ) -> float:
        """Adaptive stepper that advances the state in time."""
        dt_opt = solver.info["dt"]  # time step from last step

        # call compiled stepper
        t_final: float
        t_final, dt_opt, steps, solver.info["post_step_data"] = compiled_stepper(
            state_data,
            t_start,
            t_end,
            dt_init=dt_opt,
            dt_stats=solver.info["dt_statistics"],
            post_step_data=solver.info["post_step_data"],
        )

        solver.info["dt"] = dt_opt  # save last optimal time step
        solver.info["steps"] += steps
        return t_final

    solver._logger.info("Initialized adaptive stepper")
    return adaptive_stepper


def make_inner_stepper(solver: SolverBase, state: TField) -> InnerStepperType:
    """Return a stepper function using an explicit scheme.

    Args:
        solver (:class:`~pde.solvers.base.AdaptiveSolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: numpy.ndarray,
        t_start: float, t_end: float)`
    """
    # get the actual inner stepper
    if isinstance(solver, AdaptiveSolverBase) and solver.adaptive:
        # dealing with an adaptive stepper
        if isinstance(solver, EulerSolver) and solver.adaptive:
            return _make_adaptive_stepper_euler(solver, state)
        return _make_adaptive_stepper_general(solver, state)

    # dealing with an fixed stepper
    if isinstance(solver, AdamsBashforthSolver):
        return _make_adams_bashforth_stepper(solver, state)
    return _make_fixed_stepper(solver, state)
