"""Implements jax-accelerated solvers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from ...solvers import AdamsBashforthSolver, EulerSolver, RungeKuttaSolver, ScipySolver
from ...solvers.base import AdaptiveSolverBase, SolverBase, _make_dt_adjuster
from ...tools.math import OnlineStatistics

if TYPE_CHECKING:
    from jax import Array

    from ...tools.typing import StepperHook, TField
    from .typing import JaxInnerStepperType


TESTED_SOLVERS = [
    EulerSolver,
    RungeKuttaSolver,
    AdamsBashforthSolver,
    ScipySolver,
]


def _make_post_step_hook(solver: SolverBase, state: TField) -> StepperHook:
    """Create a callable that executes the PDE's post-step hook for JAX.

    The returned function calls the post-step hook provided by the PDE (if any)
    after each completed time step.  If no hook is provided (or
    :attr:`SolverBase._use_post_step_hook` is ``False``), a no-op is returned and
    ``solver.info["post_step_data"]`` is set to ``None``.

    The hook always has the signature
    ``(state_data, t: float, post_step_data) -> (state_data, post_step_data)``
    and operates on JAX arrays.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance.
        state (:class:`~pde.fields.base.FieldBase`):
            Example field providing grid information for the PDE hook.

    Returns:
        callable: The (possibly no-op) post-step hook.
    """
    post_step_hook = solver._make_post_step_hook(state)
    solver._logger.debug("Created post-step hook")
    return post_step_hook


def _make_fixed_stepper(solver: SolverBase, state: TField) -> JaxInnerStepperType:
    """Return a stepper function using an explicit scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed.
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which grid and other information can be
            extracted.

    Returns:
        callable:
            Function that advances the state from ``t_start`` to ``t_end`` with
            signature ``(state_data, t_start, t_end) -> (state_data, t_final)``.
    """
    dt = float(solver.info["dt"])

    # get compiled version of a single step
    single_step = solver._make_single_step_fixed_dt(state, dt)
    single_step = solver.backend.compile_function(single_step)
    post_step_hook = _make_post_step_hook(solver, state)

    def fixed_stepper(
        state_data: Array, t_start: float, t_end: float
    ) -> tuple[Array, float]:
        """Advance ``state`` from ``t_start`` to ``t_end`` using fixed steps."""
        steps = max(1, round((t_end - t_start) / dt))
        for i in range(steps):
            t = t_start + i * dt
            state_data = single_step(state_data, t)  # type: ignore
            state_data, solver.info["post_step_data"] = post_step_hook(
                state_data, t, solver.info["post_step_data"]
            )
        solver.info["steps"] += steps
        return state_data, t_start + steps * dt

    solver._logger.info("Initialized fixed stepper with dt=%g", dt)
    return fixed_stepper


def _make_adams_bashforth_stepper(
    solver: AdamsBashforthSolver, state: TField
) -> JaxInnerStepperType:
    """Return a stepper function using the Adams-Bashforth scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.adams_bashforth.AdamsBashforthSolver`):
            The solver instance.
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which grid and other information can be
            extracted.

    Returns:
        callable:
            Function that advances the state from ``t_start`` to ``t_end`` with
            signature ``(state_data, t_start, t_end) -> (state_data, t_final)``.
    """
    if solver.pde.is_sde:
        msg = "Deterministic Adams-Bashforth does not support stochastic equations"
        raise RuntimeError(msg)

    dt = float(solver.info["dt"])
    rhs_pde = solver.pde.make_pde_rhs(state, backend=solver.backend)
    rhs_pde = solver.backend.compile_function(rhs_pde)
    post_step_hook = _make_post_step_hook(solver, state)

    # use a mutable container so the inner function can update the previous state
    _state_prev: list[Array | None] = [None]

    def fixed_stepper(
        state_data: Array, t_start: float, t_end: float
    ) -> tuple[Array, float]:
        """Advance ``state`` from ``t_start`` to ``t_end`` using Adams-Bashforth."""
        if _state_prev[0] is None:
            # initialise the previous state with a backward Euler estimate
            _state_prev[0] = state_data - dt * rhs_pde(state_data, t_start)

        state_prev = _state_prev[0]
        steps = max(1, round((t_end - t_start) / dt))

        for i in range(steps):
            t = t_start + i * dt
            rhs_prev = rhs_pde(state_prev, t - dt)  # type: ignore
            rhs_cur = rhs_pde(state_data, t)
            state_prev = state_data
            state_data = state_data + dt * (1.5 * rhs_cur - 0.5 * rhs_prev)  # type: ignore
            state_data, solver.info["post_step_data"] = post_step_hook(
                state_data, t, solver.info["post_step_data"]
            )

        _state_prev[0] = state_prev
        solver.info["steps"] += steps
        return state_data, t_start + steps * dt

    solver._logger.info("Initialized explicit Adams-Bashforth stepper with dt=%g", dt)
    return fixed_stepper


def _make_adaptive_stepper_general(
    solver: AdaptiveSolverBase, state: TField
) -> JaxInnerStepperType:
    """Return a stepper function using an explicit adaptive scheme.

    Args:
        solver (:class:`~pde.solvers.base.AdaptiveSolverBase`):
            The solver instance, which determines how the stepper is constructed.
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which grid and other information can be
            extracted.

    Returns:
        callable:
            Function that advances the state from ``t_start`` to ``t_end`` with
            signature ``(state_data, t_start, t_end) -> (state_data, t_final)``.
    """
    # add extra information
    solver.info["dt_adaptive"] = solver.adaptive
    solver.info["dt_statistics"] = OnlineStatistics()

    # obtain functions determining how the PDE is evolved
    single_step_error = solver._make_single_step_error_estimate(state)
    single_step_error = solver.backend.compile_function(single_step_error)
    post_step_hook = _make_post_step_hook(solver, state)

    # obtain auxiliary functions
    adjust_dt = _make_dt_adjuster(solver.dt_min, solver.dt_max)
    tolerance = solver.tolerance
    dt_min = solver.dt_min

    def adaptive_stepper(
        state_data: Array, t_start: float, t_end: float
    ) -> tuple[Array, float]:
        """Adaptive stepper that advances the state in time."""
        dt_opt = float(solver.info["dt"])
        t = t_start
        steps = 0

        while t < t_end:
            # use a smaller (but not too small) time step if close to t_end
            dt_step = max(min(dt_opt, t_end - t), dt_min)

            new_state, error = single_step_error(state_data, t, dt_step)  # type: ignore
            error_rel = float(error) / tolerance

            if error_rel <= 1:
                steps += 1
                t += dt_step
                state_data = new_state  # type: ignore
                state_data, solver.info["post_step_data"] = post_step_hook(
                    state_data, t, solver.info["post_step_data"]
                )
                solver.info["dt_statistics"].add(dt_step)

            if t < t_end:
                dt_opt = adjust_dt(dt_step, error_rel)
            else:
                break

        solver.info["dt"] = dt_opt
        solver.info["steps"] += steps
        return state_data, t

    solver._logger.info("Initialized adaptive stepper")
    return adaptive_stepper


def _make_adaptive_stepper_euler(
    solver: AdaptiveSolverBase, state: TField
) -> JaxInnerStepperType:
    """Return a stepper function using the adaptive Euler scheme.

    Args:
        solver (:class:`~pde.solvers.explicit.EulerSolver`):
            The solver instance, which determines how the stepper is constructed.
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which grid and other information can be
            extracted.

    Returns:
        callable:
            Function that advances the state from ``t_start`` to ``t_end`` with
            signature ``(state_data, t_start, t_end) -> (state_data, t_final)``.
    """
    # add extra information
    solver.info["dt_adaptive"] = solver.adaptive
    solver.info["dt_statistics"] = OnlineStatistics()

    # obtain functions determining how the PDE is evolved
    rhs_pde = solver.backend.make_pde_rhs(solver.pde, state)
    rhs_pde = solver.backend.compile_function(rhs_pde)
    post_step_hook = _make_post_step_hook(solver, state)

    # obtain auxiliary functions
    adjust_dt = _make_dt_adjuster(solver.dt_min, solver.dt_max)
    tolerance = solver.tolerance
    dt_min = solver.dt_min

    def adaptive_stepper(
        state_data: Array, t_start: float, t_end: float
    ) -> tuple[Array, float]:
        """Adaptive Euler stepper that advances the state in time."""
        dt_opt = float(solver.info["dt"])
        rate = rhs_pde(state_data, t_start)  # calculate initial rate
        t = t_start
        steps = 0

        while t < t_end:
            # use a smaller (but not too small) time step if close to t_end
            dt_step = max(min(dt_opt, t_end - t), dt_min)

            # do single step with dt
            step_large = state_data + dt_step * rate
            # do double step with half the time step
            step_small = state_data + 0.5 * dt_step * rate

            try:
                # calculate rate at the midpoint of the double step
                rate_midpoint = rhs_pde(step_small, t + 0.5 * dt_step)
            except Exception:
                # an exception likely signals that the rate could not be calculated
                error_rel = float("nan")
            else:
                # advance to end of double step
                step_small = step_small + 0.5 * dt_step * rate_midpoint

                # calculate maximal error (NaN propagates naturally in JAX)
                error = jnp.abs(step_large - step_small).max()
                error_rel = float(error) / tolerance

            if error_rel <= 1:
                try:
                    # calculate rate at the putative new state
                    rate = rhs_pde(step_small, t)
                except Exception:
                    # calculating the rate failed => retry with smaller dt
                    error_rel = float("nan")
                else:
                    # everything worked => do the step
                    steps += 1
                    t += dt_step
                    state_data, solver.info["post_step_data"] = post_step_hook(
                        step_small, t, solver.info["post_step_data"]
                    )
                    state_data = step_small
                    solver.info["dt_statistics"].add(dt_step)

            if t < t_end:
                dt_opt = adjust_dt(dt_step, error_rel)
            else:
                break

        solver.info["dt"] = dt_opt
        solver.info["steps"] += steps
        return state_data, t

    solver._logger.info("Initialized adaptive Euler stepper")
    return adaptive_stepper


def make_inner_stepper(solver: SolverBase, state: TField) -> JaxInnerStepperType:
    """Return a stepper function for the JAX backend.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed.
        state (:class:`~pde.fields.base.FieldBase`):
            Example field from which grid and other information is read.

    Returns:
        callable:
            Function that advances the state from ``t_start`` to ``t_end`` with
            signature ``(state_data, t_start, t_end) -> (state_data, t_final)``.
    """
    if solver.__class__ not in TESTED_SOLVERS:
        solver._logger.warning(
            "Solver %s not supported by backend %s", solver, solver.backend
        )

    # get the actual inner stepper
    if isinstance(solver, AdaptiveSolverBase) and solver.adaptive:
        # dealing with an adaptive stepper
        if isinstance(solver, EulerSolver):
            return _make_adaptive_stepper_euler(solver, state)
        return _make_adaptive_stepper_general(solver, state)

    # dealing with a fixed stepper
    if isinstance(solver, AdamsBashforthSolver):
        return _make_adams_bashforth_stepper(solver, state)
    return _make_fixed_stepper(solver, state)
