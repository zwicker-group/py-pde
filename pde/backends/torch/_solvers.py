"""Implements torch-accelerated solvers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ...solvers import AdamsBashforthSolver, EulerSolver, MilsteinSolver
from ...solvers.base import AdaptiveSolverBase, SolverBase
from ...tools.math import OnlineStatistics
from .backend import TorchBackend

if TYPE_CHECKING:
    from ...tools.typing import StepperHook, TField
    from .typing import TorchInnerStepperType, TorchRHSType


def _make_post_step_hook(
    solver: SolverBase, state: TField, backend: TorchBackend
) -> StepperHook | None:
    """Create a callable that executes the PDE's post-step hook.

    If no hook is provided by the PDE (i.e., `make_post_step_hook` raises
    :class:`NotImplementedError`) or if the solver's `_use_post_step_hook` flag
    is `False`, ``None`` is returned and ``solver.info["post_step_data"]`` is set to
    ``None``.

    The hook returned by this method always conforms to the signature
    `(state_data: numpy.ndarray, t: float, post_step_data: numpy.ndarray) -> None`
    and is suitable for JIT compilation where supported.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the hook is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            Example field providing the array shape and grid information required
            by the PDE when constructing the post-step hook.
        backend (:class:`~pde.backends.torch.TorchBackend`):
            The specific backend used to create the stepper

    Returns:
        callable or None:
            A compiled function invoking the PDE's post-step hook, or ``None`` if
            no hook is defined.
    """
    # get uncompiled post_step_hook
    post_step_hook: StepperHook | None = None

    if solver._use_post_step_hook:
        try:
            # try to get hook function and initial data from PDE instance
            post_step_hook, solver.info["post_step_data"] = (
                solver.pde.make_post_step_hook(state, backend=backend)
            )
            solver._logger.info("Created post-step hook from PDE")

        except NotImplementedError:
            pass  # no hook function defined on the PDE

    if post_step_hook is None:
        solver.info["post_step_data"] = None
    else:
        # compile post_step_hook
        post_step_hook = backend.compile_function(post_step_hook)
        solver._logger.debug("Compiled post-step hook")

    return post_step_hook


class TorchStepper(torch.nn.Module):
    """Basic single-step integrator module."""

    def single_step(
        self, state_data: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        """Advance the state by one time step.

        Args:
            state_data (:class:`torch.Tensor`):
                The current state in native torch format.
            t (float):
                The current time.
            dt (float):
                The time-step size.

        Returns:
            :class:`torch.Tensor`:
                The updated state after one step.
        """
        raise NotImplementedError

    def single_step_error_estimate(
        self, state_data: torch.Tensor, t: float, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate the local integration error using Richardson extrapolation.

        Args:
            state_data (:class:`torch.Tensor`):
                The current state in native torch format.
            t (float):
                The current time.
            dt (float):
                The time-step size.

        Returns:
            tuple:
                A tuple ``(state, error)`` containing the updated state and a scalar
                absolute error estimate.
        """
        # single step with dt
        k1 = self.single_step(state_data, t, dt)

        # double step with half the time step
        k2a = self.single_step(state_data, t, 0.5 * dt)
        k2 = self.single_step(k2a, t + 0.5 * dt, 0.5 * dt)

        # calculate maximal error
        error = torch.max(torch.abs(k1 - k2))

        return k2, error


class EulerStepper(TorchStepper):
    def __init__(self, solver: EulerSolver, state: TField):
        """Initialize the explicit Euler single-step module.

        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                The solver instance, which determines how the stepper is constructed
            state (:class:`~pde.fields.base.FieldBase`):
                Example field from which grid and other information is read
        """
        super().__init__()
        self.rhs = solver.backend.make_pde_rhs(solver.pde, state)

    def single_step(
        self, state_data: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        """Basic implementation of Euler scheme."""
        # we need to wrap time in a tensor to prevent re-compilation of RHS
        t_device = torch.tensor(t, device=state_data.device)
        return state_data + dt * self.rhs(state_data, t_device)  # type: ignore


class EulerMaruyamaStepper(EulerStepper):
    def __init__(self, solver: EulerSolver, state: TField):
        """Initialize the Euler-Maruyama single-step module."""
        super().__init__(solver, state)
        assert solver.pde.use_noise_variance
        self.noise_drift_factor = solver._noise_drift_factor
        self.has_noise_drift_term = self.noise_drift_factor != 0
        self.noise_var = solver.pde.make_noise_variance(  # type: ignore
            state, backend=solver.backend, ret_diff=self.has_noise_drift_term
        )
        self.gaussian_noise = solver.backend.make_gaussian_noise(
            state, rng=solver.pde.rng
        )

    def single_step(
        self, state_data: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        """Basic implementation of Euler-Maruyama scheme."""
        # we need to wrap time in a tensor to prevent re-compilation of RHS
        t_device = torch.tensor(t, device=state_data.device)

        # evaluate deterministic part and variance without modifying field, yet
        evolution_rate = self.rhs(state_data, t_device)  # type: ignore
        if self.has_noise_drift_term:
            noise_var, noise_var_diff = self.noise_var(state_data, t_device)
        else:
            noise_var = self.noise_var(state_data, t_device)

        # change the state
        scale = torch.sqrt(torch.tensor(dt) * noise_var)
        state_data = state_data + dt * evolution_rate + scale * self.gaussian_noise()

        # add a drift term if the interpretation is not Itô
        if self.has_noise_drift_term:
            state_data += 0.5 * dt * self.noise_drift_factor * noise_var_diff

        return state_data


class EulerMilsteinStepper(EulerStepper):
    def __init__(self, solver: MilsteinSolver, state: TField):
        """Initialize the Euler-Milstein single-step module."""
        super().__init__(solver, state)
        assert solver.pde.use_noise_variance
        self.noise_drift_factor = solver._noise_drift_factor
        self.noise_var = solver.pde.make_noise_variance(  # type: ignore
            state, backend=solver.backend, ret_diff=True
        )
        self.gaussian_noise = solver.backend.make_gaussian_noise(
            state, rng=solver.pde.rng
        )

    def single_step(
        self, state_data: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        """Basic implementation of Euler-Maruyama scheme."""
        # we need to wrap time in a tensor to prevent re-compilation of RHS
        t_device = torch.tensor(t, device=state_data.device)

        # evaluate deterministic part and variance without modifying field, yet
        evolution_rate = self.rhs(state_data, t_device)  # type: ignore
        noise_var, noise_var_diff = self.noise_var(state_data, t_device)

        # change the state
        dW = torch.sqrt(torch.tensor(dt)) * self.gaussian_noise()
        return (  # type: ignore
            state_data
            + dt * evolution_rate
            + 0.5 * dt * self.noise_drift_factor * noise_var_diff
            + torch.sqrt(noise_var) * dW
            + 0.25 * noise_var_diff * (dW**2 - dt)
        )


class FixedSolver(torch.nn.Module):
    def __init__(
        self,
        stepper: TorchStepper,
        dt: float,
        post_step_hook: StepperHook | None,
        post_step_data: Any,
    ):
        """Initialize a fixed-step time integrator.

        Args:
            stepper (:class:`TorchStepper`):
                Compiled single-step module performing individual updates.
            dt (float):
                Constant time-step size.
            post_step_hook (callable):
                Optional hook applied after each accepted step.
            post_step_data (Any):
                Mutable data passed to the post-step hook.
        """
        super().__init__()
        self.stepper = stepper
        self.dt = dt
        self.post_step_hook = post_step_hook
        if post_step_hook is not None:
            self.register_buffer("post_step_data", torch.tensor(post_step_data))

    def forward(
        self, state_data: torch.Tensor, t_start: float, steps: int
    ) -> torch.Tensor:
        """Advance the state for a fixed number of steps.

        Args:
            state_data (:class:`torch.Tensor`):
                The initial state.
            t_start (float):
                The initial time.
            steps (int):
                Number of integration steps to perform.

        Returns:
            :class:`torch.Tensor`:
                The updated state.
        """
        for i in range(steps):
            t = t_start + i * self.dt  # get current time
            # perform single time step
            state_data = self.stepper.single_step(state_data, t, self.dt)
            if self.post_step_hook is not None:  # apply to post-step hook
                state_data, self.post_step_data[...] = self.post_step_hook(  # type: ignore
                    state_data, t, self.post_step_data
                )
        return state_data


def _make_fixed_stepper(solver: SolverBase, state: TField) -> TorchInnerStepperType:
    """Return a backend-level stepping function for fixed time stepping.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            Example field from which grid and other information is read

    Returns:
        callable:
            Function advancing a torch state from ``t_start`` to ``t_end``.
    """
    assert isinstance(solver.backend, TorchBackend)

    dt = float(solver.info["dt"])
    # create subfunctions
    solver.backend.make_pde_rhs(solver.pde, state)
    post_step_hook = _make_post_step_hook(solver, state, backend=solver.backend)

    # get compiled version of a single step
    if solver.pde.is_sde and solver.pde.use_noise_realization:
        msg = "make_noise_realization interface not supported by torch backend"
        raise NotImplementedError(msg)
    if isinstance(solver, MilsteinSolver):
        stepper: TorchStepper = EulerMilsteinStepper(solver, state)
    elif isinstance(solver, EulerSolver):
        if solver.pde.is_sde:
            stepper = EulerMaruyamaStepper(solver, state)
        else:
            stepper = EulerStepper(solver, state)
    else:
        msg = f"Torch backend does not support {solver}"
        raise NotImplementedError(msg)
    stepper = solver.backend.compile_function(stepper)

    # define the executable fixed-step integrator module
    inner_solver = FixedSolver(
        stepper,
        dt,
        post_step_hook=post_step_hook,
        post_step_data=solver.info["post_step_data"],
    ).to(solver.backend.device)

    def fixed_stepper(
        state_data: torch.Tensor, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, float]:
        """Advance `state` from `t_start` to `t_end` using fixed steps."""
        # calculate number of steps that lead to an end time closest to t_end
        steps = max(1, round((t_end - t_start) / dt))
        # execute the fixed-step integrator module
        state_data = inner_solver(state_data, t_start, steps)
        solver.info["steps"] += steps
        if post_step_hook is not None:
            solver.info["post_step_data"] = inner_solver.post_step_data.cpu()
        return state_data, t_start + dt * steps

    return fixed_stepper


class TorchAdaptiveSolverBase(torch.nn.Module):
    """Base module for adaptive time integrator."""

    atol = 1e-6

    def __init__(
        self,
        dt_init: float,
        post_step_hook: StepperHook | None,
        post_step_data: Any,
        *,
        tolerance: float,
        dt_min: float,
        dt_max: float,
    ):
        """Initialize an adaptive time integrator.

        Args:
            dt_init (float):
                Initial time-step size.
            post_step_hook (callable):
                Optional hook applied after each accepted step.
            post_step_data (Any):
                Mutable data passed to the post-step hook.
            tolerance (float):
                Target error tolerance for adaptive stepping.
            dt_min (float):
                Minimum allowed time-step size.
            dt_max (float):
                Maximum allowed time-step size.
        """
        super().__init__()
        self.dt = dt_init
        self.post_step_hook = post_step_hook
        if post_step_hook is not None:
            self.register_buffer("post_step_data", torch.tensor(post_step_data))

        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tolerance = tolerance
        self.steps = 0
        self.dt_stats = OnlineStatistics()

        self._dt_min_nan_err = f"Encountered NaN even though dt < {dt_min}"
        self._dt_min_err = f"Time step below {dt_min}"

    def adjust_dt(self, dt: float, error_rel: torch.Tensor) -> float:
        """Adjust the next time-step size.

        The goal is to keep the relative error `error_rel` close to 1.

        Args:
            dt (float):
                Current time step
            error_rel (:class:`torch.Tensor`):
                Current (normalized) error estimate

        Returns:
            float: Time step of the next iteration
        """
        # adjust the time step
        if error_rel < 0.00057665:
            # error was very small => maximal increase in dt
            # The constant on the right hand side of the comparison is chosen to
            # agree with the equation for adjusting dt below
            dt *= 4.0
        elif torch.isnan(error_rel):
            # state contained NaN => decrease time step strongly
            dt *= 0.25
        else:
            # otherwise, adjust time step according to error
            dt_factor = 0.9 * error_rel.item() ** -0.2
            if dt_factor < 0.1:
                dt_factor = 0.1
            dt *= dt_factor  # torch.clamp(dt_factor, 0.1, 10)

        # limit time step to permissible bracket
        if dt > self.dt_max:
            dt = self.dt_max
        elif dt < self.dt_min:
            if torch.isnan(error_rel):
                raise RuntimeError(self._dt_min_nan_err)
            raise RuntimeError(self._dt_min_err)

        return dt

    def forward(
        self, state_data: torch.Tensor, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, float]:
        """Advance the state from ``t_start`` to ``t_end`` adaptively.

        Args:
            state_data (:class:`torch.Tensor`):
                The initial state.
            t_start (float):
                The initial time.
            t_end (float):
                The final time.

        Returns:
            tuple:
                A tuple ``(state, t)`` containing the final state and the reached
                final time.
        """
        raise NotImplementedError


class TorchAdaptiveGeneralSolver(TorchAdaptiveSolverBase):
    """General adaptive time integrator."""

    def __init__(
        self,
        stepper: TorchStepper,
        dt_init: float,
        post_step_hook: StepperHook | None,
        post_step_data: Any,
        *,
        tolerance: float,
        dt_min: float,
        dt_max: float,
    ):
        """Initialize an adaptive time integrator.

        Args:
            stepper (:class:`TorchStepper`):
                Compiled single-step module performing individual updates.
            dt_init (float):
                Initial time-step size.
            post_step_hook (callable):
                Optional hook applied after each accepted step.
            post_step_data (Any):
                Mutable data passed to the post-step hook.
            tolerance (float):
                Target error tolerance for adaptive stepping.
            dt_min (float):
                Minimum allowed time-step size.
            dt_max (float):
                Maximum allowed time-step size.
        """
        super().__init__(
            dt_init,
            post_step_hook,
            post_step_data,
            tolerance=tolerance,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.stepper = stepper

    def forward(
        self, state_data: torch.Tensor, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, float]:
        """Advance the state from ``t_start`` to ``t_end`` adaptively.

        Args:
            state_data (:class:`torch.Tensor`):
                The initial state.
            t_start (float):
                The initial time.
            t_end (float):
                The final time.

        Returns:
            tuple:
                A tuple ``(state, t)`` containing the final state and the reached
                final time.
        """
        t = t_start
        while True:
            # use a smaller (but not too small) time step if close to t_end
            if self.dt < self.dt_min:
                dt_step = self.dt_min
            elif self.dt > t_end - t + self.atol:
                dt_step = t_end - t + self.atol
            else:
                dt_step = self.dt

            # try two different step sizes to estimate errors
            new_state, error = self.stepper.single_step_error_estimate(
                state_data, t, dt_step
            )

            error_rel = error / self.tolerance  # normalize error to given tolerance

            # do the step if the error is sufficiently small
            if error_rel <= 1:
                self.steps += 1
                t = t + dt_step

                if self.post_step_hook is not None:  # apply to post-step hook
                    state_data, self.post_step_data[...] = self.post_step_hook(  # type: ignore
                        new_state, t, self.post_step_data
                    )
                else:
                    state_data = new_state  # simply set new state to current state

                self.dt_stats.add(dt_step)

            if t < t_end:
                # adjust the time step and continue (happens in every MPI process)
                self.dt = self.adjust_dt(dt_step, error_rel)
            else:
                break  # return to the controller

        return state_data, t


class TorchAdaptiveEulerSolver(TorchAdaptiveSolverBase):
    """Adaptive time integrator optimized for the explicit Euler scheme."""

    def __init__(
        self,
        rhs: TorchRHSType,
        dt_init: float,
        post_step_hook: StepperHook | None,
        post_step_data: Any,
        *,
        tolerance: float,
        dt_min: float,
        dt_max: float,
    ):
        """Initialize an adaptive Euler integrator.

        Args:
            rhs (callable):
                Function evaluating the deterministic time derivative.
            dt_init (float):
                Initial time-step size.
            post_step_hook (callable):
                Optional hook applied after each accepted step.
            post_step_data (Any):
                Mutable data passed to the post-step hook.
            tolerance (float):
                Target error tolerance for adaptive stepping.
            dt_min (float):
                Minimum allowed time-step size.
            dt_max (float):
                Maximum allowed time-step size.
        """
        super().__init__(
            dt_init,
            post_step_hook,
            post_step_data,
            tolerance=tolerance,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.rhs = rhs

    def _eval_rhs(self, state_data: torch.Tensor, t: float) -> torch.Tensor:
        """Evaluate the right-hand side with time on the correct device."""
        t_device = torch.tensor(t, device=state_data.device)
        return self.rhs(state_data, t_device)

    def forward(
        self, state_data: torch.Tensor, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, float]:
        """Advance the state from ``t_start`` to ``t_end`` adaptively."""
        state_cur = state_data
        rate = self._eval_rhs(state_cur, t_start)

        t = t_start
        while True:
            # use a smaller (but not too small) time step if close to t_end
            if self.dt < self.dt_min:
                dt_step = self.dt_min
            elif self.dt > t_end - t + self.atol:
                dt_step = t_end - t + self.atol
            else:
                dt_step = self.dt

            # do single step with dt
            step_large = state_cur + dt_step * rate
            # do double step with half the time step
            step_small = state_cur + 0.5 * dt_step * rate

            try:
                # calculate rate at the midpoint of the double step
                rate_midpoint = self._eval_rhs(step_small, t + 0.5 * dt_step)
            except Exception:
                # an exception likely signals that rate could not be calculated
                error_rel = torch.tensor(float("nan"), device=state_cur.device)
            else:
                # advance to end of double step
                step_small = step_small + 0.5 * dt_step * rate_midpoint

                # calculate maximal error
                error = torch.max(torch.abs(step_large - step_small))
                error_rel = error / self.tolerance

            if error_rel <= 1:
                try:
                    # calculate the rate at the putative new step
                    rate = self._eval_rhs(step_small, t)
                except Exception:
                    # calculating the rate failed => retry with smaller dt
                    error_rel = torch.tensor(float("nan"), device=state_cur.device)
                else:
                    self.steps += 1
                    t += dt_step

                    if self.post_step_hook is not None:
                        state_cur, self.post_step_data[...] = self.post_step_hook(  # type: ignore
                            step_small, t, self.post_step_data
                        )
                    else:
                        state_cur = step_small

                    self.dt_stats.add(dt_step)

            if t < t_end:
                self.dt = self.adjust_dt(dt_step, error_rel)
            else:
                break

        return state_cur, t


def _make_adaptive_stepper_general(
    solver: AdaptiveSolverBase, state: TField
) -> TorchInnerStepperType:
    """Return a backend-level stepping function for adaptive time stepping.

    Args:
        solver (:class:`~pde.solvers.base.AdaptiveSolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            Example field from which grid and other information is read

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: torch.Tensor,
        t_start: float, t_end: float)`
    """
    assert isinstance(solver.backend, TorchBackend)

    if solver.pde.is_sde:
        msg = "Adaptive stepping does not support stochastic equations"
        raise RuntimeError(msg)

    # create subfunctions
    rhs = solver.backend.make_pde_rhs(solver.pde, state)
    rhs = solver.backend.compile_function(rhs)
    post_step_hook = _make_post_step_hook(solver, state, backend=solver.backend)

    if isinstance(solver, EulerSolver):
        # define an optimized integrator for the Euler single-step module
        inner_solver: TorchAdaptiveSolverBase = TorchAdaptiveEulerSolver(
            rhs,
            dt_init=float(solver.info["dt"]),
            post_step_hook=post_step_hook,
            post_step_data=solver.info["post_step_data"],
            tolerance=float(solver.tolerance),
            dt_min=float(solver.dt_min),
            dt_max=float(solver.dt_max),
        )
    else:
        # get a compiled single-step module
        msg = f"Solver {solver} is not supported by torch backend."
        raise NotImplementedError(msg)
        # stepper.to(solver.backend.device)

        # # define the adaptive integrator module
        # inner_solver = TorchAdaptiveGeneralSolver(
        #     stepper,
        #     dt_init=float(solver.info["dt"]),
        #     post_step_hook=post_step_hook,
        #     post_step_data=solver.info["post_step_data"],
        #     tolerance=float(solver.tolerance),
        #     dt_min=float(solver.dt_min),
        #     dt_max=float(solver.dt_max),
        # )
    inner_solver.to(solver.backend.device)

    # add extra information
    solver.info["dt_adaptive"] = solver.adaptive
    solver.info["dt_statistics"] = inner_solver.dt_stats

    def adaptive_stepper(
        state_data: torch.Tensor, t_start: float, t_end: float
    ) -> tuple[torch.Tensor, float]:
        """Advance `state` from `t_start` to `t_end` using adaptive steps."""
        state_data, t_final = inner_solver(state_data, t_start, t_end)
        # save some data for the solver
        solver.info["dt"] = inner_solver.dt
        solver.info["steps"] = inner_solver.steps
        if post_step_hook is not None:
            solver.info["post_step_data"] = inner_solver.post_step_data.cpu()
        return state_data, t_final

    return adaptive_stepper


def make_inner_stepper(solver: SolverBase, state: TField) -> TorchInnerStepperType:
    """Return the backend-level stepping function for the torch backend.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            Example field from which grid and other information is read

    Returns:
        callable:
            Function that can be called to advance the state from ``t_start`` to
            ``t_end``.
    """
    # get the backend-level stepping function
    if isinstance(solver, AdaptiveSolverBase) and solver.adaptive:
        return _make_adaptive_stepper_general(solver, state)

    # dealing with a solver configured for fixed time stepping
    if isinstance(solver, AdamsBashforthSolver):
        raise NotImplementedError
    return _make_fixed_stepper(solver, state)
