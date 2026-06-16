"""
Defines a PDE class implementing a scalar `reaction diffusion equation
<https://en.wikipedia.org/wiki/Reaction-diffusion_system>`_.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .pde import PDE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..grids.boundaries.axes import BoundariesData
    from ..tools.typing import ArrayLike, NumberOrArray, PostStepHook


class ReactionDiffusionPDE(PDE):
    r"""Reaction-diffusion equation

    The equation being solved reads

    .. math::
        \partial_t c_i = D_i \partial_\alpha^2 c_i + s_i(\{c_j\}, t)

    where `c_i` are the concentration fields, :math:`D_{ij}` is the diffusivity matrix,
    and :math:`s_i` are sink/source terms that account for chemical reactions.
    """

    def __init__(
        self,
        variables: Sequence[str],
        diffusivity: ArrayLike,
        sources: Sequence[str] | dict[str, str],
        *,
        bc: BoundariesData | None = None,
        bc_ops: dict[str, BoundariesData] | None = None,
        post_step_hook: PostStepHook | None = None,
        user_funcs: dict[str, Callable] | None = None,
        consts: dict[str, NumberOrArray] | None = None,
        noise: ArrayLike | dict[str, NumberOrArray] = 0,
        rng: np.random.Generator | None = None,
    ):
        r"""
        Args:
            variables (list of strings):
                The names and order of the variables in the system
            diffusivity (:class:`~numpy.ndarray`):
                Diffusivities of all species. A scalar sets the same diffusivity for all
                species. A vector is supported for setting different diffusivities per
                species.
            sources (list of str or dict of str):
                Specifies the source terms of each species. Must be a list with an entry
                for each variable. Alternatively, a dictionary may be used to only
                specify the source term of a few variables, while all others are
                assumed to not have any sources.
            bc:
                General boundary conditions for all operators that do not have a
                specialized condition given in `bc_ops`.
                {ARG_BOUNDARIES}
            bc_ops (dict):
                Special boundary conditions for specific operators. The keys in this
                dictionary specify where the boundary condition will be applied.
                The keys follow the format "VARIABLE:OPERATOR", where VARIABLE specifies
                the expression in `rhs` where the boundary condition is applied to the
                operator specified by OPERATOR. For both identifiers, the wildcard
                symbol "\*" denotes that all fields and operators are affected,
                respectively. For instance, the identifier "c:\*" allows specifying a
                condition for all operators of the field named `c`.
            post_step_hook (callable):
                A function with signature `(state_data, t)` that will be called after
                every time step. The function must return `state_data`, which can be
                modified in place. The hook can also abort the simulation immediately by
                raising `StopIteration` (might not work with all backends).
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expressions in `rhs`.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. These can be either scalar numbers or fields defined on the
                same grid as the actual simulation.
            noise (float, :class:`~numpy.ndarray`, or dict):
                Variance of additive Gaussian white noise. The default value of zero
                implies deterministic partial differential equations will be solved.
                Different noise magnitudes can be supplied for each field in coupled
                PDEs by either specifying a sequence of numbers or a dictionary with
                values for each field.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
                used for stochastic simulations. Note that this random number generator
                is only used for numpy functions, while compiled numba code uses the
                random number generator of numba. Moreover, in simulations using
                multiprocessing, setting the same generator in all processes might yield
                unintended correlations in the simulation results.
        """
        # define all the variables
        if len(variables) != len(set(variables)):
            msg = "Variable names are not unique"
            raise ValueError(msg)
        variables = list(variables)
        dim = len(variables)
        diffusivity = np.broadcast_to(diffusivity, (dim,))

        # prepare sources dictionary
        if isinstance(sources, dict):
            sources = sources.copy()  # we want to modify the dict below
        elif len(sources) != dim:
            msg = "Length mismatch between `variables` and `sources`"
            raise ValueError(msg)
        else:
            sources = dict(zip(variables, sources, strict=True))

        # prepare the right hand side
        rhs = {
            v: f"{d} * laplace({v}) + {sources.pop(v, '0')}"
            for v, d in zip(variables, diffusivity, strict=True)
        }
        if sources:
            self._logger.warning("Unused `sources` entries: %s", sources)

        # initialize the PDE class
        super().__init__(
            rhs=rhs,
            bc=bc,
            bc_ops=bc_ops,
            post_step_hook=post_step_hook,
            user_funcs=user_funcs,
            consts=consts,
            noise=noise,
            rng=rng,
        )
