"""
Defines a PDE class whose right hand side is given as a string

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Tuple

import numba as nb
import numpy as np
from numba.typed import Dict as NumbaDict
from sympy import Symbol
from sympy.core.function import UndefinedFunction

from ..fields import FieldCollection, VectorField
from ..fields.base import DataFieldBase, FieldBase
from ..grids.boundaries.axes import BoundariesData
from ..grids.boundaries.local import BCDataError
from ..pdes.base import PDEBase
from ..tools.docstrings import fill_in_docstring
from ..tools.numba import jit
from ..tools.typing import ArrayLike, NumberOrArray

# Define short notations that can appear in mathematical equations and need to be
# expanded. Since these replacements are replaced in order, it's advisable to start with
# more complex expressions first
_EXPRESSION_REPLACEMENT: Dict[str, str] = {
    r"\|\s*∇\s*(\w+)\s*\|(²|\*\*2)": r"gradient_squared(\1)",  # |∇c|² or |∇c|**2
    r"∇(²|\*\*2)\s*(\w+)": r"laplace(\2)",  # ∇²c or ∇**2 c
    r"∇(²|\*\*2)\s*\(": r"laplace(",  # ∇²(c) or ∇**2(c)
    r"²": r"**2",
    r"³": r"**3",
}


class PDE(PDEBase):
    """PDE defined by mathematical expressions

    Attributes:
        variables (tuple):
            The name of the variables (i.e., fields) in the order they are expected to
            appear in the `state`.
    """

    @fill_in_docstring
    def __init__(
        self,
        rhs: Dict[str, str],
        *,
        noise: ArrayLike = 0,
        bc: BoundariesData = "auto_periodic_neumann",
        bc_ops: Dict[str, BoundariesData] = None,
        user_funcs: Dict[str, Callable] = None,
        consts: Dict[str, NumberOrArray] = None,
    ):
        r"""
        Warning:
            {WARNING_EXEC}

        Args:
            rhs (dict):
                The expressions defining the evolution rate. The dictionary keys define
                the name of the fields whose evolution is considered, while the values
                specify their evolution rate as a string that can be parsed by
                :mod:`sympy`. These expression may contain variables (i.e., the fields
                themselves, spatial coordinates of the grid, and `t` for the time),
                standard local mathematical operators defined by sympy, and the
                operators defined in the :mod:`pde` package. Note that operators need to
                be specified with their full name, i.e., `laplace` for a scalar
                Laplacian and `vector_laplace` for a Laplacian operating on a vector
                field. Moreover, the dot product between two vector fields can be
                denoted by using `dot(field1, field2)` in the expression, an outer
                product is calculated using `outer(field1, field2)`, and
                `integral(field)` denotes an integral over a field.
                More information can be found in the
                :ref:`expression documentation <documentation-expressions>`.
            noise (float or :class:`~numpy.ndarray`):
                Magnitude of additive Gaussian white noise. The default value of zero
                implies deterministic partial differential equations will be solved.
                Different noise magnitudes can be supplied for each field in coupled
                PDEs by either specifying a sequence of numbers or a dictionary with
                values for each field.
            bc:
                Boundary conditions for the operators used in the expression. The
                conditions here are applied to all operators that do not have a
                specialized condition given in `bc_ops`.
                {ARG_BOUNDARIES}
            bc_ops (dict):
                Special boundary conditions for some operators. The keys in this
                dictionary specify where the boundary condition will be applied.
                The keys follow the format "VARIABLE:OPERATOR", where VARIABLE specifies
                the expression in `rhs` where the boundary condition is applied to the
                operator specified by OPERATOR. For both identifiers, the wildcard
                symbol "\*" denotes that all fields and operators are affected,
                respectively. For instance, the identifier "c:\*" allows specifying a
                condition for all operators of the field named `c`.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expressions in `rhs`.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. These can be either scalar numbers or fields defined on the
                same grid as the actual simulation.

        Note:
            The order in which the fields are given in `rhs` defines the order in which
            they need to appear in the `state` variable when the evolution rate is
            calculated. Note that `dict` keep the insertion order since Python version
            3.7, so a normal dictionary can be used to define the equations.
        """
        from sympy.core.function import AppliedUndef

        from ..tools.expressions import ScalarExpression

        # parse noise strength
        if isinstance(noise, dict):
            noise = [noise.get(var, 0) for var in rhs]
        if hasattr(noise, "__iter__") and len(noise) != len(rhs):  # type: ignore
            raise ValueError("Number of noise strengths does not match field count")

        super().__init__(noise=noise)

        # validate input
        if not isinstance(rhs, dict):
            rhs = dict(rhs)
        if "t" in rhs:
            raise ValueError("Cannot name field `t` since it denotes time")
        if consts is None:
            consts = {}

        # turn the expression strings into sympy expressions
        self._rhs_expr, self._operators = {}, {}
        explicit_time_dependence = False
        complex_valued = False
        for var, rhs_item in rhs.items():
            # replace shorthand operators
            if isinstance(rhs_item, str):
                rhs_item_old = rhs_item
                for search, repl in _EXPRESSION_REPLACEMENT.items():
                    rhs_item = re.sub(search, repl, rhs_item)
                if rhs_item != rhs_item_old:
                    self._logger.info("Transformed expression to `%s`", rhs_item)

            # create placeholder dictionary of constants that will be specified later
            consts_d: Dict[str, NumberOrArray] = {name: 0 for name in consts}
            rhs_expr = ScalarExpression(
                rhs_item,
                user_funcs=user_funcs,
                consts=consts_d,
                explicit_symbols=rhs.keys(),  # type: ignore
            )

            if rhs_expr.depends_on("t"):
                explicit_time_dependence = True
            if rhs_expr.complex:
                complex_valued = True

            # determine undefined functions in the expression
            self._operators[var] = {
                func.__class__.__name__
                for func in rhs_expr._sympy_expr.atoms(AppliedUndef)
                if func.__class__.__name__ not in rhs_expr.user_funcs
            }

            self._rhs_expr[var] = rhs_expr

        # set public instance attributes
        self.rhs = rhs
        self.variables = tuple(rhs.keys())
        self.consts = consts
        self.explicit_time_dependence = explicit_time_dependence
        self.complex_valued = complex_valued
        operators = frozenset().union(*self._operators.values())

        # setup boundary conditions
        if bc_ops is None:
            bcs = {"*:*": bc}
        else:
            bcs = dict(bc_ops)
            if "*:*" in bcs and bc != "auto_periodic_neumann":
                self._logger.warning("Found default BCs in `bcs` and `bc_ops`")
            bcs["*:*"] = bc  # append default boundary conditions

        self.bcs: Dict[str, Any] = {}
        for key_str, value in bcs.items():
            # split on . and :
            parts = re.split(r"\.|:", key_str)
            if len(parts) == 1:
                if len(self.variables):
                    key = f"{self.variables[0]}:{key_str}"
                else:
                    raise ValueError(
                        f'Boundary condition "{key_str}" is ambiguous. Use format '
                        '"VARIABLE:OPERATOR" instead.'
                    )
            elif len(parts) == 2:
                key = ":".join(parts)
            else:
                raise ValueError(f'Cannot parse boundary condition "{key_str}"')
            if key in self.bcs:
                self._logger.warning(f"Two boundary conditions for key {key}")
            self.bcs[key] = value

        # save information for easy inspection
        self.diagnostics["definition"] = {
            "variables": self.variables,
            "constants": tuple(self.consts),
            "explicit_time_dependence": explicit_time_dependence,
            "complex_valued_rhs": complex_valued,
            "operators": operators,
        }
        self._cache: Dict[str, Dict[str, Any]] = {}

    @property
    def expressions(self) -> Dict[str, str]:
        """show the expressions of the PDE"""
        return {k: v.expression for k, v in self._rhs_expr.items()}

    def _compile_rhs_single(
        self,
        var: str,
        ops: Dict[str, Callable],
        state: FieldBase,
        backend: str = "numpy",
    ):
        """compile a function determining the right hand side for one variable

        Args:
            var (str):
                The variable that is considered
            ops (dict):
                A dictionary of operators that can be used by this function. Note that
                this dictionary might be modified in place
            state (:class:`~pde.fields.FieldBase`):
                The field describing the state of the PDE
            backend (str):
                The backend for which the data is prepared

        Returns:
            callable: The function calculating the RHS
        """
        # modify a copy of the expression and the general operator array
        expr = self._rhs_expr[var].copy()

        # obtain the (differential) operators for this variable
        for func in self._operators[var]:
            if func in ops:
                continue

            # determine boundary conditions for this operator and variable
            for bc_key, bc in self.bcs.items():
                bc_var, bc_func = bc_key.split(":")
                var_match = bc_var == var or bc_var == "*"
                func_match = bc_func == func or bc_func == "*"
                if var_match and func_match:
                    break  # found a matching boundary condition
            else:
                raise RuntimeError(
                    "Could not find suitable boundary condition for function "
                    f"`{func}` applied in equation for `{var}`"
                )

            # Tell the user what BC we chose for a given operator
            msg = "Using boundary condition `%s` for operator `%s` in PDE for `%s`"
            self._logger.info(msg, bc, func, var)

            # create the function evaluating the operator
            try:
                ops[func] = state.grid.make_operator(func, bc=bc)
            except BCDataError:
                # wrong data was supplied for the boundary condition
                raise
            except ValueError:
                # any other exception should signal that the operator is not
                # defined, so we (almost) silently assume that sympy defines the
                # operator
                self._logger.info(
                    "Assuming that sympy knows undefined operator `%s`", func
                )

            # add `bc_args` as an argument to the call of the operators to be able
            # to pass additional information, like time
            expr._sympy_expr = expr._sympy_expr.replace(
                # only modify the relevant operator
                lambda expr: isinstance(expr.func, UndefinedFunction)
                and expr.name == func
                # and do not modify it when the bc_args have already been set
                and not (
                    isinstance(expr.args[-1], Symbol)
                    and expr.args[-1].name == "bc_args"
                ),
                # otherwise, add None and bc_args as arguments
                lambda expr: expr.func(*expr.args, Symbol("none"), Symbol("bc_args")),
            )

        # obtain the function to calculate the right hand side
        signature = self.variables + ("t", "none", "bc_args")

        # check whether this function depends on additional input
        if any(expr.depends_on(c) for c in state.grid.axes):
            # expression has a spatial dependence, too

            # extend the signature
            signature += tuple(state.grid.axes)
            # inject the spatial coordinates into the expression for the rhs
            extra_args = tuple(  # @UnusedVariable
                state.grid.cell_coords[..., i] for i in range(state.grid.num_axes)
            )

        else:
            # expression only depends on the actual variables
            extra_args = tuple()  # @UnusedVariable

        # check whether all variables are accounted for
        extra_vars = set(expr.vars) - set(signature)
        if extra_vars:
            extra_vars_str = ", ".join(sorted(extra_vars))
            raise RuntimeError(f"Undefined variable in expression: {extra_vars_str}")
        expr.vars = signature

        self._logger.info("RHS for `%s` has signature %s", var, signature)

        # prepare the actual function being called in the end
        if backend == "numpy":
            func_inner = expr._get_function(single_arg=False, user_funcs=ops)
        elif backend == "numba":
            func_pure = expr._get_function(
                single_arg=False, user_funcs=ops, prepare_compilation=True
            )
            func_inner = jit(func_pure)
        else:
            raise ValueError(f"Unsupported backend {backend}")

        def rhs_func(*args) -> np.ndarray:
            """wrapper that inserts the extra arguments and initialized bc_args"""
            bc_args = NumbaDict()  # args for differential operators
            bc_args["t"] = args[-1]  # pass time to differential operators
            return func_inner(*args, None, bc_args, *extra_args)  # type: ignore

        return rhs_func

    def _prepare_cache(
        self, state: FieldBase, backend: str = "numpy"
    ) -> Dict[str, Any]:
        """prepare the expression by setting internal variables in the cache

        Note that the expensive calculations in this method are only carried out if the
        state attributes change.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                The field describing the state of the PDE
            backend (str):
                The backend for which the data is prepared

        Returns:
            dict: A dictionary with information that can be reused
        """
        # check the cache
        cache = self._cache.get(backend, {})
        if state.attributes == cache.get("state_attributes", None):
            return cache  # this cache was already prepared
        cache = self._cache[backend] = {}  # clear cache, if there was any

        # check whether PDE has variables with same names as grid axes
        name_overlap = set(self.rhs) & set(state.grid.axes)
        if name_overlap:
            raise ValueError(f"Coordinate {name_overlap} cannot be used as field name")

        # check whether the state is compatible with the PDE
        num_fields = len(self.variables)
        self.diagnostics["definition"]["num_fields"] = num_fields
        if isinstance(state, FieldCollection):
            if num_fields != len(state):
                raise ValueError(
                    f"Expected {num_fields} fields in state, but got {len(state)} ones"
                )
        elif isinstance(state, DataFieldBase):
            if num_fields != 1:
                raise ValueError(
                    f"Expected {num_fields} fields in state, but got only one"
                )
        else:
            raise ValueError(f"Unknown state class {state.__class__.__name__}")

        # check compatibility of constants and update the rhs accordingly
        for name, value in self.consts.items():
            # check whether the constant has a supported value
            if np.isscalar(value):
                pass  # this simple case is fine
            elif isinstance(value, DataFieldBase):
                value.grid.assert_grid_compatible(state.grid)
                value = value.data  # just keep the actual discretized data
            else:
                raise TypeError(f"Constant has unsupported type {value.__class__}")

            for rhs in self._rhs_expr.values():
                rhs.consts[name] = value  # type: ignore

        # obtain functions used in the expression
        ops_general: Dict[str, Callable] = {}

        # create special operators if necessary
        operators = self.diagnostics["definition"]["operators"]
        if "dot" in operators:
            # add dot product between two vector fields. This can for instance
            # appear when two gradients of scalar fields need to be multiplied
            ops_general["dot"] = VectorField(state.grid).make_dot_operator(backend)

        if "inner" in operators:
            # inner is a synonym for dot product operator
            ops_general["inner"] = VectorField(state.grid).make_dot_operator(backend)

        if "outer" in operators:
            # generate an operator that calculates an outer product
            vec_field = VectorField(state.grid)
            ops_general["outer"] = vec_field.make_outer_prod_operator(backend)

        if "integral" in operators:
            # add an operator that integrates a field
            ops_general["integral"] = state.grid.make_integrator()

        # Create the right hand sides for all variables. It is important to do this in a
        # separate function, so the closures work reliably
        cache["rhs_funcs"] = [
            self._compile_rhs_single(var, ops_general.copy(), state, backend)
            for var in self.variables
        ]

        # add extra information for field collection
        if isinstance(state, FieldCollection):
            # isscalar be False even if start == stop (e.g. vector fields)
            isscalar = tuple(field.rank == 0 for field in state)
            starts = tuple(slc.start for slc in state._slices)
            stops = tuple(slc.stop for slc in state._slices)

            def get_data_tuple(state_data: np.ndarray) -> Tuple[np.ndarray, ...]:
                """helper for turning state_data into a tuple of field data"""
                return tuple(
                    state_data[starts[i]]
                    if isscalar[i]
                    else state_data[starts[i] : stops[i]]
                    for i in range(num_fields)
                )

            cache["get_data_tuple"] = get_data_tuple

        # store the attributes in the cache, which allows to later circumvent
        # calculating the quantities above again. Note that this has to be the
        # last expression of the method, so the cache is only valid when the
        # prepare function worked successfully
        cache["state_attributes"] = state.attributes
        return cache

    def evolution_rate(self, state: FieldBase, t: float = 0.0) -> FieldBase:
        """evaluate the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                The field describing the state of the PDE
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldBase`:
            Field describing the evolution rate of the PDE
        """
        cache = self._prepare_cache(state, backend="numpy")

        # create an empty copy of the current field
        result = state.copy()

        # fill it with data
        if isinstance(state, DataFieldBase):
            # state is a single field
            result.data[:] = cache["rhs_funcs"][0](state.data, t)

        elif isinstance(state, FieldCollection):
            # state is a collection of fields
            for i in range(len(state)):
                data_tpl = cache["get_data_tuple"](state.data)
                result[i].data[:] = cache["rhs_funcs"][i](*data_tpl, t)  # type: ignore

        else:
            raise TypeError(f"Unsupported field {state.__class__.__name__}")

        return result

    def _make_pde_rhs_numba_coll(
        self, state: FieldCollection, cache: Dict[str, Any]
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create the compiled rhs if `state` is a field collection

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
            cache (dict):
                Cached information that will be used in the function. The cache is
                populated by :meth:`PDE._prepare_cache`.

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`~numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`~numpy.ndarray` giving
            the evolution rate.
        """
        num_fields = len(state)
        data_shape = state.data.shape
        rhs_list = tuple(jit(cache["rhs_funcs"][i]) for i in range(num_fields))

        starts = tuple(slc.start for slc in state._slices)
        stops = tuple(slc.stop for slc in state._slices)

        # In the future, the following should be possible:
        #         @jit
        #         def evolution_rate(state_data, t, out):
        #             """ evolve all agents explicitly """
        #             for i in nb.literal_unroll(range(num_fields)):
        #                 out[i] = rhs_list[i](*state_data, t)
        #         return evolver

        get_data_tuple = cache["get_data_tuple"]

        def chain(
            i: int = 0, inner: Callable[[np.ndarray, float, np.ndarray], None] = None
        ) -> Callable[[np.ndarray, float], np.ndarray]:
            """recursive helper function for applying all rhs"""
            # run through all functions
            rhs = rhs_list[i]

            if inner is None:
                # the innermost function does not need to call a child
                @jit
                def wrap(data_tpl: np.ndarray, t: float, out: np.ndarray):
                    out[starts[i] : stops[i]] = rhs(*data_tpl, t)

            else:
                # all other functions need to call one deeper in the chain
                @jit
                def wrap(data_tpl: np.ndarray, t: float, out: np.ndarray):
                    inner(data_tpl, t, out)  # type: ignore
                    out[starts[i] : stops[i]] = rhs(*data_tpl, t)

            if i < num_fields - 1:
                # there are more items in the chain
                return chain(i + 1, inner=wrap)
            else:
                # this is the outermost function
                @jit
                def evolution_rate(state_data: np.ndarray, t: float = 0) -> np.ndarray:
                    out = np.empty(data_shape)
                    with nb.objmode():
                        data_tpl = get_data_tuple(state_data)
                        wrap(data_tpl, t, out)
                    return out

                return evolution_rate  # type: ignore

        # compile the recursive chain
        return chain()

    def _make_pde_rhs_numba(
        self, state: FieldBase, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create a compiled function evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called with an
            instance of :class:`~numpy.ndarray` of the state data and the time to
            obtained an instance of :class:`~numpy.ndarray` giving the evolution rate.
        """
        cache = self._prepare_cache(state, backend="numba")

        if isinstance(state, DataFieldBase):
            # state is a single field
            return jit(cache["rhs_funcs"][0])  # type: ignore

        elif isinstance(state, FieldCollection):
            # state is a collection of fields
            return self._make_pde_rhs_numba_coll(state, cache)

        else:
            raise TypeError(f"Unsupported field {state.__class__.__name__}")
