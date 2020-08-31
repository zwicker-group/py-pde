"""
Defines a PDE class whose right hand side is given as a string

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

import re
from collections import OrderedDict
from typing import Any, Callable, Dict  # @UnusedImport

import numpy as np
from pde.fields import FieldCollection, VectorField
from pde.fields.base import DataFieldBase, FieldBase, OptionalArrayLike
from pde.grids.boundaries.axes import BoundariesData
from pde.pdes.base import PDEBase
from pde.tools.docstrings import fill_in_docstring
from pde.tools.numba import jit, nb


class PDE(PDEBase):
    """PDE defined by a mathematical expression

    Attributes:
        variables (tuple):
            The name of the variables (i.e., fields) in the order they are
            expected to appear in the `state`.
        diagnostics (dict):
            Additional diagnostic information that might help with analyzing
            problems, e.g., when :mod:`sympy` cannot parse or :mod`numba` cannot
            compile a function.
    """

    @fill_in_docstring
    def __init__(
        self,
        rhs: "OrderedDict[str, str]",
        noise: OptionalArrayLike = 0,
        bc: BoundariesData = "natural",
        bc_ops: "OrderedDict[str, BoundariesData]" = None,
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            rhs (OrderedDict):
                The expressions defining the evolution rate. The dictionary keys
                define the name of the fields whose evolution is considered,
                while the values specify their evolution rate as a string that
                can be parsed by :mod:`sympy`. These expression may contain the
                fields themselves, standard local mathematical operators defined
                by sympy, and the operators defined in the :mod:`pde` package.
                Note that operators need to be specified with their full name,
                i.e., `laplace` for a scalar Laplacian and `vector_laplace` for
                a Laplacian operating on a vector field. Moreover, the dot
                product between two vector fields can be denoted by using
                `dot(field1, field2)` in the expression.
            noise (float or :class:`numpy.ndarray`):
                Magnitude of additive Gaussian white noise. The default value of
                zero implies deterministic partial differential equations will
                be solved. Different noise magnitudes can be supplied for each
                field in coupled PDEs.
            bc:
                Boundary conditions for the operators used in the expression.
                The conditions here are applied to all operators that do not
                have a specialized condition given in `bc_ops`
                {ARG_BOUNDARIES}
            bc_ops (dict):
                Special boundary conditions for some operators. The keys in this
                dictionary specify where the boundary condition will be applied.
                The keys follow the format "FIELD:OPERATOR", where FIELD is the
                name of a field and OPERATOR denotes the name of an operator.
                For both identifiers, the wildcard symbol "*" denotes that all
                fields and operators are affected, respectively.

        Note:
            The order in which the fields are given in `rhs` defines the order
            in which they need to appear in the `state` variable when the
            evolution rate is calculated. Note that the insertion order of
            `dict` is guaranteed since Python version 3.7, so a normal
            dictionary can be used to define the equations.
        """
        from sympy.core.function import AppliedUndef

        from ..tools.expressions import ScalarExpression

        super().__init__(noise=noise)

        # validate input
        if not isinstance(rhs, OrderedDict):
            rhs = OrderedDict(rhs)
        if "t" in rhs:
            raise RuntimeError("`t` is not allowed as a variable since it denotes time")

        # turn the expression strings into sympy expressions
        signature = tuple(rhs.keys()) + ("t",)
        self._rhs_expr, self._operators = {}, {}
        explicit_time_dependence = False
        for var, rhs_str in rhs.items():
            rhs_expr = ScalarExpression(rhs_str, signature=signature)
            self._rhs_expr[var] = rhs_expr

            if rhs_expr.depends_on("t"):
                explicit_time_dependence = True

            # determine undefined functions in the expression
            self._operators[var] = {
                func.__class__.__name__
                for func in rhs_expr._sympy_expr.atoms(AppliedUndef)
            }

        # set public instance attributes
        self.rhs = rhs
        self.variables = tuple(rhs.keys())
        self.explicit_time_dependence = explicit_time_dependence
        operators = frozenset().union(*self._operators.values())  # type: ignore

        # setup boundary conditions
        if bc_ops is None:
            bcs = {"*:*": bc}
        else:
            bcs = OrderedDict(bc_ops)
            if "*:*" in bcs and bc != "natural":
                self._logger.warning("Two default boundary conditions.")
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
        self.diagnostics = {
            "variables": self.variables,
            "function_signature": signature,
            "explicit_time_dependence": explicit_time_dependence,
            "operators": operators,
        }
        self._cache: Dict[str, Any] = {}

    @property
    def expressions(self) -> Dict[str, str]:
        """ show the expressions of the PDE """
        return {k: v.expression for k, v in self._rhs_expr.items()}

    def _prepare(self, state: FieldBase) -> None:
        """prepare the expression by setting internal variables in the cache

        Note that the expensive calculations in this method are only carried
        out if the state attributes change.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                The field describing the state of the PDE
        """
        # check whether this function actually needs to be called
        if state.attributes == self._cache.get("state_attributes", None):
            return  # prepare was already called
        self._cache = {}  # clear cache, if there was something

        # check whether the state is compatible with the PDE
        num_fields = len(self.variables)
        self.diagnostics["num_fields"] = num_fields
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

        # obtain functions used in the expression
        ops_general = {}

        # create a dot operator if necessary
        if "dot" in self.diagnostics["operators"]:  # type: ignore
            # add dot product between two vector fields. This can for instance
            # appear when two gradients of scalar fields need to be multiplied
            ops_general["dot"] = VectorField(state.grid).get_dot_operator()

        # obtain the python functions for the rhs
        rhs_funcs = []
        for var in self.variables:
            ops = ops_general.copy()

            # obtain the (differential) operators
            for func in self._operators[var]:
                if func in ops:
                    continue
                # determine boundary conditions
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

                ops[func] = state.grid.get_operator(func, bc=bc)

            rhs_funcs.append(self._rhs_expr[var]._get_function(user_funcs=ops))

        self._cache["rhs_funcs"] = rhs_funcs

        # add extra information for field collection
        if isinstance(state, FieldCollection):
            # isscalar be False even if start == stop (e.g. vector fields)
            isscalar = tuple(field.rank == 0 for field in state)
            starts = tuple(slc.start for slc in state._slices)
            stops = tuple(slc.stop for slc in state._slices)

            def get_data_tuple(state_data):
                """ helper for turning state_data into a tuple of field data """
                return tuple(
                    state_data[starts[i]]
                    if isscalar[i]
                    else state_data[starts[i] : stops[i]]
                    for i in range(num_fields)
                )

            self._cache["get_data_tuple"] = get_data_tuple

        # store the attributes in the cache, which allows to later circumvent
        # calculating the quantities above again. Note that this has to be the
        # last expression of the method, so the cache is only valid when the
        # prepare function worked successfully
        self._cache["state_attributes"] = state.attributes

    def evolution_rate(self, state: FieldBase, t: float = 0) -> FieldBase:
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
        self._prepare(state)

        if isinstance(state, DataFieldBase):
            rhs = self._cache["rhs_funcs"][0]
            return state.copy(data=rhs(state.data, t))

        elif isinstance(state, FieldCollection):
            result = state.copy()
            for i in range(len(state)):
                data_tpl = self._cache["get_data_tuple"](state.data)
                result[i].data = self._cache["rhs_funcs"][i](*data_tpl, t)
            return result

        else:
            raise TypeError(f"Unsupported field {state.__class__.__name__}")

    def _make_pde_rhs_numba_coll(self, state: FieldCollection) -> Callable:
        """create the compiled rhs if `state` is a field collection

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`numpy.ndarray` giving
            the evolution rate.
        """
        num_fields = len(state)
        data_shape = state.data.shape
        rhs_list = tuple(jit(self._cache["rhs_funcs"][i]) for i in range(num_fields))

        starts = tuple(slc.start for slc in state._slices)
        stops = tuple(slc.stop for slc in state._slices)

        # In the future, the following should be possible:
        #         @jit
        #         def evolution_rate(state_data, t, out):
        #             """ evolve all agents explicitly """
        #             for i in nb.literal_unroll(range(num_fields)):
        #                 out[i] = rhs_list[i](*state_data, t)
        #         return evolver

        get_data_tuple = self._cache["get_data_tuple"]

        def chain(i=0, inner=None):
            """ recursive helper function for applying all rhs """
            # run through all evolvers
            rhs = rhs_list[i]

            if inner is None:
                # the innermost function does not need to call a child
                @jit
                def wrap(data_tpl, t, out):
                    out[starts[i] : stops[i]] = rhs(*data_tpl, t)

            else:
                # all other functions need to call one deeper in the chain
                @jit
                def wrap(data_tpl, t, out):
                    inner(data_tpl, t, out)
                    out[starts[i] : stops[i]] = rhs(*data_tpl, t)

            if i < num_fields - 1:
                # there are more items in the chain
                return chain(i + 1, inner=wrap)
            else:
                # this is the outermost function
                @jit
                def evolution_rate(state_data: np.ndarray, t: float = 0):
                    out = np.empty(data_shape)
                    with nb.objmode():
                        data_tpl = get_data_tuple(state_data)
                        wrap(data_tpl, t, out)
                    return out

                return evolution_rate

        # compile the recursive chain
        return chain()  # type: ignore

    def _make_pde_rhs_numba(self, state: FieldBase) -> Callable:
        """create a compiled function evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`numpy.ndarray` giving
            the evolution rate.
        """
        self._prepare(state)

        if isinstance(state, DataFieldBase):
            return jit(self._cache["rhs_funcs"][0])  # type: ignore
        elif isinstance(state, FieldCollection):
            return self._make_pde_rhs_numba_coll(state)
        else:
            raise TypeError(f"Unsupported field {state.__class__.__name__}")
