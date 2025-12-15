"""Handling mathematical expressions with sympy.

This module provides classes representing expressions that can be provided as
human-readable strings and are converted to :mod:`numpy` and :mod:`numba`
representations using :mod:`sympy`.

.. autosummary::
   :nosignatures:

   parse_number
   ScalarExpression
   TensorExpression
   evaluate

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import copy
import logging
import numbers
import re
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import sympy
from sympy.core import basic

from ..fields.base import FieldBase
from ..fields.collection import FieldCollection
from ..fields.datafield_base import DataFieldBase
from ..grids.boundaries.local import BCDataError
from .cache import cached_method, cached_property
from .docstrings import fill_in_docstring
from .misc import number, number_array
from .typing import Number, NumberOrArray, NumericArray

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ..grids.base import GridBase
    from ..grids.boundaries.axes import BoundariesData


_base_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger for expressions."""


@fill_in_docstring
def parse_number(
    expression: str | Number, variables: Mapping[str, Number] | None = None
) -> Number:
    r"""Return a number compiled from an expression.

    Warning:
        {WARNING_EXEC}

    Args:
        expression (str or Number):
            An expression that can be interpreted as a number
        variables (dict):
            A dictionary of values that replace variables in the expression

    Returns:
        Number: the calculated value
    """
    from sympy.parsing import sympy_parser

    if variables is None:
        variables = {}

    expr = sympy_parser.parse_expr(str(expression))
    try:
        value = number(expr.subs(variables))
    except TypeError as err:
        if not err.args:
            err.args = ("",)
        err.args = (*err.args, f"Expression: `{expr}`")
        raise

    return value


# special functions that we want to support in expressions but that are not defined by
# sympy version 1.6 or have a different signature than expected by numba/numpy
SPECIAL_FUNCTIONS: dict[str, Callable] = {
    "Heaviside": np.heaviside,  # _heaviside_implementation,
    "hypot": np.hypot,
}


def parse_expr_guarded(expression: str, symbols=None, functions=None) -> basic.Basic:
    """Parse an expression using sympy with extra guards.

    This function essentially uses :func:`~sympy.parse_expr`, albeit with a carefully
    curated dictionary for local variables so extra symbols and functions are not parsed
    as sympy functions. We also treat the heaviside function specially, since we
    otherwise could only support very specific syntax.

    Args:
        expression (str):
            The expression to be parsed
        symbols:
            (nested) collection of symbols explicitly treated as symbols by sympy
        functions:
            (nested) collection of symbols explicitly treated as functions by sympy

    Returns:
        :class:`sympy.core.basic.Basic`: The sympy expression
    """
    # parse the expression using sympy
    from sympy.parsing import sympy_parser

    # collect all symbols that are likely present and should thus be interpreted as
    # symbols by sympy. If we omit defining `local_dict`, many expressions will be
    # parsed as objects inherent to sympy, which breaks our expressions.
    local_dict = {}

    def fill_locals(element, sympy_cls):
        """Recursive function for obtaining all symbols."""
        if isinstance(element, str):
            local_dict[element] = sympy_cls(element)
        elif hasattr(element, "__iter__"):
            for el in element:
                fill_locals(el, sympy_cls)

    fill_locals(symbols, sympy_cls=sympy.Symbol)
    fill_locals(functions, sympy_cls=sympy.Function)

    expr = sympy_parser.parse_expr(expression, local_dict=local_dict)

    # replace lower-case heaviside (which would translate to numpy function) by
    # upper-case Heaviside, which is directly recognized by sympy. Down the line, this
    # allows easier handling of special cases
    def substitute(expr):
        """Helper function substituting expressions."""
        if isinstance(expr, list):
            return [substitute(e) for e in expr]
        return expr.subs(sympy.Function("heaviside"), sympy.Heaviside)

    return substitute(expr)


ExpressionType = Union[float, str, NumericArray, basic.Basic, "ExpressionBase"]


class ExpressionBase(metaclass=ABCMeta):
    """Abstract base class for handling expressions."""

    _logger: logging.Logger

    @fill_in_docstring
    def __init__(
        self,
        expression: basic.Basic,
        signature: Sequence[str | list[str]] | None = None,
        *,
        user_funcs: dict[str, Callable] | None = None,
        consts: dict[str, NumberOrArray] | None = None,
        repl: dict[str, str] | None = None,
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            expression (:class:`sympy.core.basic.Basic`):
                A sympy expression or array. This could for instance be an
                instance of :class:`~sympy.core.expr.Expr` or
                :class:`~sympy.tensor.array.ndim_array.NDimArray`.
            signature (list of str, optional):
                The signature defines which variables are expected in the expression.
                This is typically a list of strings identifying the variable names.
                Individual names can be specified as list, in which case any of these
                names can be used. The first item in such a list is the definite name
                and if another name of the list is used, the associated variable is
                renamed to the definite name. If signature is `None`, all variables in
                `expressions` are allowed.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
            repl (dict, optional):
                Replacements that are applied to symbols before turning the expression
                into a python equivalent.
        """
        try:
            self._sympy_expr = sympy.simplify(expression)
        except TypeError:
            # work-around for sympy bug (github.com/sympy/sympy/issues/19829)
            self._sympy_expr = expression
        if repl is not None:
            self._sympy_expr = self._sympy_expr.subs(repl)
        self.user_funcs = {} if user_funcs is None else user_funcs
        self.consts = {} if consts is None else consts

        # check consistency of the arguments
        self._check_signature(signature)
        for name, value in self.consts.items():
            if isinstance(value, FieldBase):
                self._logger.warning(
                    "Constant `%(name)s` is a field, but expressions usually require "
                    "numerical arrays. Did you mean to use `%(name)s.data`?",
                    {"name": name},
                )

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)
        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.expression}", signature={self.vars})'

    def __eq__(self, other):
        """Compare this expression to another one."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        # compare what the expressions depend on
        if set(self.vars) != set(other.vars):
            return False

        # compare the auxiliary data
        if self.user_funcs != other.user_funcs or self.consts != other.consts:
            return False

        # compare the expressions themselves by checking their difference
        difference = sympy.simplify(self._sympy_expr - other._sympy_expr)
        if isinstance(self._sympy_expr, sympy.NDimArray):
            return difference == sympy.Array(np.zeros(self._sympy_expr.shape, int))
        return difference == 0

    @property
    def _free_symbols(self) -> set:
        """Return symbols that appear in the expression and are not in self.consts."""
        return {
            sym for sym in self._sympy_expr.free_symbols if sym.name not in self.consts
        }

    @property
    def constant(self) -> bool:
        """bool: whether the expression is a constant"""
        return len(self._free_symbols) == 0

    @property
    def complex(self) -> bool:
        """bool: whether the expression contains the imaginary unit I"""
        return sympy.I in self._sympy_expr.atoms()

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """tuple: the shape of the tensor"""

    def _check_signature(self, signature: Sequence[str | list[str]] | None = None):
        """Validate the variables of the expression against the signature."""
        # get arguments of the expressions
        if self.constant:
            # constant expression do not depend on any variables
            args: set[str] = set()
            if signature is None:
                signature = []

        else:
            # general expressions might have a variable
            args = {str(s).split("[")[0] for s in self._free_symbols}
            if signature is None:
                # create signature from arguments
                signature = sorted(args)

        self._logger.debug("Expression arguments: %s", args)

        # check whether variables are in signature
        self.vars: Any = []
        found = set()
        for sig in signature:
            sig_list = [sig] if isinstance(sig, str) else sig

            # use the first item as the variable name
            arg_name = sig_list[0]
            self.vars.append(arg_name)

            # check whether this part of the signature is present
            for arg in args:
                if arg in sig_list:
                    if arg != arg_name:  # synonym has been used
                        old = sympy.symbols(arg)
                        new = sympy.symbols(arg_name)
                        self._sympy_expr = self._sympy_expr.subs(old, new)
                        self._logger.info('Renamed variable "%s"->"%s"', old, new)
                    found.add(arg)
                    break

        args = set(args) - found - set(self.consts)
        if len(args) > 0:
            msg = (
                f"Arguments {args} were not defined in expression signature {signature}"
            )
            raise RuntimeError(msg)

    @property
    def expression(self) -> str:
        """str: the expression in string form"""
        # turn numerical values into easily readable text
        if isinstance(self._sympy_expr, sympy.NDimArray):
            expr = self._sympy_expr.applyfunc(lambda x: x.evalf(chop=True))
        else:
            expr = self._sympy_expr.evalf(chop=True)

        return str(expr.xreplace({n: float(n) for n in expr.atoms(sympy.Float)}))

    @property
    def rank(self) -> int:
        """int: the rank of the expression"""
        return len(self.shape)

    def depends_on(self, variable: str) -> bool:
        """Determine whether the expression depends on `variable`

        Args:
            variable (str): the name of the variable to check for

        Returns:
            bool: whether the variable appears in the expression
        """
        if self.constant:
            return False
        return any(variable == str(symbol) for symbol in self._free_symbols)

    def get_function(
        self,
        backend: str = "numpy",
        *,
        single_arg: bool = False,
        user_funcs: dict[str, Callable] | None = None,
    ) -> Callable[..., NumberOrArray]:
        """Return a function evaluating expression for a particular backend.

        Args:
            backend (str):
                Backend (e.g., `numba` or `numpy`) for constructing the operator
            single_arg (bool):
                Determines whether the returned function accepts all variables in a
                single argument as an array or whether all variables need to be
                supplied separately.
            user_funcs (dict):
                Additional functions that can be used in the expression.

        Returns:
            function: the function
        """
        from ..backends import backends
        # TODO add caching

        return backends[backend].make_expression_function(
            self, single_arg=single_arg, user_funcs=user_funcs
        )

    def _get_function(
        self,
        single_arg: bool = False,
        user_funcs: dict[str, Callable] | None = None,
        prepare_compilation: bool = False,
    ) -> Callable[..., NumberOrArray]:
        """Return function evaluating expression.

        Args:
            single_arg (bool):
                Determines whether the returned function accepts all variables
                in a single argument as an array or whether all variables need
                to be supplied separately
            user_funcs (dict):
                Additional functions that can be used in the expression
            prepare_compilation (bool):
                Determines whether user functions compiled with numba

        Returns:
            function: the function
        """
        # deprecated since 2025-12-06
        warnings.warn(
            "`_get_function` is deprecated. Use `get_function` instead.", stacklevel=2
        )
        if prepare_compilation:
            return self.get_function(
                "numba", single_arg=single_arg, user_funcs=user_funcs
            )
        return self.get_function("numpy", single_arg=single_arg, user_funcs=user_funcs)

    @cached_method()
    def _get_function_cached(self) -> Callable[..., NumberOrArray]:
        """Return a cached version of the function evaluating expression."""
        return self.get_function(backend="numpy", single_arg=False)

    def __call__(self, *args, **kwargs) -> NumberOrArray:
        """Return the value of the expression for the given values."""
        return self._get_function_cached()(*args, **kwargs)

    @cached_method()
    def get_compiled(self, single_arg: bool = False) -> Callable[..., NumberOrArray]:
        """Return numba function evaluating expression.

        Args:
            single_arg (bool):
                Determines whether the function takes all variables in a single argument
                as an array or whether all variables need to be supplied separately.

        Returns:
            function: the compiled function
        """
        # deprecated since 2025-12-06
        warnings.warn(
            "`get_compiled` is deprecated. Use `get_function` instead.", stacklevel=2
        )
        return self.get_function("numba", single_arg=single_arg)


class ScalarExpression(ExpressionBase):
    """Describes a mathematical expression of a scalar quantity."""

    shape: tuple[int, ...] = ()

    @fill_in_docstring
    def __init__(
        self,
        expression: ExpressionType = 0,
        signature: Sequence[str | list[str]] | None = None,
        *,
        user_funcs: dict[str, Callable] | None = None,
        consts: dict[str, NumberOrArray] | None = None,
        repl: dict[str, str] | None = None,
        explicit_symbols: Sequence[str] | None = None,
        allow_indexed: bool = False,
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            expression (str or float):
                The expression, either a number or a string that sympy can parse.
            signature (list of str):
                The signature defines which variables are expected in the expression.
                This is typically a list of strings identifying the variable names.
                Individual names can be specified as lists, in which case any of these
                names can be used. The first item in such a list is the definite name
                and if another name of the list is used, the associated variable is
                renamed to the definite name. If signature is `None`, all variables in
                `expressions` are allowed.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
            repl (dict, optional):
                Replacements that are applied to symbols before turning the expression
                into a python equivalent.
            explicit_symbols (list of str):
                List of symbols that need to be interpreted as general sympy symbols
            allow_indexed (bool):
                Whether to allow indexing of variables. If enabled, array variables are
                allowed to be indexed using square bracket notation.
        """
        self.allow_indexed = allow_indexed

        # parse the expression
        if isinstance(expression, ScalarExpression):
            # copy constructor
            sympy_expr = copy.copy(expression._sympy_expr)
            if signature is None:
                signature = expression.vars
            self.allow_indexed = expression.allow_indexed

            if user_funcs is None:
                user_funcs = expression.user_funcs
            else:
                user_funcs.update(expression.user_funcs)

            if consts is None:
                consts = expression.consts
            else:
                consts.update(expression.consts)

        elif callable(expression):
            # expression is some other callable -> not allowed anymore
            msg = "Expression must be a string and not a function"
            raise TypeError(msg)

        elif isinstance(expression, numbers.Number):
            # expression is a simple number
            if np.iscomplex(expression):  # type: ignore
                sympy_expr = sympy.sympify(expression)
            else:
                sympy_expr = sympy.Float(expression)

        elif bool(expression):
            # parse expression as a string
            expression = self._prepare_expression(str(expression))
            sympy_expr = parse_expr_guarded(
                expression,
                symbols=[signature, consts, explicit_symbols],
                functions=user_funcs,
            )

        else:
            # expression is empty, False or None => set it to zero
            sympy_expr = sympy.Float(0)

        super().__init__(
            expression=sympy_expr,
            signature=signature,
            user_funcs=user_funcs,
            consts=consts,
            repl=repl,
        )

    def copy(self) -> ScalarExpression:
        """Return a copy of the current expression."""
        # __init__ copies all relevant attributes
        return self.__class__(self)

    @property
    def value(self) -> Number:
        """float: the value for a constant expression"""
        if self.constant:
            try:
                # try simply evaluating the expression as a number
                value = number(self._sympy_expr.evalf())

            except TypeError:
                # This can fail if user_funcs are supplied, which would not be replaced
                # in the numeric implementation above. We thus also try to call the
                # expression without any arguments
                value = number(self())  # type: ignore
                # Note that this may fail when the expression is actually constant, but
                # has a signature that forces it to depend on some arguments. However,
                # we feel this situation should not be very common, so we do not (yet)
                # deal with it.

            return value

        msg = "Only constant expressions have a defined value"
        raise TypeError(msg)

    @property
    def is_zero(self) -> bool:
        """bool: returns whether the expression is zero"""
        return self.constant and self.value == 0

    def __bool__(self) -> bool:
        """Tests whether the expression is nonzero."""
        return not self.constant or self.value != 0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.allow_indexed == other.allow_indexed

    def _prepare_expression(self, expression: str) -> str:
        """Replace indexed variables, if allowed.

        Args:
            expression (str):
                An expression string that might contain variables that are indexed using
                square brackets. If this is the case, they are rewritten using the
                sympy object `IndexedBase`.
        """
        if self.allow_indexed:
            return re.sub(r"(\w+)(\[\w+\])", r"IndexedBase(\1)\2", expression)
        return expression

    def _var_indexed(self, var: str) -> bool:
        """Checks whether the variable `var` is used in an indexed form."""
        from sympy.tensor.indexed import Indexed

        return any(
            isinstance(s, Indexed) and s.base.name == var for s in self._free_symbols
        )

    def differentiate(self, var: str) -> ScalarExpression:
        """Return the expression differentiated with respect to var."""
        if self.constant:
            # return empty expression
            return ScalarExpression(
                expression=0, signature=self.vars, allow_indexed=self.allow_indexed
            )
        if self.allow_indexed and self._var_indexed(var):
            msg = "Cannot differentiate with respect to vector"
            raise NotImplementedError(msg)

        # turn variable into sympy object and treat an indexed variable separately
        var_expr = self._prepare_expression(var)
        if "[" in var:
            from sympy.parsing import sympy_parser

            var_symbol = sympy_parser.parse_expr(var_expr)
        else:
            var_symbol = sympy.Symbol(var_expr)

        return ScalarExpression(
            self._sympy_expr.diff(var_symbol),
            signature=self.vars,
            allow_indexed=self.allow_indexed,
            user_funcs=self.user_funcs,
            consts=self.consts,
        )

    @cached_property()
    def derivatives(self) -> TensorExpression:
        """Differentiate the expression with respect to all variables."""
        if self.constant:
            # return empty expression
            dim = len(self.vars)
            expression = sympy.Array(np.zeros(dim), shape=(dim,))
            return TensorExpression(expression=expression, signature=self.vars)

        if self.allow_indexed and any(self._var_indexed(var) for var in self.vars):
            msg = "Cannot calculate gradient for expressions with indexed variables"
            raise RuntimeError(msg)

        grad = sympy.Array([self._sympy_expr.diff(sympy.Symbol(v)) for v in self.vars])
        return TensorExpression(
            sympy.simplify(grad),
            signature=self.vars,
            user_funcs=self.user_funcs,
            consts=self.consts,
        )


class TensorExpression(ExpressionBase):
    """Describes a mathematical expression of a tensorial quantity."""

    @fill_in_docstring
    def __init__(
        self,
        expression: ExpressionType,
        signature: Sequence[str | list[str]] | None = None,
        *,
        user_funcs: dict[str, Callable] | None = None,
        consts: dict[str, NumberOrArray] | None = None,
        repl: dict[str, str] | None = None,
        explicit_symbols: Sequence[str] | None = None,
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            expression (str or float):
                The expression, either a number or a string that sympy can parse.
            signature (list of str):
                The signature defines which variables are expected in the expression.
                This is typically a list of strings identifying the variable names.
                Individual names can be specified as list, in which case any of these
                names can be used. The first item in such a list is the definite name
                and if another name of the list is used, the associated variable is
                renamed to the definite name. If signature is `None`, all variables in
                `expressions` are allowed.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
            repl (dict, optional):
                Replacements that are applied to symbols before turning the expression
                into a python equivalent.
            explicit_symbols (list of str):
                List of symbols that need to be interpreted as general sympy symbols
        """
        from sympy.tensor.array.ndim_array import ImmutableNDimArray

        # parse the expression
        if isinstance(expression, TensorExpression):
            # copy constructor
            sympy_expr = copy.copy(expression._sympy_expr)
            if user_funcs is None:
                user_funcs = expression.user_funcs
            else:
                user_funcs.update(expression.user_funcs)
            if consts is None:
                consts = expression.consts
            else:
                consts.update(expression.consts)

        elif isinstance(expression, (np.ndarray, list, tuple)):
            # expression is a constant array
            sympy_expr = sympy.Array(sympy.sympify(expression))

        elif isinstance(expression, ImmutableNDimArray):
            # expression is an array of sympy expressions
            sympy_expr = expression

        else:
            # parse expression as a string
            sympy_expr_raw = parse_expr_guarded(
                str(expression),
                symbols=[signature, consts, explicit_symbols],
                functions=user_funcs,
            )
            sympy_expr = sympy.Array(sympy_expr_raw)

        super().__init__(
            expression=sympy_expr,
            signature=signature,
            user_funcs=user_funcs,
            consts=consts,
            repl=repl,
        )

    def __repr__(self):
        if self.shape == (0,):
            # work-around for sympy bug (github.com/sympy/sympy/issues/19829)
            return f'{self.__class__.__name__}("[]", signature={self.vars})'
        return super().__repr__()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._sympy_expr.shape  # type: ignore

    @property
    def rank(self) -> int:
        """int: rank of the tensor expression"""
        return len(self.shape)

    def __getitem__(self, index):
        expr = self._sympy_expr[index]
        if isinstance(expr, sympy.Array):
            return TensorExpression(
                expr,
                signature=self.vars,
                user_funcs=self.user_funcs,
                consts=self.consts,
            )
        return ScalarExpression(
            expr,
            signature=self.vars,
            user_funcs=self.user_funcs,
            consts=self.consts,
        )

    @property
    def value(self):
        """The value for a constant expression."""
        if self.constant:
            try:
                # try simply evaluating the expression as a number
                value = number_array(self._sympy_expr.tolist())

            except TypeError:
                # This can fail if user_funcs are supplied, which would not be replaced
                # in the numeric implementation above. We thus also try to call the
                # expression without any arguments
                value = number_array(self())
                # Note that this may fail when the expression is actually constant, but
                # has a signature that forces it to depend on some arguments. However,
                # we feel this situation should not be very common, so we do not (yet)
                # deal with it.

            return value

        msg = "Only constant expressions have a defined value"
        raise TypeError(msg)

    def differentiate(self, var: str) -> TensorExpression:
        """Return the expression differentiated with respect to var."""
        if self.constant:
            derivative = np.zeros(self.shape)
        else:
            derivative = self._sympy_expr.diff(sympy.Symbol(var))
        return TensorExpression(
            derivative, self.vars, user_funcs=self.user_funcs, consts=self.consts
        )

    @cached_property()
    def derivatives(self) -> TensorExpression:
        """Differentiate the expression with respect to all variables."""
        shape = (len(self.vars), *self.shape)

        if self.constant:
            # return empty expression
            derivatives = sympy.Array(np.zeros(shape), shape)
        else:
            # perform the derivatives with respect to all variables
            dx = sympy.Array([sympy.Symbol(s) for s in self.vars])
            derivatives = sympy.derive_by_array(self._sympy_expr, dx)

        return TensorExpression(
            derivatives, self.vars, user_funcs=self.user_funcs, consts=self.consts
        )

    def get_compiled_array(
        self, single_arg: bool = True
    ) -> Callable[[NumericArray, NumericArray | None], NumericArray]:
        """Compile the tensor expression such that a numpy array is returned.

        Args:
            single_arg (bool):
                Whether the compiled function expects all arguments as a single array
                or whether they are supplied individually.
        """
        from ..backends.numba import numba_backend

        # deprecated since 2025-12-06
        warnings.warn(
            "`get_compiled_array` is deprecated. Use `_make_expression_array` of the "
            "`numba` backend instead.",
            stacklevel=2,
        )

        return numba_backend._make_expression_array(self, single_arg=single_arg)


@fill_in_docstring
def evaluate(
    expression: str,
    fields: dict[str, DataFieldBase] | FieldCollection,
    *,
    bc: BoundariesData = "auto_periodic_neumann",
    bc_ops: dict[str, BoundariesData] | None = None,
    user_funcs: dict[str, Callable] | None = None,
    consts: dict[str, NumberOrArray] | None = None,
    label: str | None = None,
) -> DataFieldBase:
    """Evaluate an expression involving fields.

    Warning:
        {WARNING_EXEC}

    Args:
        expression (str):
            The expression, which is parsed by :mod:`sympy`. The expression may contain
            variables (i.e., fields and spatial coordinates of the grid), standard local
            mathematical operators defined by sympy, and the operators defined in the
            :mod:`pde` package. Note that operators need to be specified with their full
            name, i.e., `laplace` for a scalar Laplacian and `vector_laplace` for a
            Laplacian operating on a vector field. Moreover, the dot product between
            two vector fields can be denoted by using `dot(field1, field2)` in the
            expression, and `outer(field1, field2)` calculates an outer product.
            More information can be found in the
            :ref:`expression documentation <documentation-expressions>`.
        fields (dict or :class:`~pde.fields.collection.FieldCollection`):
            Dictionary of the fields involved in the expression. The dictionary keys
            specify the field names allowed in `expression`. Alternatively, `fields` can
            be a :class:`~pde.fields.collection.FieldCollection` with unique labels.
        bc:
            Boundary conditions for the operators used in the expression. The conditions
            here are applied to all operators that do not have a specialized condition
            given in `bc_ops`.
            {ARG_BOUNDARIES}
        bc_ops (dict):
            Special boundary conditions for some operators. The keys in this dictionary
            specify the operator to which the boundary condition will be applied.
        user_funcs (dict, optional):
            A dictionary with user defined functions that can be used in the expressions
            in `rhs`.
        consts (dict, optional):
            A dictionary with user defined constants that can be used in the expression.
            These can be either scalar numbers or fields defined on the same grid as the
            actual simulation.
        label (str):
            Name of the field that is returned.

    Returns:
        :class:`pde.fields.base.DataFieldBase`: The resulting field. The rank of the
        returned field (and thus the precise class) is determined automatically.
    """
    from sympy.core.function import AppliedUndef

    from ..fields import VectorField

    # validate input
    if consts is None:
        consts = {}

    # get keys and values from input
    if isinstance(fields, FieldCollection):
        fields_keys = fields.labels
        fields_values = fields.fields
        if len(set(fields_keys)) != len(fields_values):
            msg = "Field names need to be unique"
            raise RuntimeError(msg)
    elif isinstance(fields, dict):
        fields_keys = fields.keys()  # type: ignore
        fields_values = fields.values()  # type: ignore
    else:
        msg = "`fields` must be dict or FieldCollection"
        raise TypeError(msg)

    # turn the expression strings into sympy expressions
    expr = ScalarExpression(expression, user_funcs=user_funcs, consts=consts)

    # determine undefined functions in the expression
    operators = {
        func.__class__.__name__
        for func in expr._sympy_expr.atoms(AppliedUndef)
        if func.__class__.__name__ not in expr.user_funcs
    }

    # setup boundary conditions
    if bc_ops is None:
        bcs: dict[str, Any] = {"*": bc}
    else:
        bcs = dict(bc_ops)
        if "*" in bcs and bc != "auto_periodic_neumann":
            _base_logger.warning("Found default BCs in `bcs` and `bc_ops`")
        bcs["*"] = bc  # append default boundary conditions

    # check whether all fields have the same grid
    grid: GridBase | None = None
    for field in fields_values:
        if grid is None:
            grid = field.grid
        else:
            field.grid.assert_grid_compatible(grid)
    if grid is None:
        msg = "No fields given"
        raise ValueError(msg)

    # prepare the differential operators

    # check whether PDE has variables with same names as grid axes
    name_overlap = set(fields_keys) & set(grid.axes)
    if name_overlap:
        msg = f"Coordinate {name_overlap} cannot be used as field name"
        raise ValueError(msg)

    # obtain the (differential) operators
    bcs_used = set()
    ops: dict[str, Callable] = {}
    for func in operators:
        if func == "dot" or func == "inner":
            # add dot product between two vector fields. This can for instance
            # appear when two gradients of scalar fields need to be multiplied
            ops[func] = VectorField(grid).make_dot_operator(backend="numpy")

        elif func == "outer":
            # generate an operator that calculates an outer product
            ops[func] = VectorField(grid).make_outer_prod_operator(backend="numpy")

        else:
            # determine boundary conditions for this operator and variable
            for bc_key, bc in bcs.items():
                if bc_key == func or bc_key == "*":  # found a matching condition
                    bcs_used.add(bc_key)  # mark it as being used
                    break
            else:
                msg = (
                    f"Could not find suitable boundary condition for function `{func}`"
                )
                raise RuntimeError(msg)

            # Tell the user what BC we chose for a given operator
            _base_logger.info("Using BC `%s` for operator `%s` in expression", bc, func)

            # create the function evaluating the operator
            try:
                ops[func] = grid.make_operator(func, bc=bc, backend="numba")
            except BCDataError:
                # wrong data was supplied for the boundary condition
                raise
            except ValueError:
                # any other exception should signal that the operator is not defined, so
                # we (almost) silently assume that sympy defines the operator
                _base_logger.info(
                    "Assuming that sympy knows undefined operator `%s`", func
                )

    # check whether there are boundary conditions that have not been used
    bcs_left = set(bcs.keys()) - bcs_used - {"*:*", "*"}
    if bcs_left:
        _base_logger.warning("Unused BCs: %s", sorted(bcs_left))

    # obtain the function to calculate the right hand side
    signature = (*tuple(fields_keys), "none", "bc_args")

    # check whether this function depends on additional input
    if any(expr.depends_on(c) for c in grid.axes):
        # expression has a spatial dependence, too

        # extend the signature
        signature += tuple(grid.axes)
        # inject the spatial coordinates into the expression for the rhs
        extra_args = tuple(grid.cell_coords[..., i] for i in range(grid.num_axes))

    else:
        # expression only depends on the actual variables
        extra_args = ()

    # check whether all variables are accounted for
    extra_vars = set(expr.vars) - set(signature)
    if extra_vars:
        extra_vars_str = ", ".join(sorted(extra_vars))
        msg = f"Undefined variable in expression: {extra_vars_str}"
        raise RuntimeError(msg)
    expr.vars = signature

    _base_logger.info("Expression has signature %s", signature)

    # extract input field data and calculate result
    field_data = [field.data for field in fields_values]

    # calculate the result of the expression
    func = expr._get_function(single_arg=False, user_funcs=ops)
    result_data = func(*field_data, None, {}, *extra_args)

    # turn result into a proper field
    if np.isscalar(result_data):
        result_data = np.broadcast_to(result_data, grid.shape)
    elif TYPE_CHECKING:
        result_data = np.asanyarray(result_data)
    result_rank = result_data.ndim - grid.num_axes
    result_cls = DataFieldBase.get_class_by_rank(result_rank)
    return result_cls(grid, result_data, label=label)


__all__ = [
    "ExpressionBase",
    "ScalarExpression",
    "TensorExpression",
    "evaluate",
    "parse_number",
]
