"""
Handling mathematical expressions with sympy

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

import builtins
import copy
import logging
import math
import numbers
import re
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numba as nb  # lgtm [py/import-and-import-from]
import numpy as np
import sympy
from sympy.core import basic
from sympy.printing.pycode import PythonCodePrinter
from sympy.utilities.lambdify import _get_namespace

from ..fields.base import DataFieldBase, FieldBase
from ..grids.boundaries.axes import BoundariesData
from ..grids.boundaries.local import BCDataError
from .cache import cached_method, cached_property
from .docstrings import fill_in_docstring
from .misc import Number, number, number_array
from .numba import convert_scalar, jit
from .typing import NumberOrArray

try:
    from numba.core.extending import overload
except ImportError:
    # assume older numba module structure
    from numba.extending import overload


@fill_in_docstring
def parse_number(
    expression: Union[str, Number], variables: Mapping[str, Number] = None
) -> Number:
    r"""return a number compiled from an expression

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
        err.args = err.args + (f"Expression: `{expr}`",)
        raise

    return value


def _heaviside_implemention(x1, x2=0.5):
    """implementation of the Heaviside function used for numba and sympy

    Args:
        x1 (float): Argument of the function
        x2 (float): Value returned when the argument is zero

    Returns:
        float: 0 if x1 is negative, 1 if x1 is positive, and x2 if x1 == 0
    """
    if np.isnan(x1):
        return math.nan
    elif x1 == 0:
        return x2
    elif x1 < 0:
        return 0.0
    else:
        return 1.0


@overload(np.heaviside)
def np_heaviside(x1, x2):
    """numba implementation of the Heaviside function"""
    return _heaviside_implemention


# special functions that we want to support in expressions but that are not defined by
# sympy version 1.6 or have a different signature than expected by numba/numpy
SPECIAL_FUNCTIONS = {"Heaviside": _heaviside_implemention}


class ListArrayPrinter(PythonCodePrinter):
    """special sympy printer returning arrays as lists"""

    def _print_ImmutableDenseNDimArray(self, arr):
        arrays = ", ".join(f"{self._print(expr)}" for expr in arr)
        return f"[{arrays}]"


class NumpyArrayPrinter(PythonCodePrinter):
    """special sympy printer returning numpy arrays"""

    def _print_ImmutableDenseNDimArray(self, arr):
        arrays = ", ".join(f"asarray({self._print(expr)})" for expr in arr)
        return f"array(broadcast_arrays({arrays}))"


def parse_expr_guarded(expression: str, symbols=None, functions=None) -> basic.Basic:
    """parse an expression using sympy with extra guards

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
    # parsed as objects inherent to sympy , which breaks our expressions.
    local_dict = {}

    def fill_locals(element, sympy_cls):
        """recursive function for obtaining all symbols"""
        if isinstance(element, str):
            local_dict[element] = sympy_cls(element)
        elif hasattr(element, "__iter__"):
            for el in element:
                fill_locals(el, sympy_cls)

    fill_locals(symbols, sympy_cls=sympy.Symbol)
    fill_locals(functions, sympy_cls=sympy.Function)

    return sympy_parser.parse_expr(expression, local_dict=local_dict)  # type: ignore


ExpressionType = Union[float, str, np.ndarray, basic.Basic, "ExpressionBase"]


class ExpressionBase(metaclass=ABCMeta):
    """abstract base class for handling expressions"""

    @fill_in_docstring
    def __init__(
        self,
        expression: basic.Basic,
        signature: Sequence[Union[str, List[str]]] = None,
        *,
        user_funcs: Dict[str, Callable] = None,
        consts: Dict[str, NumberOrArray] = None,
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
                The signature defines which variables are expected in the
                expression. This is typically a list of strings identifying
                the variable names. Individual names can be specified as list,
                in which case any of these names can be used. The first item in
                such a list is the definite name and if another name of the list
                is used, the associated variable is renamed to the definite
                name. If signature is `None`, all variables in `expressions`
                are allowed.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
        """
        try:
            self._sympy_expr = sympy.simplify(expression)
        except TypeError:
            # work-around for sympy bug (github.com/sympy/sympy/issues/19829)
            self._sympy_expr = expression
        self._logger = logging.getLogger(self.__class__.__name__)
        self.user_funcs = {} if user_funcs is None else user_funcs
        self.consts = {} if consts is None else consts

        # check consistency of the arguments
        self._check_signature(signature)
        for name, value in self.consts.items():
            if isinstance(value, FieldBase):
                self._logger.warning(
                    f"Constant `{name}` is a field, but expressions usually require "
                    f"numerical arrays. Did you mean to use `{name}.data`?"
                )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}("{self.expression}", ' f"signature={self.vars})"
        )

    def __eq__(self, other):
        """compare this expression to another one"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        # compare what the expressions depend on
        if set(self.vars) != set(other.vars):
            return False

        # compare the auxiliary data
        if self.user_funcs != other.user_funcs or self.consts != other.consts:
            return False

        # compare the expressions themselves by checking their difference
        diff = sympy.simplify(self._sympy_expr - other._sympy_expr)
        if isinstance(self._sympy_expr, sympy.NDimArray):
            return diff == sympy.Array(np.zeros(self._sympy_expr.shape))
        else:
            return diff == 0

    @property
    def _free_symbols(self) -> Set:
        """return symbols that appear in the expression and are not in self.consts"""
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
    def shape(self) -> Tuple[int, ...]:
        pass

    def _check_signature(self, signature: Sequence[Union[str, List[str]]] = None):
        """validate the variables of the expression against the signature"""
        # get arguments of the expressions
        if self.constant:
            # constant expression do not depend on any variables
            args: Set[str] = set()
            if signature is None:
                signature = []

        else:
            # general expressions might have a variable
            args = set(str(s).split("[")[0] for s in self._free_symbols)
            if signature is None:
                # create signature from arguments
                signature = list(sorted(args))

        self._logger.debug(f"Expression arguments: {args}")

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
                        self._logger.info(f'Renamed variable "{old}"->"{new}"')
                    found.add(arg)
                    break

        args = set(args) - found
        if len(args) > 0:
            raise RuntimeError(
                f"Arguments {args} were not defined in expression signature {signature}"
            )

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
        """determine whether the expression depends on `variable`

        Args:
            variable (str): the name of the variable to check for

        Returns:
            bool: whether the variable appears in the expression
        """
        if self.constant:
            return False
        else:
            return any(variable == str(symbol) for symbol in self._free_symbols)

    def _get_function(
        self,
        single_arg: bool = False,
        user_funcs: Dict[str, Callable] = None,
        prepare_compilation: bool = False,
    ) -> Callable[..., NumberOrArray]:
        """return function evaluating expression

        Args:
            single_arg (bool):
                Determines whether the returned function accepts all variables
                in a single argument as an array or whether all variables need
                to be supplied separately
            user_funcs (dict):
                Additional functions that can be used in the expression
            prepare_compilation (bool):
                Determines whether all user functions are marked with
                :func:`numba.extending.register_jitable` to prepare for compilation.

        Returns:
            function: the function
        """
        # collect all the user functions
        user_functions = self.user_funcs.copy()
        if user_funcs is not None:
            user_functions.update(user_funcs)
        user_functions.update(SPECIAL_FUNCTIONS)

        if prepare_compilation:
            # transform the user functions, so they can be compiled using numba
            def compile_func(func):
                if isinstance(func, np.ufunc):
                    # this is a work-around that allows to compile numpy ufuncs
                    return jit(lambda *args: func(*args))
                else:
                    return jit(func)

            user_functions = {k: compile_func(v) for k, v in user_functions.items()}

        # initialize the printer that deals with numpy arrays correctly
        if prepare_compilation:
            printer_class: Type[PythonCodePrinter] = ListArrayPrinter
        else:
            printer_class = NumpyArrayPrinter
        printer = printer_class(
            {
                "fully_qualified_modules": False,
                "inline": True,
                "allow_unknown_functions": True,
                "user_functions": {k: k for k in user_functions},
            }
        )

        # determine the list of variables that the function depends on
        variables = (self.vars,) if single_arg else tuple(self.vars)
        constants = tuple(self.consts)

        # turn the expression into a callable function
        self._logger.info("Parse sympy expression `%s`", self._sympy_expr)
        func = sympy.lambdify(
            variables + constants,
            self._sympy_expr,
            modules=[user_functions, "numpy"],
            printer=printer,
        )

        # Apply the constants if there are any. Note that we use this pattern of a
        # partial function instead of replacing the constants in the sympy expression
        # directly since sympy does not work well with numpy arrays.
        if constants:
            const_values = tuple(self.consts[c] for c in constants)  # @UnusedVariable

            if prepare_compilation:
                func = jit(func)

            # TOOD: support keyword arguments

            def result(*args):
                return func(*args, *const_values)

        else:
            result = func
        return result

    @cached_method()
    def _get_function_cached(
        self, single_arg: bool = False, prepare_compilation: bool = False
    ) -> Callable[..., NumberOrArray]:
        """return function evaluating expression

        Args:
            single_arg (bool):
                Determines whether the returned function accepts all variables
                in a single argument as an array or whether all variables need
                to be supplied separately
            prepare_compilation (bool):
                Determines whether all user functions are marked with
                :func:`numba.extending.register_jitable` to prepare for compilation.

        Returns:
            function: the function
        """
        return self._get_function(single_arg, prepare_compilation=prepare_compilation)

    def __call__(self, *args, **kwargs) -> NumberOrArray:
        """return the value of the expression for the given values"""
        return self._get_function_cached(single_arg=False)(*args, **kwargs)

    @cached_method()
    def get_compiled(self, single_arg: bool = False) -> Callable[..., NumberOrArray]:
        """return numba function evaluating expression

        Args:
            single_arg (bool): Determines whether the returned function accepts
                all variables in a single argument as an array or whether all
                variables need to be supplied separately

        Returns:
            function: the compiled function
        """
        # compile the actual expression
        func = self._get_function_cached(
            single_arg=single_arg, prepare_compilation=True
        )
        return jit(func)  # type: ignore


class ScalarExpression(ExpressionBase):
    """describes a mathematical expression of a scalar quantity"""

    shape: Tuple[int, ...] = tuple()

    @fill_in_docstring
    def __init__(
        self,
        expression: ExpressionType = 0,
        signature: Optional[Sequence[Union[str, List[str]]]] = None,
        *,
        user_funcs: Optional[Dict[str, Callable]] = None,
        consts: Optional[Dict[str, NumberOrArray]] = None,
        explicit_symbols: Sequence[str] = None,
        allow_indexed: bool = False,
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            expression (str or float):
                The expression, which is either a number or a string that sympy
                can parse
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
                expression
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
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
            raise TypeError("Expression must be a string and not a function")

        elif isinstance(expression, numbers.Number):
            # expression is a simple number
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
        )

    def copy(self) -> ScalarExpression:
        """return a copy of the current expression"""
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

        else:
            raise TypeError("Only constant expressions have a defined value")

    @property
    def is_zero(self) -> bool:
        """bool: returns whether the expression is zero"""
        return self.constant and self.value == 0

    def __bool__(self) -> bool:
        """tests whether the expression is nonzero"""
        return not self.constant or self.value != 0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.allow_indexed == other.allow_indexed

    def _prepare_expression(self, expression: str) -> str:
        """replace indexed variables, if allowed

        Args:
            expression (str):
                An expression string that might contain variables that are indexed using
                square brackets. If this is the case, they are rewritten using the
                sympy object `IndexedBase`.
        """
        if self.allow_indexed:
            return re.sub(r"(\w+)(\[\w+\])", r"IndexedBase(\1)\2", expression)
        else:
            return expression

    def _var_indexed(self, var: str) -> bool:
        """checks whether the variable `var` is used in an indexed form"""
        from sympy.tensor.indexed import Indexed

        return any(
            isinstance(s, Indexed) and s.base.name == var for s in self._free_symbols
        )

    def differentiate(self, var: str) -> ScalarExpression:
        """return the expression differentiated with respect to var"""
        if self.constant:
            # return empty expression
            return ScalarExpression(
                expression=0, signature=self.vars, allow_indexed=self.allow_indexed
            )
        if self.allow_indexed:
            if self._var_indexed(var):
                raise NotImplementedError("Cannot differentiate with respect to vector")

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
        )

    @cached_property()
    def derivatives(self) -> TensorExpression:
        """differentiate the expression with respect to all variables"""
        if self.constant:
            # return empty expression
            dim = len(self.vars)
            expression = sympy.Array(np.zeros(dim), shape=(dim,))
            return TensorExpression(expression=expression, signature=self.vars)

        if self.allow_indexed:
            if any(self._var_indexed(var) for var in self.vars):
                raise RuntimeError(
                    "Cannot calculate gradient for expressions with indexed variables"
                )

        grad = sympy.Array([self._sympy_expr.diff(sympy.Symbol(v)) for v in self.vars])
        return TensorExpression(
            sympy.simplify(grad), signature=self.vars, user_funcs=self.user_funcs
        )


class TensorExpression(ExpressionBase):
    """describes a mathematical expression of a tensorial quantity"""

    @fill_in_docstring
    def __init__(
        self,
        expression: ExpressionType,
        signature: Optional[Sequence[Union[str, List[str]]]] = None,
        *,
        user_funcs: Optional[Dict[str, Callable]] = None,
        consts: Optional[Dict[str, NumberOrArray]] = None,
        explicit_symbols: Sequence[str] = None,
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            expression (str or float):
                The expression, which is either a number or a string that sympy
                can parse
            signature (list of str):
                The signature defines which variables are expected in the
                expression. This is typically a list of strings identifying
                the variable names. Individual names can be specified as list,
                in which case any of these names can be used. The first item in
                such a list is the definite name and if another name of the list
                is used, the associated variable is renamed to the definite
                name. If signature is `None`, all variables in `expressions`
                are allowed.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
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
        )

    def __repr__(self):
        if self.shape == (0,):
            # work-around for sympy bug (github.com/sympy/sympy/issues/19829)
            return f'{self.__class__.__name__}("[]", signature={self.vars})'
        else:
            return super().__repr__()

    @property
    def shape(self) -> Tuple[int, ...]:
        """tuple: the shape of the tensor"""
        return self._sympy_expr.shape  # type: ignore

    def __getitem__(self, index):
        expr = self._sympy_expr[index]
        if isinstance(expr, sympy.Array):
            return TensorExpression(
                expr, signature=self.vars, user_funcs=self.user_funcs
            )
        else:
            return ScalarExpression(
                expr, signature=self.vars, user_funcs=self.user_funcs
            )

    @property
    def value(self):
        """the value for a constant expression"""
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

        else:
            raise TypeError("Only constant expressions have a defined value")

    def differentiate(self, var: str) -> TensorExpression:
        """return the expression differentiated with respect to var"""
        if self.constant:
            derivative = np.zeros(self.shape)
        else:
            derivative = self._sympy_expr.diff(sympy.Symbol(var))
        return TensorExpression(derivative, self.vars, user_funcs=self.user_funcs)

    @cached_property()
    def derivatives(self) -> TensorExpression:
        """differentiate the expression with respect to all variables"""
        shape = (len(self.vars),) + self.shape

        if self.constant:
            # return empty expression
            derivatives = sympy.Array(np.zeros(shape), shape)
        else:
            # perform the derivatives with respect to all variables
            dx = sympy.Array([sympy.Symbol(s) for s in self.vars])
            derivatives = sympy.derive_by_array(self._sympy_expr, dx)

        return TensorExpression(derivatives, self.vars, user_funcs=self.user_funcs)

    def get_compiled_array(
        self, single_arg: bool = True
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """compile the tensor expression such that a numpy array is returned

        Args:
            single_arg (bool):
                Whether the compiled function expects all arguments as a single array
                or whether they are supplied individually.
        """
        assert isinstance(self._sympy_expr, sympy.Array), "Expression must be an array"
        variables = ", ".join(v for v in self.vars)
        shape = self._sympy_expr.shape

        if nb.config.DISABLE_JIT:
            # special path used by coverage test without jitting. This can be
            # removed once the `convert_scalar` wrapper is obsolete
            lines = [
                f"    out[{str(idx + (...,))[1:-1]}] = {self._sympy_expr[idx]}"
                for idx in np.ndindex(*self._sympy_expr.shape)
            ]
        else:
            lines = [
                f"    out[{str(idx + (...,))[1:-1]}] = "
                f"convert_scalar({self._sympy_expr[idx]})"
                for idx in np.ndindex(*self._sympy_expr.shape)
            ]
        # TODO: replace the np.ndindex with np.ndenumerate eventually. This does not
        # work with numpy 1.18, so we have the work around using np.ndindex

        # TODO: We should also support constants similar to ScalarExpressions. They
        # could be written in separate lines and prepended to the actual code. However,
        # we would need to make sure to print numpy arrays correctly.

        if variables:
            # the expression takes variables as input

            if single_arg:
                # the function takes a single input array
                first_dim = 0 if len(self.vars) == 1 else 1
                code = "def _generated_function(arr, out=None):\n"
                code += f"    arr = asarray(arr)\n"
                code += f"    {variables} = arr\n"
                code += f"    if out is None:\n"
                code += f"        out = empty({shape} + arr.shape[{first_dim}:])\n"

            else:
                # the function takes each variables as an argument
                code = f"def _generated_function({variables}, out=None):\n"
                code += f"    if out is None:\n"
                code += f"        out = empty({shape} + shape({self.vars[0]}))\n"

        else:
            # the expression is constant
            if single_arg:
                code = "def _generated_function(arr=None, out=None):\n"
            else:
                code = "def _generated_function(out=None):\n"
            code += f"    if out is None:\n"
            code += f"        out = empty({shape})\n"

        code += "\n".join(lines) + "\n"
        code += "    return out"

        self._logger.debug("Code for `get_compiled_array`: %s", code)

        namespace = _get_namespace("numpy")
        namespace["convert_scalar"] = convert_scalar
        namespace["builtins"] = builtins
        namespace.update(self.user_funcs)
        local_vars: Dict[str, Any] = {}
        exec(code, namespace, local_vars)
        function = local_vars["_generated_function"]

        return jit(function)  # type: ignore


@fill_in_docstring
def evaluate(
    expression: str,
    fields: Dict[str, DataFieldBase],
    *,
    bc: BoundariesData = "auto_periodic_neumann",
    bc_ops: Dict[str, BoundariesData] = None,
    user_funcs: Dict[str, Callable] = None,
    consts: Dict[str, NumberOrArray] = None,
    label: str = None,
) -> DataFieldBase:
    """evaluate an expression involving fields

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
        fields (dict):
            Dictionary of the fields involved in the expression.
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

    logger = logging.getLogger("evaluate")

    # validate input
    if consts is None:
        consts = {}

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
        bcs: Dict[str, Any] = {"*": bc}
    else:
        bcs = dict(bc_ops)
        if "*" in bcs and bc != "auto_periodic_neumann":
            logger.warning("Found default BCs in `bcs` and `bc_ops`")
        bcs["*"] = bc  # append default boundary conditions

    # check whether all fields have the same grid
    grid = None
    for field in fields.values():
        if grid is None:
            grid = field.grid
        else:
            field.grid.assert_grid_compatible(grid)
    if grid is None:
        raise ValueError("No fields given")

    # prepare the differential operators

    # check whether PDE has variables with same names as grid axes
    name_overlap = set(fields) & set(grid.axes)
    if name_overlap:
        raise ValueError(f"Coordinate {name_overlap} cannot be used as field name")

    # obtain the (differential) operators
    ops: Dict[str, Callable] = {}
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
                if bc_key == func or bc_key == "*":
                    break  # found a matching boundary condition
            else:
                raise RuntimeError(
                    f"Could not find suitable boundary condition for function `{func}`"
                )

            # Tell the user what BC we chose for a given operator
            logger.info("Using BC `%s` for operator `%s` in expression", bc, func)

            # create the function evaluating the operator
            try:
                ops[func] = grid.make_operator(func, bc=bc)
            except BCDataError:
                # wrong data was supplied for the boundary condition
                raise
            except ValueError:
                # any other exception should signal that the operator is not defined, so
                # we (almost) silently assume that sympy defines the operator
                logger.info("Assuming that sympy knows undefined operator `%s`", func)

    # obtain the function to calculate the right hand side
    signature = tuple(fields.keys()) + ("none", "bc_args")

    # check whether this function depends on additional input
    if any(expr.depends_on(c) for c in grid.axes):
        # expression has a spatial dependence, too

        # extend the signature
        signature += tuple(grid.axes)
        # inject the spatial coordinates into the expression for the rhs
        extra_args = tuple(  # @UnusedVariable
            grid.cell_coords[..., i] for i in range(grid.num_axes)
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

    logger.info("Expression has signature %s", signature)

    # extract input field data and calculate result
    field_data = [field.data for field in fields.values()]

    # calculate the result of the expression
    func = expr._get_function(single_arg=False, user_funcs=ops)
    result_data = func(*field_data, None, {}, *extra_args)

    # turn result into a proper field
    if np.isscalar(result_data):
        result_data = np.broadcast_to(result_data, grid.shape)
    result_rank = result_data.ndim - grid.num_axes
    result_cls = DataFieldBase.get_class_by_rank(result_rank)
    return result_cls(grid, result_data, label=label)


__all__ = [
    "ExpressionBase",
    "ScalarExpression",
    "TensorExpression",
    "parse_number",
    "evaluate",
]
