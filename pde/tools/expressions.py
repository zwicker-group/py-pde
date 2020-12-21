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
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""


import builtins
import copy
import logging
import numbers
import re
from abc import ABCMeta, abstractproperty
from typing import Optional  # @UnusedImport
from typing import Set  # @UnusedImport
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import numba as nb  # lgtm [py/import-and-import-from]
import numpy as np
import sympy
from sympy.printing.pycode import PythonCodePrinter
from sympy.utilities.lambdify import _get_namespace

from ..tools.misc import Number, number, number_array
from .cache import cached_method, cached_property
from .docstrings import fill_in_docstring
from .numba import convert_scalar, jit

try:
    from numba.core.extending import overload
except ImportError:
    # assume older numba module structure
    from numba.extending import overload

if TYPE_CHECKING:
    from sympy.core import basic  # @UnusedImport


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


@overload(np.heaviside)
def np_heaviside(x1, x2):
    """ numba implementation of the heaviside function """

    def heaviside_impl(x1, x2):
        if np.isnan(x1):
            return np.nan
        elif x1 == 0:
            return x2
        elif x1 < 0:
            return 0.0
        else:
            return 1.0

    return heaviside_impl


# special functions that we want to support in expressions but that are not defined by
# sympy version 1.6
SPECIAL_FUNCTIONS = {"Heaviside": lambda x: np.heaviside(x, 0.5)}


class ListArrayPrinter(PythonCodePrinter):
    """ special sympy printer returning arrays as lists """

    def _print_ImmutableDenseNDimArray(self, arr):
        arrays = ", ".join(f"{self._print(expr)}" for expr in arr)
        return f"[{arrays}]"


class NumpyArrayPrinter(PythonCodePrinter):
    """ special sympy printer returning numpy arrays """

    def _print_ImmutableDenseNDimArray(self, arr):
        arrays = ", ".join(f"asarray({self._print(expr)})" for expr in arr)
        return f"array(broadcast_arrays({arrays}))"


ExpressionType = Union[float, str, "ExpressionBase"]


class ExpressionBase(metaclass=ABCMeta):
    """ abstract base class for handling expressions """

    @fill_in_docstring
    def __init__(
        self,
        expression: "basic.Basic",
        signature: Sequence[Union[str, List[str]]] = None,
        user_funcs: Dict[str, Any] = None,
        consts: Dict[str, Any] = None,
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
                expression
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression
        """
        try:
            self._sympy_expr = sympy.simplify(expression)
        except TypeError:
            # work-around for sympy bug (github.com/sympy/sympy/issues/19829)
            self._sympy_expr = expression
        self._logger = logging.getLogger(self.__class__.__name__)
        self.user_funcs = {} if user_funcs is None else user_funcs
        self.consts = {} if consts is None else consts
        self._check_signature(signature)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}("{self.expression}", ' f"signature={self.vars})"
        )

    def __eq__(self, other):
        """ compare this expression to another one """
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
        """ return symbols that appear in the expression and are not in self.consts """
        return {
            sym for sym in self._sympy_expr.free_symbols if sym.name not in self.consts
        }

    @property
    def constant(self) -> bool:
        """ bool: whether the expression is a constant """
        return len(self._free_symbols) == 0

    @property
    def complex(self) -> bool:
        """ bool: whether the expression contains the imaginary unit I """
        return sympy.I in self._sympy_expr.atoms()

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        pass

    def _check_signature(self, signature: Sequence[Union[str, List[str]]] = None):
        """ validate the variables of the expression against the signature """
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
        """ str: the expression in string form """
        # turn numerical values into easily readable text
        if isinstance(self._sympy_expr, sympy.NDimArray):
            expr = self._sympy_expr.applyfunc(lambda x: x.evalf(chop=True))
        else:
            expr = self._sympy_expr.evalf(chop=True)

        return str(expr.xreplace({n: float(n) for n in expr.atoms(sympy.Float)}))

    @property
    def rank(self) -> int:
        """ int: the rank of the expression """
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
    ) -> Callable:
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
            printer_class = ListArrayPrinter
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
    ) -> Callable:
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

    def __call__(self, *args, **kwargs):
        """ return the value of the expression for the given values """
        return self._get_function_cached(single_arg=False)(*args, **kwargs)

    @cached_method()
    def get_compiled(self, single_arg: bool = False) -> Callable:
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
    """ describes a mathematical expression of a scalar quantity """

    shape: Tuple[int, ...] = tuple()

    @fill_in_docstring
    def __init__(
        self,
        expression: ExpressionType = 0,
        signature: Optional[Sequence[Union[str, List[str]]]] = None,
        user_funcs: Optional[Dict[str, Any]] = None,
        consts: Optional[Dict[str, Any]] = None,
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
                The signature defines which variables are expected in the
                expression. This is typically a list of strings identifying
                the variable names. Individual names can be specified as lists,
                in which case any of these names can be used. The firstm item in
                such a list is the definite name and if another name of the list
                is used, the associated variable is renamed to the definite
                name. If signature is `None`, all variables in `expressions`
                are allowed.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression
            allow_indexed (bool):
                Whether to allow indexing of variables. If enabled, array
                variables are allowed to be indexed using square bracket
                notation.
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

            # parse the expression using sympy
            from sympy.parsing import sympy_parser

            sympy_expr = sympy_parser.parse_expr(expression)

        else:
            # expression is empty, False or None => set it to zero
            sympy_expr = sympy.Float(0)

        super().__init__(
            expression=sympy_expr,
            signature=signature,
            user_funcs=user_funcs,
            consts=consts,
        )

    @property
    def value(self) -> Number:
        """ float: the value for a constant expression """
        if self.constant:
            try:
                # try simply evaluating the expression as a number
                value = number(self._sympy_expr.evalf())

            except TypeError:
                # This can fail if user_funcs are supplied, which would not be replaced
                # in the numeric implementation above. We thus also try to call the
                # expression without any arguments
                value = number(self())
                # Note that this may fail when the expression is actually constant, but
                # has a signature that forces it to depend on some arguments. However,
                # we feel this situation should not be very common, so we do not (yet)
                # deal with it.

            return value

        else:
            raise TypeError("Only constant expressions have a defined value")

    @property
    def is_zero(self) -> bool:
        """ bool: returns whether the expression is zero """
        return self.constant and self.value == 0

    def __bool__(self):
        """ tests whether the expression is nonzero """
        return not self.constant or self.value != 0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.allow_indexed == other.allow_indexed

    def _prepare_expression(self, expression: str) -> str:
        """ replace indexed variables, if allowed """
        if self.allow_indexed:
            return re.sub(r"(\w+)(\[\w+\])", r"IndexedBase(\1)\2", expression)
        else:
            return expression

    def _var_indexed(self, var: str) -> bool:
        """ checks whether the variable `var` is used in an indexed form """
        from sympy.tensor.indexed import Indexed

        return any(
            isinstance(s, Indexed) and s.base.name == var for s in self._free_symbols
        )

    def differentiate(self, var: str) -> "ScalarExpression":
        """ return the expression differentiated with respect to var """
        if self.constant:
            # return empty expression
            return ScalarExpression(
                expression=0, signature=self.vars, allow_indexed=self.allow_indexed
            )
        if self.allow_indexed:
            if self._var_indexed(var):
                # TODO: implement this
                raise NotImplementedError("Cannot differentiate with respect to vector")

        var = self._prepare_expression(var)
        return ScalarExpression(
            self._sympy_expr.diff(var),
            signature=self.vars,
            allow_indexed=self.allow_indexed,
            user_funcs=self.user_funcs,
        )

    @cached_property()
    def derivatives(self) -> "TensorExpression":
        """ differentiate the expression with respect to all variables """
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

        grad = sympy.Array([self._sympy_expr.diff(v) for v in self.vars])
        return TensorExpression(
            sympy.simplify(grad), signature=self.vars, user_funcs=self.user_funcs
        )


class TensorExpression(ExpressionBase):
    """ describes a mathematical expression of a tensorial quantity """

    @fill_in_docstring
    def __init__(
        self,
        expression: ExpressionType,
        signature: Optional[Sequence[Union[str, List[str]]]] = None,
        user_funcs: Optional[Dict[str, Any]] = None,
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
                expression
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
            expression = str(expression)

            # parse the expression using sympy
            from sympy.parsing import sympy_parser

            sympy_expr = sympy.Array(sympy_parser.parse_expr(expression))

        super().__init__(
            expression=sympy_expr, signature=signature, user_funcs=user_funcs
        )

    def __repr__(self):
        if self.shape == (0,):
            # work-around for sympy bug (github.com/sympy/sympy/issues/19829)
            return f'{self.__class__.__name__}("[]", signature={self.vars})'
        else:
            return super().__repr__()

    @property
    def shape(self) -> Tuple[int, ...]:
        """ tuple: the shape of the tensor """
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
        """ the value for a constant expression """
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

    def differentiate(self, var: str) -> "TensorExpression":
        """ return the expression differentiated with respect to var """
        if self.constant:
            derivative = np.zeros(self.shape)
        else:
            derivative = self._sympy_expr.diff(var)
        return TensorExpression(derivative, self.vars, user_funcs=self.user_funcs)

    @cached_property()
    def derivatives(self) -> "TensorExpression":
        """ differentiate the expression with respect to all variables """
        shape = (len(self.vars),) + self.shape

        if self.constant:
            # return empty expression
            derivatives = sympy.Array(np.zeros(shape), shape)
        else:
            # perform the derivatives with respect to all variables
            dx = sympy.Array([sympy.Symbol(s) for s in self.vars])
            derivatives = sympy.derive_by_array(self._sympy_expr, dx)

        return TensorExpression(derivatives, self.vars, user_funcs=self.user_funcs)

    def get_compiled_array(self, single_arg: bool = True) -> Callable:
        """compile the tensor expression such that a numpy array is returned

        Args:
            single_arg (bool):
                Whether the compiled function expects all arguments as a single array
                or whether they are supplied individually.
        """
        assert isinstance(self._sympy_expr, sympy.Array)
        variables = ", ".join(v for v in self.vars)
        shape = self._sympy_expr.shape

        if nb.config.DISABLE_JIT:
            # special path used by coverage test without jitting. This can be
            # removed once the `convert_scalar` wrapper is obsolete
            lines = [
                f"    out[{str(idx + (...,))[1:-1]}] = {val}"
                for idx, val in np.ndenumerate(self._sympy_expr)
            ]
        else:
            lines = [
                f"    out[{str(idx + (...,))[1:-1]}] = convert_scalar({val})"
                for idx, val in np.ndenumerate(self._sympy_expr)
            ]

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


__all__ = ["ExpressionBase", "ScalarExpression", "TensorExpression"]
