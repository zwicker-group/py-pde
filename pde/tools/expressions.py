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


from abc import ABCMeta, abstractproperty
import copy
import logging
import re
from typing import (Union, Callable, List, Optional, Dict, Any,
                    Set, Sequence, Tuple)  # @UnusedImport
from numbers import Number

import sympy
import numpy as np

from .cache import cached_property, cached_method
from .docstrings import fill_in_docstring

from .numba import jit



@fill_in_docstring
def parse_number(expression: Union[str, float],
                 variables: Dict[str, float] = None) -> float:
    r""" return a number compiled from an expression
    
    Warning:
        {WARNING_EXEC}
    
    Args:
        expression (str or float):
            An expression that can be interpreted as a number
        variables (dict):
            A dictionary of values that replace variables in the expression
        
    Returns:
        float: the calculated value
    """
    from sympy.parsing import sympy_parser
    
    if variables is None:
        variables = {}
    
    expr = sympy_parser.parse_expr(str(expression))
    try:
        value = float(expr.subs(variables))
    except TypeError as err:
        if not err.args: 
            err.args = ('',)
        err.args = err.args + (f"Expression: `{expr}`",)
        raise 
    return value



ExpressionType = Union[float, str, "ExpressionBase"]


class ExpressionBase(metaclass=ABCMeta):
    """ abstract base class for handling expressions """

    @fill_in_docstring
    def __init__(self, expression,
                 signature: Optional[Sequence[Union[str, List[str]]]] = None,
                 user_funcs: Optional[Dict[str, Any]] = None):
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
        self._sympy_expr = expression
        self._logger = logging.getLogger(self.__class__.__name__)
        self.user_funcs = {} if user_funcs is None else user_funcs
        self._check_signature(signature)


    def __repr__(self):
        return (f'{self.__class__.__name__}("{self.expression}", '
                f'signature={self.vars})')
        
        
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        # compare what the expressions depend on
        if set(self.vars) != set(other.vars):
            return False
        
        # compare the expressions themselves by checking their difference
        diff = sympy.simplify(self._sympy_expr - other._sympy_expr)
        if isinstance(self._sympy_expr, sympy.NDimArray):
            return diff == sympy.Array(np.zeros(self._sympy_expr.shape))
        else:
            return diff == 0


    @property
    def constant(self) -> bool:
        """ bool: whether the expression is a constant """
        return len(self._sympy_expr.free_symbols) == 0


    @abstractproperty
    def shape(self) -> Tuple[int, ...]: pass


    def _check_signature(self,
                         signature: Sequence[Union[str, List[str]]] = None):
        """ validate the variables of the expression against the signature """
        # get arguments of the expressions
        if self.constant:
            # constant expression do not depend on any variables
            args: Set[str] = set()
            if signature is None:
                signature = []
                
        else:
            # general expressions might have a variable
            args = set(str(s).split('[')[0]
                       for s in self._sympy_expr.free_symbols)
            if signature is None:
                # create signature from arguments
                signature = list(sorted(args))
        
        self._logger.debug(f'Expression arguments: {args}')
        
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
            raise RuntimeError(f'Arguments {args} were not defined in '
                               f'expression signature {signature}')
            
    
    @property
    def expression(self) -> str:
        """ str: the expression in string form """
        # turn numerical values into easily readable text
        if isinstance(self._sympy_expr, sympy.NDimArray):
            expr = self._sympy_expr.applyfunc(lambda x: x.evalf(chop=True))
        else:
            expr = self._sympy_expr.evalf(chop=True)
            
        return str(expr.xreplace({n: float(n)
                                  for n in expr.atoms(sympy.Float)}))

        
    @property
    def rank(self) -> int:
        """ int: the rank of the expression """
        return len(self.shape)
        
        
    def depends_on(self, variable: str) -> bool:
        """ determine whether the expression depends on `variable`
        
        Args:
            variable (str): the name of the variable to check for
            
        Returns:
            bool: whether the variable appears in the expression
        """
        if self.constant:
            return False
        else:
            return any(variable == str(symbol) 
                       for symbol in self._sympy_expr.free_symbols) 
    
    
    def _get_function(self, single_arg: bool = False,
                      user_funcs: Dict[str, Callable] = None) -> Callable:
        """ return function evaluating expression
        
        Args:
            single_arg (bool):
                Determines whether the returned function accepts all variables
                in a single argument as an array or whether all variables need
                to be supplied separately
            user_funcs (dict):
                Additional functions that can be used in the expression 
        
        Returns:
            function: the function
        """
        if user_funcs is None:
            user_funcs = {}
        
        variables = (self.vars,) if single_arg else self.vars
        return sympy.lambdify(variables, self._sympy_expr,  # type: ignore
                              modules=[user_funcs, self.user_funcs, 'numpy'])
        
        
    @cached_method()
    def _get_function_cached(self, single_arg: bool = False) -> Callable:
        """ return function evaluating expression
        
        Args:
            single_arg (bool):
                Determines whether the returned function accepts all variables
                in a single argument as an array or whether all variables need
                to be supplied separately
        
        Returns:
            function: the function
        """
        return self._get_function(single_arg)


    def __call__(self, *args, **kwargs):
        """ return the value of the expression for the given values """
        return self._get_function_cached(single_arg=False)(*args, **kwargs)

    
    @cached_method()
    def get_compiled(self, single_arg: bool = False) -> Callable:  
        """ return numba function evaluating expression
        
        Args:
            single_arg (bool): Determines whether the returned function accepts
                all variables in a single argument as an array or whether all
                variables need to be supplied separately
        
        Returns:
            function: the compiled function
        """
        func = self._get_function_cached(single_arg=single_arg)
        return jit(func)  # type: ignore
    


class ScalarExpression(ExpressionBase):
    """ describes a mathematical expression of a scalar quantity """
    
    shape: Tuple[int, ...] = tuple()
    
    
    @fill_in_docstring
    def __init__(self, expression: ExpressionType = 0,
                 signature: Optional[Sequence[Union[str, List[str]]]] = None,
                 user_funcs: Optional[Dict[str, Any]] = None,
                 allow_indexed: bool = False):
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
            signature = expression.vars
            self.allow_indexed = expression.allow_indexed
            if user_funcs is None:
                user_funcs = expression.user_funcs
            else:
                user_funcs.update(expression.user_funcs)
            
        elif callable(expression):
            # expression is some other callable -> not allowed anymore
            raise TypeError('Expression must be provided as string and not as '
                            'a callable function')
        
        elif isinstance(expression, Number):
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
            
        super().__init__(expression=sympy_expr, signature=signature,
                         user_funcs=user_funcs)
    
    
    @property
    def value(self) -> float:
        """ float: the value for a constant expression """
        if self.constant:
            return float(self._sympy_expr)
        else:
            raise TypeError('Only constant expressions have a defined value')
        
        
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
        return (super().__eq__(other) and
                self.allow_indexed == other.allow_indexed)
        

    def _prepare_expression(self, expression: str) -> str:
        """ replace indexed variables, if allowed """
        if self.allow_indexed:
            return re.sub(r"(\w+)(\[\w+\])", r"IndexedBase(\1)\2", expression)
        else:
            return expression
        
        
    def _var_indexed(self, var: str) -> bool:
        """ checks whether the variable `var` is used in an indexed form """
        from sympy.tensor.indexed import Indexed
        return any(isinstance(s, Indexed) and s.base.name == var
                   for s in self._sympy_expr.free_symbols)


    def differentiate(self, var: str) -> "ScalarExpression":
        """ return the expression differentiated with respect to var """
        if self.constant:
            # return empty expression
            return ScalarExpression(expression=0,
                                    signature=self.vars,
                                    allow_indexed=self.allow_indexed)  
        if self.allow_indexed:
            if self._var_indexed(var):
                # TODO: implement this
                raise NotImplementedError('Cannot differentiate with respect '
                                          'to a vector')
            
        var = self._prepare_expression(var)
        return ScalarExpression(self._sympy_expr.diff(var),
                                signature=self.vars,
                                allow_indexed=self.allow_indexed,
                                user_funcs=self.user_funcs)
    
    
    @cached_property()
    def derivatives(self) -> "TensorExpression":
        """ differentiate the expression with respect to all variables """ 
        if self.constant:
            # return empty expression
            value = np.zeros(len(self.vars))
            return TensorExpression(expression=value,
                                    signature=self.vars)
            
        if self.allow_indexed:
            if any(self._var_indexed(var) for var in self.vars):
                raise RuntimeError('Cannot calculate gradient for expressions '
                                   'with indexed variables')
                
        grad = sympy.Array([self._sympy_expr.diff(v) for v in self.vars])
        return TensorExpression(sympy.simplify(grad),
                                signature=self.vars,
                                user_funcs=self.user_funcs)
        
        
        
class TensorExpression(ExpressionBase):
    """ describes a mathematical expression of a tensorial quantity """
    
    
    @fill_in_docstring
    def __init__(self, expression: ExpressionType,
                 signature: Optional[Sequence[Union[str, List[str]]]] = None,
                 user_funcs: Optional[Dict[str, Any]] = None):
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
            
        elif isinstance(expression, np.ndarray):
            # expression is a constant array
            sympy_expr = sympy.sympify(expression)
            
        elif isinstance(expression, ImmutableNDimArray):
            # expression is an array of sympy expressions
            sympy_expr = expression
            
        else:
            # parse expression as a string
            expression = str(expression)

            # parse the expression using sympy                
            from sympy.parsing import sympy_parser
            parsed = sympy.Array(sympy_parser.parse_expr(expression))
            sympy_expr = sympy.simplify(parsed)

        super().__init__(expression=sympy_expr, signature=signature,
                         user_funcs=user_funcs)


    @property
    def shape(self) -> Tuple[int, ...]:
        """ tuple: the shape of the tensor """
        return self._sympy_expr.shape  # type: ignore


    @property
    def value(self):
        """ the value for a constant expression """
        if self.constant:
            return np.array(self._sympy_expr.tolist(), dtype=np.double)
        else:
            raise TypeError('Only constant expressions have a defined value')

        
    def differentiate(self, var: str) -> "TensorExpression":
        """ return the expression differentiated with respect to var """
        if self.constant:
            # return empty expression
            return TensorExpression(expression=np.zeros(self.shape),
                                    signature=self.vars)
        return TensorExpression(self._sympy_expr.diff(var),
                                signature=self.vars,
                                user_funcs=self.user_funcs)
    
    
    @cached_property()
    def derivatives(self) -> "TensorExpression":
        """ differentiate the expression with respect to all variables """ 
        shape = (len(self.vars),) + self.shape
        
        if self.constant:
            # return empty expression
            return TensorExpression(np.zeros(shape), signature=self.vars)
                
        # perform the derivatives with respect to all variables
        derivs = sympy.derive_by_array(self._sympy_expr,
                                       [sympy.Symbol(s) for s in self.vars])
        return TensorExpression(derivs,
                                signature=self.vars,
                                user_funcs=self.user_funcs)
            
            
            
__all__ = ["ExpressionBase", "ScalarExpression", "TensorExpression"]
