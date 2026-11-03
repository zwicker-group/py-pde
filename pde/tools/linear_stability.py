"""Handling linear stability analysis with sympy.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging

import sympy


def make_linear_operator(dependent_variables):
    """Return a linear differential sympy operator.

    Args:
        dependent_variables (sequence):
            All variables which the differential operator should be applied to. This can
            be explicit spatial coordinates but also fields that depend on these
            coordinates. All other symbols are assumed to be constants.
    """

    class LinDiffOperator(sympy.Function):
        """Represents a linear operators acting on variables listed in `dependent`"""

        dependent = dependent_variables  # variables that the operator can affect
        _logger = logging.getLogger("LinearOperator")

        # def _sympystr(self, printer):
        #     return "D"

        # def _latex(self, printer):
        #     return r"\mathcal{D}"

        @classmethod
        def eval(cls, arg):
            """Evaluate common cases of the differential operator."""
            if isinstance(arg, sympy.Order):
                if any(v in arg.args[0].free_symbols for v in cls.dependent):
                    return arg  # pull Order out of operator
                return sympy.Number(0)  # Order variable is a constant

            if not arg.free_symbols:
                return sympy.Number(0)  # differential of a constant vanishes
            return None

        def doit(self, deep=True, **hints):
            """Evaluates the differential operator for as many cases as we can."""
            if len(self.args) != 1:
                msg = "Operator expects exactly one argument"
                raise ValueError(msg)
            arg = self.args[0]  # extract singular argument

            if not any(v in arg.free_symbols for v in self.dependent):
                self._logger.debug("Eliminate LinDiffOperator(%s)", arg)
                return sympy.Number(0)

            if isinstance(arg, sympy.Add):
                # operator is linear
                return sympy.Add(
                    *[self.__class__(t).doit(deep=deep, **hints) for t in arg.args]
                )

            if isinstance(arg, sympy.Mul):
                # operator is linear
                terms = []
                for i in range(len(arg.args)):
                    consts = [
                        arg.args[j].doit(deep=deep, **hints)
                        for j in range(len(arg.args))
                        if j != i
                    ]
                    arg_i = arg.args[i]
                    if isinstance(arg_i, sympy.Order):
                        prod = sympy.Mul(arg_i.doit(deep=deep, **hints), *consts)
                    else:
                        prod = sympy.Mul(
                            self.__class__(arg_i).doit(deep=deep, **hints), *consts
                        )
                    terms.append(prod)
                return sympy.Add(*terms)

            if isinstance(arg, sympy.Pow):
                base, exp = arg.args
                if exp.is_real and exp.is_positive:
                    return exp * base ** (exp - 1) * self.__class__(base)

            return self.__class__(self.args[0].doit(deep=deep, **hints))

        def _eval_rewrite(self, rule, args, **hints):
            """Rewrite expressions using common replacements."""
            if len(args) != 1:
                msg = "Operator expects exactly one argument"
                raise ValueError(msg)
            arg = args[0]  # extract singular argument

            if rule == sympy.Add and isinstance(arg, sympy.Add):
                # operator is linear
                return sympy.Add(*[self.__class__(x) for x in arg.args])

            if rule == sympy.Mul and isinstance(arg, sympy.Mul):
                # operator is linear
                terms = []
                factors = arg.args
                for i in range(len(factors)):
                    consts = [factors[j] for j in range(len(factors)) if j != i]
                    terms.append(sympy.Mul(self.__class__(factors[i]), *consts))
                return sympy.Add(*terms)

            if rule == sympy.Pow and isinstance(arg, sympy.Pow):
                base, exp = arg.args
                if exp.is_real and exp.is_positive:
                    return exp * base ** (exp - 1) * self.__class__(base)
            return None

        def _eval_nseries(self, x, n, logx, cdir=0):
            """Evaluate power series of the linear operator."""
            if len(self.args) != 1:
                msg = "Operator expects exactly one argument"
                raise ValueError(msg)
            inner = self.args[0]._eval_nseries(x, n, logx, cdir=cdir)
            return self.__class__(inner)

        def _eval_as_leading_term(self, x, logx, cdir):
            """Return leading term of power series of the linear operator."""
            if len(self.args) != 1:
                msg = "Operator expects exactly one argument"
                raise ValueError(msg)
            if logx is not None:
                msg = f"Don't know how to deal with {logx=}"
                raise NotImplementedError(msg)
            # Not sure if the following works for all cases
            return sympy.diff(self.args[0], x)

    return LinDiffOperator
