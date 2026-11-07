"""Handling linear stability analysis with sympy.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from collections.abc import Iterable, Sequence
from typing import Any, Literal, overload

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef


class LinearDifferentialOperator(sp.Function):
    """Base class for abstract linear differential operators."""

    nargs = 1

    @classmethod
    def eval(cls, arg: sp.Basic) -> sp.Basic | None:
        """Simplify the operator when its argument is constant.

        Args:
            arg: Expression to which the operator is applied.

        Returns:
            Zero if ``arg`` is a constant, otherwise ``None`` to leave it unevaluated.
        """
        # Any linear differential operator applied to a constant is zero.
        if arg.is_number:
            return sp.S.Zero
        return None

    def _eval_derivative(self, symbol: sp.Symbol) -> sp.Basic:
        """Differentiate the argument and apply the same operator.

        Args:
            symbol: Symbol with respect to which the expression is differentiated.

        Returns:
            The same operator applied to the differentiated argument.
        """
        return type(self)(sp.diff(self.args[0], symbol))


class Laplace(LinearDifferentialOperator): ...


class Gradient(LinearDifferentialOperator): ...


class Divergence(LinearDifferentialOperator): ...


class Dot(sp.Function):
    """Abstract inner product of two vector expressions."""

    nargs = 2

    @classmethod
    def eval(cls, left: sp.Basic, right: sp.Basic) -> sp.Basic | None:
        """Simplify the inner product when either argument is zero.

        Args:
            left: First expression in the inner product.
            right: Second expression in the inner product.

        Returns:
            Zero if either argument is zero, otherwise ``None`` to leave it unevaluated.
        """
        # 0 · v = v · 0 = 0
        if left == 0 or right == 0:
            return sp.S.Zero
        return None

    def _eval_derivative(self, symbol: sp.Symbol) -> sp.Basic:
        """Differentiate the inner product using the product rule.

        Args:
            symbol: Symbol with respect to which the expression is differentiated.

        Returns:
            The derivative of the inner product.
        """
        left, right = self.args

        return Dot(sp.diff(left, symbol), right) + Dot(left, sp.diff(right, symbol))


class GradientSquared(sp.Function):
    """Abstract squared gradient: |∇f|²."""

    nargs = 1

    @classmethod
    def eval(cls, arg: sp.Basic) -> sp.Basic:
        """Rewrite the squared gradient as an inner product.

        Args:
            arg: Expression whose gradient is squared.

        Returns:
            The inner product of the gradient with itself.
        """
        # |∇ const|² = 0
        if arg.is_number:
            return sp.S.Zero

        # Use the canonical representation ∇f · ∇f.
        return Dot(Gradient(arg), Gradient(arg))


# differential operators that annihilate homogeneous expressions
DIFF_OPERATORS: dict[str, type[LinearDifferentialOperator]] = {
    "gradient": Gradient,
    "divergence": Divergence,
    "laplace": Laplace,
    "inner": Dot,
    "dot": Dot,
    "gradient_squared": GradientSquared,
}


def replace_many(
    expr: sp.Basic, search: Iterable[sp.Basic], replace: Iterable[sp.Basic | float]
) -> sp.Basic:
    """Replace multiple subexpressions simultaneously.

    Args:
        expr: Expression in which substitutions are made.
        search: Subexpressions to replace.
        replace: Replacement expressions or numeric values, paired with ``search``.

    Returns:
        The expression with the substitutions applied simultaneously.
    """
    return expr.subs(dict(zip(search, replace, strict=True)), simultaneous=True)


def _simplify_homogeneous_state(
    expr: sp.Basic, homogeneous_symbols: set[sp.Symbol]
) -> sp.Basic:
    """Simplify abstract differential operators around a homogeneous state.

    Args:
        expr:
            SymPy expression containing fields and abstract differential operators.
        homogeneous_symbols:
            Symbols known to be spatially homogeneous, including parameters and
            reference-state values such as ``{c0, phi0, D, a, b}``.

    Returns:
        The simplified expression.

    Notes:
        For each linear differential operator ``D``, this implements
        ``D[A] = 0`` for homogeneous ``A``, ``D[f + g] = D[f] + D[g]``, and
        ``D[A * f] = A * D[f]`` when ``A`` is homogeneous. Nonlinear operators
        should not be included.
    """
    expr = sp.sympify(expr)

    def is_homogeneous(value: sp.Basic) -> Any:
        """Return whether a value is spatially constant in this analysis.

        Args:
            value: Expression to check for dependence on nonhomogeneous symbols.

        Returns:
            Whether the expression contains only homogeneous symbols.
        """
        return value.free_symbols <= homogeneous_symbols

    def simplify(value: sp.Basic) -> sp.Basic:
        """Recursively simplify abstract differential operators.

        Args:
            value: Expression to simplify.

        Returns:
            The recursively simplified expression.
        """
        # Recursively simplify ordinary SymPy expressions first.
        if not value.args:
            return value

        args = tuple(simplify(arg) for arg in value.args)
        value = value.func(*args)

        # Apply special rules to the abstract differential operators.
        if any(isinstance(value, op) for op in DIFF_OPERATORS.values()):
            operator = value.func
            arg = value.args[0]

            # D[constant] = 0
            if is_homogeneous(arg):
                return sp.S.Zero

            # D[f + g] = D[f] + D[g]
            if arg.is_Add:
                return sp.Add(*(simplify(operator(term)) for term in arg.args))

            # D[A f] = A D[f] when A is homogeneous.
            if arg.is_Mul:
                homogeneous_factors = []
                nonhomogeneous_factors = []

                for factor in arg.args:
                    if is_homogeneous(factor):
                        homogeneous_factors.append(factor)
                    else:
                        nonhomogeneous_factors.append(factor)

                if homogeneous_factors:
                    prefactor = sp.Mul(*homogeneous_factors)
                    remainder = sp.Mul(*nonhomogeneous_factors)

                    # The all-homogeneous case was already handled above,
                    # but retaining this guard makes the logic explicit.
                    if remainder == 1:
                        return sp.S.Zero

                    return prefactor * simplify(operator(remainder))

            return value

        return value

    return sp.simplify(simplify(expr))


def _replace_operators(expr: sp.Basic) -> sp.Basic:
    """Replace differential operators with abstract SymPy operators.

    Args:
        expr: Expression containing differential operators.

    Returns:
        The expression with recognized operators replaced.
    """
    undefined_operators = list(expr.atoms(AppliedUndef))
    for func in undefined_operators:
        if func.__class__.__name__ in DIFF_OPERATORS:
            expr = expr.replace(func.__class__, DIFF_OPERATORS[func.__class__.__name__])
    return expr


def linearize_expression(
    expression: sp.Basic | str, state_hom: dict[str, float], *, check: bool = True
) -> sp.Basic:
    """Linearize one expression around a homogeneous base state.

    Note:
        This function assumes that the base state

    Args:
        expression: Expression to linearize.
        state_hom: Mapping from the dependent-variable name to its base value.
        check: If True, verify that the expression vanishes at the base state.

    Returns:
        The expression linearized around the base state.
    """
    # parse base state
    if len(state_hom) != 1:
        msg = "Need exactly one dependent variable"
        raise ValueError(msg)
    ((variable, base_value),) = state_hom.items()
    variable = sp.Symbol(variable)

    # turn differential operators into their Sympy equivalents
    if isinstance(expression, str):
        expression = sp.parse_expr(expression)
    expr = _replace_operators(expression)

    if check:
        # check whether the expression vanishes for the base state
        base_rate = sp.N(expr.subs(variable, base_value), chop=True)
        if not base_rate.is_zero:
            msg = f"Expression {expression} does not vanish for base state {state_hom}"
            raise ValueError(msg)

    # linearize the expr wrt. the dependent variable
    base_state = sp.Dummy("base")
    eps = sp.Dummy("eps")
    perturbation = sp.Dummy("pert")
    expr_perturbed = expr.subs(variable, base_state + eps * perturbation)
    linearized = sp.diff(expr_perturbed, eps).subs(eps, 0)

    # simplify the linearized expression
    linearized_simpl = _simplify_homogeneous_state(
        linearized, homogeneous_symbols={base_state}
    )

    # replace expression linearized around base state
    return linearized_simpl.subs({base_state: base_value, perturbation: variable})


def _parse_base_state(
    state_hom: dict[str, float],
) -> tuple[list[sp.Basic], list[float]]:
    # parse base state
    variables = []
    base_values = []
    for name, value in state_hom.items():
        variables.append(sp.Symbol(name))
        base_values.append(value)
    return variables, base_values


def linearize_expressions(
    expressions: Sequence[sp.Basic | str],
    state_hom: dict[str, float],
    *,
    check: bool = True,
) -> list[sp.Basic]:
    """Linearize several expressions around a homogeneous reference state.

    Args:
        expressions: Expressions to linearize, one for each dependent variable.
        state_hom: Mapping from dependent-variable names to their reference values.
        check: If True, verify that the expression vanishes at the base state.

    Returns:
        The linearized expressions in the same order as ``expressions``.
    """
    # parse input
    num_vars = len(state_hom)
    variables, base_values = _parse_base_state(state_hom)

    # prepare the perturbed variables
    base_state = sp.symbols(f"base_0:{num_vars}", cls=sp.Dummy)
    eps = sp.Dummy("eps")
    perturbation = sp.symbols(f"pert_0:{num_vars}", cls=sp.Dummy)
    variables_pert = [
        base + eps * pert for base, pert in zip(base_state, perturbation, strict=True)
    ]

    result = []
    for expr in expressions:
        if isinstance(expr, str):
            expr = sp.parse_expr(expr)
        # turn differential operators into their Sympy equivalents
        expr_repl = _replace_operators(expr)

        if check:
            # check whether the expression vanishes for the base state
            base_rate = sp.N(replace_many(expr_repl, variables, base_values), chop=True)
            if not base_rate.is_zero:
                msg = f"Expression {expr} does not vanish for base state {state_hom}"
                raise ValueError(msg)

        # linearize the expr wrt. the dependent variables
        expr_perturbed = replace_many(expr_repl, variables, variables_pert)
        linearized = sp.diff(expr_perturbed, eps).subs(eps, 0)

        # simplify the linearized expression
        linearized_simpl = _simplify_homogeneous_state(
            linearized, homogeneous_symbols=set(base_state)
        )

        # replace the base state and return to the original variables
        res = replace_many(linearized_simpl, base_state, base_values)
        result.append(replace_many(res, perturbation, variables))

    # replace base state
    return result


def _fourier_lower_scalar(
    expr: sp.Basic,
    amplitudes: dict[sp.Basic, sp.Basic],
    *,
    k_sq: sp.Basic,
) -> sp.Basic:
    """Replace perturbation fields and Laplacians by Fourier expressions.

    Args:
        expr:
            A linearized and homogeneous-state-simplified SymPy expression.
        amplitudes:
            Mapping from real-space perturbation symbols to Fourier amplitudes.
            For example, ``{u: u_hat, v: v_hat}``.
        k_sq:
            Symbol representing |k|**2. If omitted, a fresh symbol is created.

    Returns:
        tuple ``(expr_fourier, k_sq)`` where `expr_fourier` is the Fourier-space
        expression and `k_sq` represents the squared wave-number.
    """
    if k_sq is None:
        k_sq = sp.Dummy("k_sq", real=True, nonnegative=True)

    perturbations = set(amplitudes)

    def lower(node):
        """Recursively transform expression to Fourier space.

        Args:
            node: SymPy expression node to transform.

        Returns:
            The transformed node in Fourier space.
        """
        # Replace perturbation fields by their Fourier amplitudes.
        if node in amplitudes:
            return amplitudes[node]

        # Fourier symbol of the Laplacian:
        #
        #     ∇² f  ->  -|k|² f_hat
        #
        if isinstance(node, Laplace):
            arg = node.args[0]

            # At this stage, Laplace must not act only on a homogeneous
            # quantity. That should have been simplified to zero earlier.
            if not arg.has(*perturbations):
                msg = (
                    f"Cannot Fourier-transform {node}: it does not depend on a "
                    "perturbation. Simplify the homogeneous state first."
                )
                raise ValueError(msg)

            return -k_sq * lower(arg)

        # For now, reject vector operators in the scalar-only transformer.
        if isinstance(node, (LinearDifferentialOperator, GradientSquared)):
            msg = (
                f"Fourier lowering of {type(node).__name__} requires a "
                "vector/tensor Fourier representation."
            )
            raise NotImplementedError(msg)

        # Recurse through ordinary SymPy expressions.
        if not node.args:
            return node

        return node.func(*(lower(arg) for arg in node.args))

    return sp.expand(lower(expr))


@overload
def dispersion_relation_scalar(
    expression: sp.Basic | str,
    state_hom: dict[str, float],
    *,
    check: bool = True,
    ret_type: Literal["sympy"] = "sympy",
) -> sp.Lambda: ...


@overload
def dispersion_relation_scalar(
    expression: sp.Basic | str,
    state_hom: dict[str, float],
    *,
    check: bool = True,
    ret_type: Literal["numpy"],
) -> np.vectorize: ...


def dispersion_relation_scalar(
    expression: sp.Basic | str,
    state_hom: dict[str, float],
    *,
    check: bool = True,
    ret_type: Literal["sympy", "numpy"] = "sympy",
) -> sp.Lambda | np.vectorize:
    """Calculate the dispersion relation for a scalar perturbation.

    This function linearizes the expression around a homogeneous state, performs a
    Fourier transform, and returns the dispersion relation as a function of the
    squared wave-number.

    Args:
        expression:
            SymPy expression representing the scalar evolution equation.
        state_hom:
            Mapping from the dependent-variable name to its base state value.
        check:
            If True, verify that the expression vanishes at the base state.
        ret_type:
            Return a SymPy Lambda ("sympy") or a NumPy-vectorized callable ("numpy").

    Returns:
        A callable describing the perturbation growth rate as a function of the
        squared wave-number. The callable is a SymPy Lambda or a NumPy-vectorized
        function, depending on `ret_type`.
    """
    # get the input variable
    ((variable_name, _),) = state_hom.items()
    variable = sp.Symbol(variable_name)

    # linearize the expression around the homogeneous stationary state
    expr_lin = linearize_expression(expression, state_hom=state_hom, check=check)

    # do Fourier transform
    variable_k = sp.Dummy(variable_name)
    k_sq = sp.Dummy("k_sq")
    expr_rec = _fourier_lower_scalar(expr_lin, {variable: variable_k}, k_sq=k_sq)

    # check whether the expression is affine in the variable
    try:
        # `EX` permits arbitrary SymPy expressions as coefficients.
        poly = sp.Poly(expr_rec, variable_k, domain="EX")
    except sp.PolynomialError as err:
        msg = f"Linearized expression is not polynomial in {variable_k}: {expr_rec}"
        raise ValueError(msg) from err

    # some consistency checks
    if poly.degree() > 1:
        msg = f"Linearized expression is not linear in {variable_k}: {expr_rec}"
        raise ValueError(msg)
    # The linearization already assumed that the expression vanishes when evaluated at
    # the base state, so the following should always be true
    assert sp.N(poly.nth(0), chop=True).is_zero

    # return the perturbation growth rate
    pert_rate = poly.nth(1)  # linear coefficient
    if ret_type == "sympy":
        return sp.Lambda(k_sq, pert_rate)
    if ret_type == "numpy":
        return np.vectorize(sp.lambdify(k_sq, pert_rate), signature="()->()")
    msg = "Unknown `ret_type`. Use 'sympy' or 'numpy'."
    raise ValueError(msg)


@overload
def dispersion_matrix(
    expressions: list[sp.Basic | str],
    state_hom: dict[str, float],
    *,
    check: bool = True,
    ret_type: Literal["sympy"] = "sympy",
) -> sp.Lambda: ...


@overload
def dispersion_matrix(
    expressions: list[sp.Basic | str],
    state_hom: dict[str, float],
    *,
    check: bool = True,
    ret_type: Literal["numpy"],
) -> np.vectorize: ...


def dispersion_matrix(
    expressions: list[sp.Basic | str],
    state_hom: dict[str, float],
    *,
    check: bool = True,
    ret_type: Literal["sympy", "numpy"] = "sympy",
) -> sp.Lambda | np.vectorize:
    """Calculate the dispersion matrix for multiple coupled equations.

    This function linearizes all expressions around the homogeneous state, performs a
    Fourier transform, and returns the dispersion matrix as a function of the squared
    wave-number.

    Args:
        expressions:
            A sequence of SymPy expressions or a PDEBase instance.
        state_hom:
            Mapping from the dependent-variable name to its base state value.
        check:
            If True, verify that the expression vanishes at the base state.
        ret_type:
            Return a SymPy Lambda ("sympy") or a NumPy-vectorized callable ("numpy").

    Returns:
        A callable describing the dispersion matrix as a function of the squared
        wave-number. The callable is a SymPy Lambda or a NumPy-vectorized function,
        depending on `ret_type`.
    """
    # parse input
    variables, _ = _parse_base_state(state_hom)

    # linearize the expression around the homogeneous stationary state
    exprs_lin = linearize_expressions(expressions, state_hom=state_hom, check=check)

    # do Fourier transform
    variables_k = {v: sp.Dummy(v.name + "_k") for v in variables}
    k_sq = sp.Dummy("k_sq")
    exprs_rec = [
        _fourier_lower_scalar(expr_lin, variables_k, k_sq=k_sq)
        for expr_lin in exprs_lin
    ]

    # assemble the full Jacobian
    jacobian = []
    for expr_rec in exprs_rec:
        jac_line = []
        for var_k in variables_k.values():
            # check whether the expression is affine in the variable
            try:
                # `EX` permits arbitrary SymPy expressions as coefficients.
                poly = sp.Poly(expr_rec, var_k, domain="EX")
            except sp.PolynomialError as err:
                msg = f"Linearized expression is not polynomial in {var_k}: {expr_rec}"
                raise ValueError(msg) from err

            # some consistency checks
            if poly.degree() > 1:
                msg = f"Linearized expression is not linear in {var_k}: {expr_rec}"
                raise ValueError(msg)

            # return the perturbation growth rate
            pert_rate = poly.nth(1)  # linear coefficient
            jac_line.append(pert_rate)
        jacobian.append(jac_line)

    # return the dispersion matrix in the requested format
    if ret_type == "sympy":
        return sp.Lambda(k_sq, sp.Matrix(jacobian))
    if ret_type == "numpy":
        matrix_scalar = sp.lambdify(k_sq, sp.Matrix(jacobian))
        return np.vectorize(matrix_scalar, signature="()->(n, m)")
    msg = "Unknown `ret_type`. Use 'sympy' or 'numpy'."
    raise ValueError(msg)
