"""
Finite differences approximation
================================

This example displays various finite difference (FD) approximations of derivatives of
simple harmonic function.
"""

import matplotlib.pyplot as plt
import numpy as np

from pde import CartesianGrid, ScalarField
from pde.tools.expressions import evaluate

# create two grids with different resolution to emphasize finite difference approximation
grid_fine = CartesianGrid([(0, 2 * np.pi)], 256, periodic=True)
grid_coarse = CartesianGrid([(0, 2 * np.pi)], 10, periodic=True)

# create figure to present plots of the derivative
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

# plot first derivatives of sin(x)
f = ScalarField.from_expression(grid_coarse, "sin(x)")
f_grad = f.gradient("periodic")  # first derivative (from gradient vector field)
ScalarField.from_expression(grid_fine, "cos(x)").plot(
    ax=axes[0, 0], label="Expected f'"
)
f_grad.plot(ax=axes[0, 0], label="FD grad(f)", ls="", marker="o")
plt.legend(frameon=True)
plt.ylabel("")
plt.xlabel("")
plt.title(r"First derivative of $f(x) = \sin(x)$")

# plot second derivatives of sin(x)
f_laplace = f.laplace("periodic")  # second derivative
f_grad2 = f_grad.divergence("periodic")  # second derivative using composition
ScalarField.from_expression(grid_fine, "-sin(x)").plot(
    ax=axes[0, 1], label="Expected f''"
)
f_laplace.plot(ax=axes[0, 1], label="FD laplace(f)", ls="", marker="o")
f_grad2.plot(ax=axes[0, 1], label="FD div(grad(f))", ls="", marker="o")
plt.legend(frameon=True)
plt.xlabel("")
plt.title(r"Second derivative of $f(x) = \sin(x)$")

# plot first derivatives of sin(x)**2
g_fine = ScalarField.from_expression(grid_fine, "sin(x)**2")
g = g_fine.interpolate_to_grid(grid_coarse)
expected = evaluate("2 * cos(x) * sin(x)", {"g": g_fine})
fd_1 = evaluate("d_dx(g)", {"g": g})  # first derivative (from directional derivative)
expected.plot(ax=axes[1, 0], label="Expected g'")
fd_1.plot(ax=axes[1, 0], label="FD grad(g)", ls="", marker="o")
plt.legend(frameon=True)
plt.title(r"First derivative of $g(x) = \sin(x)^2$")

# plot second derivatives of sin(x)**2
expected = evaluate("2 * cos(2 * x)", {"g": g_fine})
fd_2 = evaluate("d2_dx2(g)", {"g": g})  # second derivative
fd_11 = evaluate("d_dx(d_dx(g))", {"g": g})  # composition of first derivatives
expected.plot(ax=axes[1, 1], label="Expected g''")
fd_2.plot(ax=axes[1, 1], label="FD laplace(g)", ls="", marker="o")
fd_11.plot(ax=axes[1, 1], label="FD div(grad(g))", ls="", marker="o")
plt.legend(frameon=True)
plt.title(r"Second derivative of $g(x) = \sin(x)^2$")

# finalize plot
plt.tight_layout()
plt.show()
