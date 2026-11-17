r"""
Linear stability Analysis
=========================

This example performs a linear stability analysis of a simple Turing system.
"""

import matplotlib.pyplot as plt
import numpy as np

from pde.tools.linear_stability import dispersion_matrix

# define the expressions of the PDE
rhs = ["laplace(u) + 0.1 - u + u**2 * v", "10 * laplace(v) + 0.9 - u**2 * v"]

# determine the dispersion matrix
matrix = dispersion_matrix(rhs, {"u": 1, "v": 0.9}, ret_type="numpy")

# determine the perturbation growth rates as a function of wave vector k
ks = np.linspace(0, 1)
omegas = np.linalg.eigvals(matrix(ks**2))

# plot the dispersion relation
l1 = plt.plot(ks, np.real(omegas), "C0-")
l2 = plt.plot(ks, np.imag(omegas), "C0:")
plt.axhline(0, color="k")
plt.xlim(0, 1)
plt.ylim(-1.05, 0.45)
plt.xlabel("Wave vector")
plt.ylabel("Perturbation growth rate")
plt.legend([l1[0], l2[0]], ["Real part", "Imaginary part"])
plt.show()
