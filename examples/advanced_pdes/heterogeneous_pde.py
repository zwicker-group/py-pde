r"""
Heterogeneous PDE
=================

This example loads an example image and uses it as the source field for a simple
reaction-diffusion equation.
"""

from pathlib import Path

from pde import PDE, ScalarField

# load a field
img_path = Path(__file__).parents[2] / "docs" / "source" / "_images" / "logo_small.png"
background = ScalarField.from_image(img_path)
state = ScalarField(background.grid)  # generate initial condition

# define the pde
eq = PDE({"c": "laplace(c) + 0.2 * source - 0.1 * c"}, consts={"source": background})
result = eq.solve(state, t_range=100, adaptive=True)
result.plot()
