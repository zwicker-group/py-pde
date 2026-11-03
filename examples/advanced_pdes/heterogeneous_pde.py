r"""
Heterogeneous PDE
=================

This example loads an example image and uses it as the source field for a simple
reaction-diffusion equation.
"""

import inspect
from pathlib import Path

from pde import PDE, ScalarField

# load a field relative to the current file
package_path = Path(inspect.getfile(lambda: None)).parents[2]
img_path = package_path / "docs" / "source" / "_images" / "logo_small.png"

background = ScalarField.from_image(img_path)  # create source field from image
state = ScalarField(background.grid)  # generate initial condition

# define the pde
eq = PDE({"c": "laplace(c) + 0.2 * source - 0.1 * c"}, consts={"source": background})
result = eq.solve(state, t_range=100, adaptive=True)
result.plot()
