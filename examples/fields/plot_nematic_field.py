r"""
Plotting a nematic field
========================

This example shows how to visualize a nematic tensor field
:math:`\boldsymbol Q = S(\boldsymbol n \otimes \boldsymbol n - \mathbb{1}/2)`.
The field shown here contains a :math:`+\frac12` defect at the origin, where the
director :math:`\boldsymbol n` rotates by :math:`\pi` around the defect and the nematic
order :math:`S` vanishes.
"""

from pde import CartesianGrid, Tensor2Field

grid = CartesianGrid([[-2, 2], [-2, 2]], 24)

order = "tanh(2 * sqrt(x**2 + y**2))"  # nematic order vanishing at the defect
q_xx = f"0.5 * {order} * x / sqrt(x**2 + y**2)"
q_xy = f"0.5 * {order} * y / sqrt(x**2 + y**2)"
field = Tensor2Field.from_expression(grid, [[q_xx, q_xy], [q_xy, f"-({q_xx})"]])

field.plot(kind="nematic", title="Nematic field with a $+1/2$ defect")
