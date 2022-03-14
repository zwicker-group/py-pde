"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import matplotlib.testing.compare
import numpy as np
import pytest

from pde.tools.plotting import add_scaled_colorbar, plot_on_axes, plot_on_figure


def test_plot_on_axes(tmp_path):
    """test the plot_on_axes decorator"""

    @plot_on_axes
    def plot(ax):
        ax.plot([0, 1], [0, 1])

    path = tmp_path / "test.png"
    plot(title="Test", filename=path)
    assert path.stat().st_size > 0


def test_plot_on_figure(tmp_path):
    """test the plot_on_figure decorator"""

    @plot_on_figure
    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot([0, 1], [0, 1])
        ax2.plot([0, 1], [0, 1])

    path = tmp_path / "test.png"
    plot(title="Test", filename=path)
    assert path.stat().st_size > 0


@pytest.mark.interactive
def test_plot_colorbar(tmp_path):
    """test the plot_on_axes decorator"""
    data = np.random.randn(3, 3)

    # do not specify axis
    img = plt.imshow(data)
    add_scaled_colorbar(img, label="Label")
    plt.savefig(tmp_path / "img1.png")
    plt.clf()

    # specify axis explicitly
    ax = plt.gca()
    img = ax.imshow(data)
    add_scaled_colorbar(img, ax=ax, label="Label")
    plt.savefig(tmp_path / "img2.png")

    # compare the two results
    cmp = matplotlib.testing.compare.compare_images(
        str(tmp_path / "img1.png"), str(tmp_path / "img2.png"), tol=0.1
    )
    assert cmp is None
