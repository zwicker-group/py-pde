"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools.plotting import plot_on_axes, plot_on_figure


def test_plot_on_axes(tmp_path):
    """ test the plot_on_axes decorator """

    @plot_on_axes
    def plot(ax):
        ax.plot([0, 1], [0, 1])

    path = tmp_path / "test.png"
    plot(title="Test", filename=path)
    assert path.stat().st_size > 0


def test_plot_on_figure(tmp_path):
    """ test the plot_on_figure decorator """

    @plot_on_figure
    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot([0, 1], [0, 1])
        ax2.plot([0, 1], [0, 1])

    path = tmp_path / "test.png"
    plot(title="Test", filename=path)
    assert path.stat().st_size > 0
