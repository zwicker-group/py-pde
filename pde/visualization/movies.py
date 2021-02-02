"""
Functions for creating movies of simulation results


.. autosummary::
   :nosignatures:

   Movie
   movie_scalar
   movie_multiple
   movie
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Dict

from ..storage.base import StorageBase
from ..tools.docstrings import fill_in_docstring
from ..tools.output import display_progress
from .plotting import ScalarFieldPlot, ScaleData


class Movie:
    """Class for creating movies from matplotlib figures using ffmpeg

    Note:
        Internally, this class uses :class:`matplotlib.animation.FFMpegWriter`.
        Note that the `ffmpeg` program needs to be installed in a system path,
        so that `matplotlib` can find it.
    """

    def __init__(
        self, filename: str, framerate: float = 30, dpi: float = None, **kwargs
    ):
        r"""
        Args:
            filename (str):
                The filename where the movie is stored. The suffix of this path
                also determines the default movie codec.
            framerate (float):
                The number of frames per second, which determines how fast the
                movie will appear to run.
            dpi (float):
                The resolution of the resulting movie
            \**kwargs:
                Additional parameters are used to initialize
                :class:`matplotlib.animation.FFMpegWriter`. Here, we can for instance
                set the bit rate of the resulting video using the `bitrate` parameter.
        """
        self.filename = str(filename)
        self.framerate = framerate
        self.dpi = dpi
        self.kwargs = kwargs

        # test whether ffmpeg is available
        from matplotlib.animation import FFMpegWriter

        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                "FFMpegWriter is not available. This is most likely because a suitable "
                "installation of FFMpeg was not found. See ffmpeg.org for how to "
                "install it properly on your system."
            )

        self._writer = None

    @classmethod
    def is_available(cls) -> bool:
        """check whether the movie infrastructure is available

        Returns:
            bool: True if movies can be created
        """
        from matplotlib.animation import FFMpegWriter

        return FFMpegWriter.isAvailable()  # type: ignore

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._end()
        return False

    def _end(self):
        """ clear up temporary things if necessary """
        if self._writer is not None:
            self._writer.finish()
        self._writer = None

    def add_figure(self, fig=None):
        """adds the figure `fig` as a frame to the current movie

        Args:
            fig (:class:`~matplotlib.figures.Figure`):
                The plot figure that is added to the movie
        """
        if fig is None:
            import matplotlib.pyplot as plt

            fig = plt.gcf()

        if self._writer is None:
            # initialize a new writer
            from matplotlib.animation import FFMpegWriter

            self._writer = FFMpegWriter(self.framerate, **self.kwargs)
            self._writer.setup(fig, self.filename, dpi=self.dpi)

        else:
            # update the figure reference on a given writer, since it might have
            # changed from the last call. In particular, this will happen when
            # figures are shown using the `inline` backend.
            self._writer.fig = fig

        # we need to impose a white background to get reasonable antialiasing
        self._writer.grab_frame(facecolor="white")

    def save(self):
        """ convert the recorded images to a movie using ffmpeg """
        self._end()


@fill_in_docstring
def movie_scalar(
    storage: StorageBase,
    filename: str,
    scale: ScaleData = "automatic",
    extras: Dict[str, Any] = None,
    progress: bool = True,
    tight: bool = False,
    show: bool = True,
) -> None:
    """produce a movie for a simulation of a scalar field

    Args:
        storage (:class:`~pde.storage.base.StorageBase`):
            The storage instance that contains all the data for the movie
        filename (str):
            The filename to which the movie is written. The extension determines
            the format used.
        scale (str, float, tuple of float):
            {ARG_PLOT_SCALE}
        extras (dict, optional):
            Additional functions that are evaluated and shown for each time
            step. The key of the dictionary is used as a panel title.
        progress (bool):
            Flag determining whether the progress of making the movie is shown.
        tight (bool):
            Whether to call :func:`matplotlib.pyplot.tight_layout`. This affects
            the layout of all plot elements.
        show (bool):
            Flag determining whether images are shown during making the movie
    """
    quantities = [{"title": "Concentration", "source": None}]
    if extras:
        for key, value in extras.items():
            quantities.append({"title": key, "source": value})

    # initialize the plot with the first data point
    plot = ScalarFieldPlot.from_storage(
        storage, quantities=quantities, scale=scale, tight=tight, show=show
    )
    # make the full movie
    plot.make_movie(storage, filename, progress=progress)


@fill_in_docstring
def movie_multiple(
    storage: StorageBase,
    filename: str,
    quantities=None,
    scale: ScaleData = "automatic",
    progress: bool = True,
) -> None:
    """produce a movie for a simulation with n components

    Args:
        storage (:class:`~pde.storage.base.StorageBase`):
            The storage instance that contains all the data for the movie
        filename (str):
            The filename to which the movie is written. The extension determines
            the format used.
        quantities:
            {ARG_PLOT_QUANTITIES}
        scale (str, float, tuple of float):
            {ARG_PLOT_SCALE}
        progress (bool):
            Flag determining whether the progress of making the movie is shown.
    """
    plot = ScalarFieldPlot.from_storage(storage, quantities=quantities, scale=scale)
    plot.make_movie(storage, filename, progress=progress)


def movie(
    storage: StorageBase,
    filename: str,
    progress: bool = True,
    dpi: float = 150,
    show_time: bool = True,
    plot_args: Dict[str, Any] = None,
) -> None:
    """produce a movie by simply plotting each frame

    Args:
        storage (:class:`~pde.storage.base.StorageBase`):
            The storage instance that contains all the data for the movie
        filename (str):
            The filename to which the movie is written. The extension determines
            the format used.
        progress (bool):
            Flag determining whether the progress of making the movie is shown.
        dpi (float):
            Resolution of the movie
        show_time (bool):
            Whether to show the simulation time in the movie
        plot_args (dict):
            Additional arguments for the function plotting the state
    """
    if plot_args is None:
        plot_args = {}

    # create the iterator over the data
    field_iter = display_progress(storage.items(), total=len(storage), enabled=progress)

    ref, fig = None, None
    with Movie(filename=filename, dpi=dpi) as movie:
        # iterate over all frames
        for t, field in field_iter:
            if ref is None:
                # create the actual figure in the first frame
                ref = field.plot(action="create", **plot_args)
                # obtain the matplotlib figure from the returned reference
                try:
                    fig = ref.ax.figure
                except AttributeError:
                    fig = ref[0].ax.figure
                if show_time:
                    title = fig.suptitle("Time %g" % t)

            else:
                # update the data in the figure
                field._update_plot(ref)
                if show_time:
                    title.set_text("Time %g" % t)

            # add the current matplotlib figure to the movie
            movie.add_figure(fig)
