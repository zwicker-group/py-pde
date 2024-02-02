"""
Module defining classes for tracking results from simulations.

The trackers defined in this module are:

.. autosummary::
   :nosignatures:

   CallbackTracker
   ProgressTracker
   PrintTracker
   PlotTracker
   LivePlotTracker
   DataTracker
   SteadyStateTracker
   RuntimeTracker
   ConsistencyTracker
   MaterialConservationTracker

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import math
import os.path
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable

import numpy as np

from ..fields import FieldCollection
from ..fields.base import FieldBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import module_available
from ..tools.output import get_progress_bar_class
from ..tools.parse_duration import parse_duration
from ..tools.typing import Real
from ..visualization.movies import Movie
from .base import FinishedSimulation, InfoDict, TrackerBase
from .interrupts import InterruptData, RealtimeInterrupts

if TYPE_CHECKING:
    import pandas


class CallbackTracker(TrackerBase):
    """Tracker calling a function periodically

    Example:
        The callback tracker can be used to check for conditions during the simulation:

        .. code-block:: python

            def check_simulation(state, time):
                if state.integral < 0:
                    raise StopIteration

            tracker = CallbackTracker(check_simulation, interval="0:10")

        Adding :code:`tracker` to the simulation will perform a check every 10 real time
        seconds. If the integral of the entire state falls below zero, the simulation
        will be aborted.
    """

    @fill_in_docstring
    def __init__(
        self,
        func: Callable,
        interrupts: InterruptData = 1,
        *,
        interval=None,
    ):
        """
        Args:
            func:
                The function to call periodically. The function signature should be
                `(state)` or `(state, time)`, where `state` contains the current state
                as an instance of :class:`~pde.fields.base.FieldBase` and `time` is a
                float value indicating the current time. Note that only a view of the
                state is supplied, implying that a copy needs to be made if the data
                should be stored. The function can thus adjust the state by modifying it
                in-place and it can even interrupt the simulation by raising the special
                exception :class:`StopIteration`.
            interrupts:
                {ARG_TRACKER_INTERRUPT}
        """
        super().__init__(interrupts=interrupts, interval=interval)
        self._callback = func
        self._num_args = len(inspect.signature(func).parameters)
        if not 0 < self._num_args < 3:
            raise ValueError(
                "`func` must be a function accepting one or two arguments, not "
                f"{self._num_args}"
            )

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if self._num_args == 1:
            self._callback(field)
        else:
            self._callback(field, t)


class ProgressTracker(TrackerBase):
    """Tracker showing the progress of the simulation"""

    name = "progress"

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData | None = None,
        *,
        fancy: bool = True,
        ndigits: int = 5,
        leave: bool = True,
        interval=None,
    ):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
                The default value `None` updates the progress bar approximately every
                (real) second.
            fancy (bool):
                Flag determining whether a fancy progress bar should be used in jupyter
                notebooks (if :mod:`ipywidgets` is installed)
            ndigits (int):
                The number of digits after the decimal point that are shown maximally.
            leave (bool):
                Whether to leave the progress bar after the simulation has finished
                (default: True)
        """
        if interrupts is None:
            interrupts = RealtimeInterrupts(duration=1)  # print every second by default

        super().__init__(interrupts=interrupts, interval=interval)
        self.fancy = fancy
        self.ndigits = ndigits
        self.leave = leave

    def initialize(self, field: FieldBase, info: InfoDict | None = None) -> float:
        """initialize the tracker with information about the simulation

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        result = super().initialize(field, info)

        # get solver information
        controller_info = {} if info is None else info.get("controller", {})

        # initialize the progress bar
        pb_cls = get_progress_bar_class(self.fancy)
        self.progress_bar = pb_cls(
            total=controller_info.get("t_end"),
            initial=controller_info.get("t_start", 0),
            leave=self.leave,
        )
        self.progress_bar.set_description("Initializing")

        return result

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        # show an update
        if self.progress_bar.total:
            t_new = min(t, self.progress_bar.total)
        else:
            t_new = t
        self.progress_bar.n = round(t_new, self.ndigits)
        self.progress_bar.set_description("")

    def finalize(self, info: InfoDict | None = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        self.progress_bar.set_description("")

        # limit progress bar to 100%
        controller_info = {} if info is None else info.get("controller", {})
        t_final = controller_info.get("t_final", -math.inf)
        t_end = controller_info.get("t_end", -math.inf)
        if t_final >= t_end and self.progress_bar.total:
            self.progress_bar.n = self.progress_bar.total
            self.progress_bar.refresh()

        if (
            controller_info.get("successful", False)
            and self.leave
            and hasattr(self.progress_bar, "sp")
        ):
            # show progress bar in green if simulation was successful. We
            # need to overwrite the default behavior (and disable the
            # progress bar) since reaching steady state means the simulation
            # was successful even though it did not reach t_final
            try:
                self.progress_bar.sp(bar_style="success")
            except TypeError:
                self.progress_bar.close()
            else:
                self.progress_bar.disable = True
        else:
            self.progress_bar.close()

    def __del__(self):
        if hasattr(self, "progress_bar") and not self.progress_bar.disable:
            self.progress_bar.close()


class PrintTracker(TrackerBase):
    """Tracker printing data to a stream (default: stdout)"""

    name = "print"

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = 1,
        stream: IO[str] = sys.stdout,
        *,
        interval=None,
    ):
        """

        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            stream:
                The stream used for printing
        """
        super().__init__(interrupts=interrupts, interval=interval)
        self.stream = stream

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        data = f"c={field.data.mean():.3g}±{field.data.std():.3g}"

        self.stream.write(f"t={t:g}, {data}\n")
        self.stream.flush()


class PlotTracker(TrackerBase):
    """Tracker plotting data on screen, to files, or writes a movie

    This tracker can be used to create movies from simulations or to simply update a
    single image file on the fly (i.e. to monitor simulations running on a cluster). The
    default values of this tracker are chosen with regular output to a file in mind.

    Example:
        To create a movie while running the simulation, you can use

        .. code-block:: python

            movie_tracker = PlotTracker(interval=10, movie="my_movie.mp4")
            eq.solve(..., tracker=movie_tracker)

        This will create the file `my_movie.mp4` during the simulation. Note that you
        can display the frames interactively by setting :code:`show=True`.
    """

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = 1,
        *,
        title: str | Callable = "Time: {time:g}",
        output_file: str | None = None,
        movie: str | Path | Movie | None = None,
        show: bool | None = None,
        tight_layout: bool = False,
        max_fps: float = math.inf,
        plot_args: dict[str, Any] | None = None,
        interval=None,
    ):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            title (str or callable):
                Title text of the figure. If this is a string, it is shown with a
                potential placeholder named `time` being replaced by the current
                simulation time. Conversely, if `title` is a function, it is called with
                the current state and the time as arguments. This function is expected
                to return a string.
            output_file (str, optional):
                Specifies a single image file, which is updated periodically, so that
                the progress can be monitored (e.g. on a compute cluster)
            movie (str or :class:`~pde.visualization.movies.Movie`):
                Create a movie. If a filename is given, all frames are written to this
                file in the format deduced from the extension after the simulation ran.
                If a :class:`~pde.visualization.movies.Movie` is supplied, frames are
                appended to the instance.
            show (bool, optional):
                Determines whether the plot is shown while the simulation is running. If
                set to `None`, the images are only shown if neither `output_file` nor
                `movie` is set, otherwise they are kept hidden. Note that showing the
                plot can slow down a simulation severely.
            tight_layout (bool):
                Determines whether :func:`~matplotlib.pyplot.tight_layout` is used.
            max_fps (float):
                Determines the maximal rate (frames per second) at which the plots are
                updated in real time during the simulation. Some plots are skipped if
                the tracker receives data at a higher rate. A larger value (e.g.,
                `math.inf`) can be used to ensure every frame is drawn, which might
                penalizes the overall performance.
            plot_args (dict):
                Extra arguments supplied to the plot call. For example, this can be used
                to specify axes ranges when a single panel is shown. For instance, the
                value :code:`{'ax_style': {'ylim': (0, 1)}}` enforces the y-axis to lie
                between 0 and 1.

        Note:
            If an instance of :class:`~pde.visualization.movies.Movie` is given as the
            `movie` argument, it can happen that the movie is not written to the file
            when the simulation ends. This is because, the movie could still be extended
            by appending frames. To write the movie to a file call its
            :meth:`~pde.visualization.movies.Movie.save` method. Beside adding frames
            before and after the simulation, an explicit movie object can also be used
            to adjust the output. For instance, the following example code creates a
            movie with a framerate of 15, a resolution of 200 dpi, and a bitrate of 6000
            kilobits per second:

            .. code-block:: python

                movie = Movie("movie.mp4", framerate=15, dpi=200, bitrate=6000)
                eq.solve(..., tracker=PlotTracker(1, movie=movie))
                movie.save()
        """
        from ..visualization.movies import Movie  # @Reimport

        # initialize the tracker
        super().__init__(interrupts=interrupts, interval=interval)
        self.title = title
        self.output_file = output_file
        self.tight_layout = tight_layout
        self.max_fps = max_fps

        self.plot_args = {} if plot_args is None else plot_args.copy()
        # make sure the plot is only create and not shown since the context
        # handles showing the plot itself
        self.plot_args["action"] = "none"

        # initialize the movie class
        if movie is None:
            self.movie: Movie | None = None
            self._save_movie = False

        elif isinstance(movie, Movie):
            self.movie = movie
            self._save_movie = False

        elif isinstance(movie, (str, Path)):
            self.movie = Movie(filename=str(movie))
            self._save_movie = True

        else:
            raise TypeError(f"Unknown type of `movie`: {movie.__class__.__name__}")

        # determine whether to show the images interactively
        self._write_images = self._save_movie or self.output_file
        if show is None:
            self.show = not self._write_images
        else:
            self.show = show

    def initialize(self, state: FieldBase, info: InfoDict | None = None) -> float:
        """initialize the tracker with information about the simulation

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        # initialize the plotting context
        import matplotlib.pyplot as plt

        from ..tools.plotting import get_plotting_context

        self._context = get_plotting_context(title="Initializing...", show=self.show)

        # do the actual plotting
        with self._context:
            self._plot_reference = state.plot(**self.plot_args)
            if self.tight_layout:
                plt.gcf().tight_layout()

        if self._context.supports_update:
            # the context supports reusing figures
            if hasattr(state.plot, "update_method"):
                # the plotting method supports updating the plot
                if state.plot.update_method is None:
                    if state.plot.mpl_class == "axes":  # type: ignore
                        self._update_method = "update_ax"
                    elif state.plot.mpl_class == "figure":  # type: ignore
                        self._update_method = "update_fig"
                    else:
                        mpl_class = state.plot.mpl_class  # type: ignore
                        raise RuntimeError(
                            f"Unknown mpl_class on plot method: {mpl_class}"
                        )
                else:
                    self._update_method = "update_data"
            else:
                raise RuntimeError(
                    "PlotTracker does not  work since the state of type "
                    f"{state.__class__.__name__} does not use the plot protocol of "
                    "`pde.tools.plotting`."
                )
        else:
            self._update_method = "replot"

        self._logger.info(f'Update method: "{self._update_method}"')
        self._last_update = time.monotonic()
        return super().initialize(state, info=info)

    def handle(self, state: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            state (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        import matplotlib.pyplot as plt

        if not self._write_images:
            # check whether we can skip this image
            time_passed = time.monotonic() - self._last_update
            if time_passed < 1 / self.max_fps:
                return  # we just recently updated the image

        if callable(self.title):
            self._context.title = str(self.title(state, t))
        else:
            self._context.title = self.title.format(time=t)

        # update the plot in the correct plotting context
        with self._context:
            if self._update_method == "update_data":
                # the state supports updating the plot data
                update_func = getattr(state, state.plot.update_method)  # type: ignore
                update_func(self._plot_reference)

            elif self._update_method == "update_ax":
                fig = self._context.fig
                fig.clf()  # type: ignore
                ax = fig.add_subplot(1, 1, 1)  # type: ignore
                state.plot(ax=ax, **self.plot_args)

            elif self._update_method == "update_fig":
                fig = self._context.fig
                fig.clf()  # type: ignore
                state.plot(fig=fig, **self.plot_args)
                if self.tight_layout:
                    plt.gcf().tight_layout()

            elif self._update_method == "replot":
                state.plot(**self.plot_args)
                if self.tight_layout:
                    plt.gcf().tight_layout()

            else:
                raise RuntimeError(f"Unknown update method `{self._update_method}`")

        if self.output_file and self._context.fig is not None:
            self._context.fig.savefig(self.output_file)
        if self.movie:
            self.movie.add_figure(self._context.fig)

        self._last_update = time.monotonic()

    def finalize(self, info: InfoDict | None = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        if self._save_movie:
            # write out movie file
            self.movie.save()  # type: ignore
            # end recording the movie (e.g. delete temporary files)
            self.movie._end()  # type: ignore

        if not self.show:
            self._context.close()


class LivePlotTracker(PlotTracker):
    """PlotTracker with defaults for live plotting

    The only difference to :class:`PlotTracker` are the changed default values, where
    output is by default shown on screen and the `interval` is set something more
    suitable for interactive plotting. In particular, this tracker can be enabled by
    simply listing 'plot' as a tracker.
    """

    name = "plot"

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = "0:03",
        *,
        show: bool = True,
        max_fps: float = 2,
        interval=None,
        **kwargs,
    ):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            title (str):
                Text to show in the title. The current time point will be
                appended to this text, so include a space for optimal results.
            output_file (str, optional):
                Specifies a single image file, which is updated periodically, so
                that the progress can be monitored (e.g. on a compute cluster)
            output_folder (str, optional):
                Specifies a folder to which all images are written. The files
                will have names with increasing numbers.
            movie_file (str, optional):
                Specifies a filename to which a movie of all the frames is
                written after the simulation.
            show (bool, optional):
                Determines whether the plot is shown while the simulation is
                running. If `False`, the files are created in the background.
                This option can slow down a simulation severely.
            max_fps (float):
                Determines the maximal rate (frames per second) at which the plots are
                updated. Some plots are skipped if the tracker receives data at a higher
                rate. A larger value (e.g., `math.inf`) can be used to ensure every
                frame is drawn, which might penalizes the overall performance.
            plot_args (dict):
                Extra arguments supplied to the plot call. For example, this can
                be used to specify axes ranges when a single panel is shown. For
                instance, the value `{'ax_style': {'ylim': (0, 1)}}` enforces
                the y-axis to lie between 0 and 1.
        """
        super().__init__(
            interrupts=interrupts,
            interval=interval,
            show=show,
            max_fps=max_fps,
            **kwargs,
        )


class DataTracker(CallbackTracker):
    """Tracker storing custom data obtained by calling a function

    Example:
        The data tracker can be used to gather statistics during the run

        .. code-block:: python

            def get_statistics(state, time):
                return {"mean": state.data.mean(), "variance": state.data.var()}

            data_tracker = DataTracker(get_statistics, interval=10)

        Adding :code:`data_tracker` to the simulation will gather the statistics every
        10 time units. After the simulation, the final result will be accessable via the
        :attr:`data` attribute or conveniently as a pandas from the :attr:`dataframe`
        attribute.

    Attributes:
        times (list):
            The time points at which the data is stored
        data (list):
            The actually stored data, which is a list of the objects returned by
            the callback function.
    """

    @fill_in_docstring
    def __init__(
        self,
        func: Callable,
        interrupts: InterruptData = 1,
        *,
        filename: str | None = None,
        interval=None,
    ):
        """
        Args:
            func:
                The function to call periodically. The function signature
                should be `(state)` or `(state, time)`, where `state` contains
                the current state as an instance of
                :class:`~pde.fields.FieldBase` and `time` is a
                float value indicating the current time. Note that only a view
                of the state is supplied, implying that a copy needs to be made
                if the data should be stored.
                Typical return values of the function are either a single
                number, a numpy array, a list of number, or a dictionary to
                return multiple numbers with assigned labels.
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            filename (str):
                A path to a file to which the data is written at the end of the
                tracking. The data format will be determined by the extension
                of the filename. '.pickle' indicates a python pickle file
                storing a tuple `(self.times, self.data)`, whereas any other
                data format requires :mod:`pandas`.
        """
        super().__init__(func=func, interrupts=interrupts, interval=interval)
        self.filename = filename
        self.times: list[float] = []
        self.data: list[Any] = []

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        self.times.append(t)
        if self._num_args == 1:
            self.data.append(self._callback(field))
        else:
            self.data.append(self._callback(field, t))

    def finalize(self, info: InfoDict | None = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        if self.filename:
            self.to_file(self.filename)

    @property
    def dataframe(self) -> pandas.DataFrame:
        """:class:`pandas.DataFrame`: the data in a dataframe

        If `func` returns a dictionary, the keys are used as column names.
        Otherwise, the returned data is enumerated starting with '0'. In any
        case the time point at which the data was recorded is stored in the
        column 'time'.
        """
        import pandas as pd

        df = pd.DataFrame(self.data)
        # insert the times and use them as an index
        df.insert(0, "time", self.times)
        return df

    def to_file(self, filename: str, **kwargs):
        r"""store data in a file

        The extension of the filename determines what format is being used. For
        instance, '.pickle' indicates a python pickle file storing a tuple
        `(self.times, self.data)`, whereas any other data format requires
        :mod:`pandas`. Supported formats include 'csv', 'json'.

        Args:
            filename (str):
                Path where the data is stored
            \**kwargs:
                Additional parameters may be supported for some formats
        """
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".pickle":
            import pickle

            with open(filename, "wb") as fp:
                pickle.dump((self.times, self.data), fp, **kwargs)

        elif extension == ".csv":
            self.dataframe.to_csv(filename, **kwargs)
        elif extension == ".json":
            self.dataframe.to_json(filename, **kwargs)
        elif extension in {".xls", ".xlsx"}:
            self.dataframe.to_excel(filename, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension `{extension}`")


class SteadyStateTracker(TrackerBase):
    """Tracker aborting the simulation once steady state is reached

    Steady state is obtained when the state does not change anymore, i.e., when the
    evolution rate is close to zero. If the argument `evolution_rate` is specified, it
    is used to calculate the evolution rate directly. If it is omitted, the evolution
    rate is estaimted by comparing the current state `cur` to the state `prev` at the
    previous time step. In both cases, convergence is assumed when the absolute value of
    the evolution rate falls below :code:`atol + rtol * cur` for all points. Here,
    `atol` and `rtol` denote absolute and relative tolerances, respectively.
    """

    name = "steady_state"

    progress_bar_format = (
        "Convergence: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"
    )
    """ determines the format of the progress bar shown when `progress = True` """

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData | None = None,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        *,
        progress: bool = False,
        evolution_rate: Callable[[np.ndarray, float], np.ndarray] | None = None,
        interval=None,
    ):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
                The default value `None` checks for the steady state approximately every
                (real) second.
            atol (float):
                Absolute tolerance that must be reached to abort the simulation
            rtol (float):
                Relative tolerance that must be reached to abort the simulation
            progress (bool):
                Flag indicating whether the progress towards convergence is shown
                graphically during the simulation
            evolution_rate (callable):
                Function to evaluate the current evolution rate. If omitted, the
                evolution rate is estimate from the change in the state variable, which
                can be less accurate. A suitable form of the function is returned by
                `eq.make_pde_rhs(state)` when `eq` is the PDE class.
        """
        if interrupts is None:
            interrupts = RealtimeInterrupts(duration=1)
        super().__init__(interrupts=interrupts, interval=interval)
        self.atol = atol
        self.rtol = rtol
        self.evolution_rate = evolution_rate
        self.progress = progress and module_available("tqdm")

        self._progress_bar: Any = None
        self._pbar_offset: float = 0  # required for calculating progress
        self._last_data: np.ndarray | None = None
        self._last_time: float | None = None
        self._best_rate_max: np.ndarray | None = None

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        finite = np.isfinite(field.data)  # ignore infinite and nan data

        # determine the maximal rate of change
        if self.evolution_rate is not None:
            # use the evolution_rate function to calculate the rate
            evolution_rate = self.evolution_rate(field.data, t)[finite]

        elif self._last_data is not None:
            # get evolution rate from the difference of current state to previous one
            diff = self._last_data[finite] - field.data[finite]
            evolution_rate = diff / (t - self._last_time)  # type: ignore

            # save current data for next comparison
            self._last_data[:] = field.data
            self._last_time = t

        else:
            # create storage for the data
            self._last_data = field.data.copy()
            self._last_time = t
            return  # do not output anything since we don't know `evolution_rate` yet

        # calculate the maximal deviation of the evolution rate from zero, subtracting
        # the relative tolerance with respect to the field values
        rate_abs = np.abs(evolution_rate) - self.rtol * np.abs(field.data[finite])
        rate_abs_max = np.max(rate_abs)

        # check wether the simulation has converged
        if rate_abs_max <= self.atol:
            if self.progress and self._progress_bar is not None:
                # advance progress bar to 100%
                self._progress_bar.n = self._pbar_offset - np.log10(self.atol)
                try:
                    self._progress_bar.disp(bar_style="success", check_delay=False)
                except (TypeError, AttributeError):
                    self._progress_bar.close()
            raise FinishedSimulation("Reached stationary state")

        if self.progress:
            # show progress of the convergence
            if self._best_rate_max is None:
                # initialize the progress bar
                pb_cls = get_progress_bar_class()
                self._pbar_offset = np.log10(rate_abs_max)
                self._progress_bar = pb_cls(
                    total=self._pbar_offset - np.log10(self.atol),
                    bar_format=self.progress_bar_format,
                )
                self._best_rate_max = rate_abs_max

            elif rate_abs_max < self._best_rate_max:
                # update progress bar if simulation got closer to convergence
                self._progress_bar.n = self._pbar_offset - np.log10(rate_abs_max)
                self._progress_bar.refresh()
                self._best_rate_max = rate_abs_max


class RuntimeTracker(TrackerBase):
    """Tracker interrupting the simulation once a duration has passed"""

    @fill_in_docstring
    def __init__(
        self, max_runtime: Real | str, interrupts: InterruptData = 1, *, interval=None
    ):
        """
        Args:
            max_runtime (float or str):
                The maximal runtime of the simulation. If the runtime is exceeded, the
                simulation is interrupted. Values can be either given as a number
                (interpreted as seconds) or as a string, which is then parsed using the
                function :func:`~pde.tools.parse_duration.parse_duration`.
            interrupts:
                {ARG_TRACKER_INTERRUPT}
        """
        super().__init__(interrupts=interrupts, interval=interval)

        try:
            self.max_runtime = float(max_runtime)
        except ValueError:
            td = parse_duration(str(max_runtime))
            self.max_runtime = td.total_seconds()

    def initialize(self, field: FieldBase, info: InfoDict | None = None) -> float:
        """
        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        self.max_time = time.monotonic() + self.max_runtime
        return super().initialize(field, info)

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if time.monotonic() > self.max_time:
            dt = timedelta(seconds=self.max_runtime)
            raise FinishedSimulation(f"Reached maximal runtime of {str(dt)}")


class ConsistencyTracker(TrackerBase):
    """Tracker interrupting the simulation when the state is not finite"""

    name = "consistency"

    @fill_in_docstring
    def __init__(self, interrupts: InterruptData | None = None, *, interval=None):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
                The default value `None` checks for consistency approximately every
                (real) second.
        """
        if interrupts is None:
            interrupts = RealtimeInterrupts(duration=1)
        super().__init__(interrupts=interrupts, interval=interval)

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if not np.all(np.isfinite(field.data)):
            raise StopIteration("Field was not finite")


class MaterialConservationTracker(TrackerBase):
    """Tracking interrupting the simulation when material conservation is broken"""

    name = "material_conservation"

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = 1,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        *,
        interval=None,
    ):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            atol (float):
                Absolute tolerance for amount deviations
            rtol (float):
                Relative tolerance for amount deviations
        """
        super().__init__(interrupts=interrupts, interval=interval)
        self.atol = atol
        self.rtol = rtol

    def initialize(self, field: FieldBase, info: InfoDict | None = None) -> float:
        """
        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        if isinstance(field, FieldCollection):
            self._reference = np.array([f.magnitude for f in field])
        else:
            self._reference = field.magnitude  # type: ignore

        return super().initialize(field, info)

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if isinstance(field, FieldCollection):
            mags = np.array([f.magnitude for f in field])
        else:
            mags = field.magnitude  # type: ignore

        c = np.isclose(mags, self._reference, rtol=self.rtol, atol=self.atol)
        if not np.all(c):
            if isinstance(field, FieldCollection):
                msg = f"Material of field {np.flatnonzero(~c)} is not conserved"
            else:
                msg = f"Material is not conserved"
            raise StopIteration(msg)


__all__ = [
    "CallbackTracker",
    "ProgressTracker",
    "PrintTracker",
    "PlotTracker",
    "LivePlotTracker",
    "DataTracker",
    "SteadyStateTracker",
    "RuntimeTracker",
    "ConsistencyTracker",
    "MaterialConservationTracker",
]
