"""
Functions and classes for plotting simulation data

.. autosummary::
   :nosignatures:

   ScalarFieldPlot
   plot_magnitudes
   plot_kymograph
   plot_kymographs
   plot_interactive
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..fields import FieldCollection
from ..fields.base import DataFieldBase, FieldBase
from ..storage.base import StorageBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import module_available
from ..tools.output import display_progress
from ..tools.plotting import (
    PlotReference,
    napari_add_layers,
    napari_viewer,
    plot_on_axes,
    plot_on_figure,
)

_logger = logging.getLogger(__name__)
ScaleData = Union[str, float, Tuple[float, float]]


def _add_horizontal_colorbar(im, ax, num_loc: int = 5) -> None:
    """adds a horizontal colorbar for image `im` to the axis `ax`

    Args:
        im: The result of calling :func:`matplotlib.pyplot.imshow`
        ax: The matplotlib axes to which the colorbar is added
        num_loc (int): Number of ticks
    """
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.1, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.locator = MaxNLocator(num_loc)
    cb.update_ticks()


def extract_field(
    fields: FieldBase,
    source: Union[None, int, Callable] = None,
    check_rank: Optional[int] = None,
) -> DataFieldBase:
    """Extracts a single field from a possible collection.

    Args:
        fields (:class:`~pde.fields.FieldBase`):
            The field from which data is extracted
        source (int or callable, optional):
            Determines how a field is extracted from `fields`. If `None`,
            `fields` is passed as is, assuming it is already a scalar field.
            This works for the simple, standard case where only a single
            :class:`~pde.fields.scalar.ScalarField` is treated. Alternatively,
            `source` can be an integer, indicating which field is extracted from
            an instance of :class:`~pde.fields.FieldCollection`.
            Lastly, `source` can be a function that takes `fields` as an
            argument and returns the desired field.
        check_rank (int, optional):
            Can be given to check whether the extracted field has the correct
            rank (0 = ScalarField, 1 = VectorField, ...).

    Returns:
        :class:`~pde.fields.DataFieldBase`: The extracted field
    """
    # extract the source
    if source is None:
        field = fields
    elif callable(source):
        field = source(fields)
    else:
        if isinstance(fields, FieldCollection):
            field = fields[source]
        else:
            raise TypeError(
                f"Cannot extract component {source} from instance of "
                f"{fields.__class__.__name__}"
            )

    if isinstance(field, FieldCollection):
        raise RuntimeError("Extract field is a collection")

    # test whether the rank is as expected
    if check_rank is not None and field.rank != check_rank:  # type: ignore
        raise RuntimeError(f"Rank of extracted field is not {check_rank}")

    return field  # type: ignore


class ScalarFieldPlot:
    """ class managing compound plots of scalar fields """

    @fill_in_docstring
    def __init__(
        self,
        fields: FieldBase,
        quantities=None,
        scale: ScaleData = "automatic",
        fig=None,
        title: Optional[str] = None,
        tight: bool = False,
        show: bool = True,
    ):
        """
        Args:
            fields (:class:`~pde.fields.base.FieldBase`):
                Collection of fields
            quantities:
                {ARG_PLOT_QUANTITIES}
            scale (str, float, tuple of float):
                {ARG_PLOT_SCALE}
            fig (:class:`matplotlib.figure.Figure):
                Figure to be used for plotting. If `None`, a new figure is
                created.
            title (str):
                Title of the plot.
            tight (bool):
                Whether to call :func:`matplotlib.pyplot.tight_layout`. This
                affects the layout of all plot elements.
            show (bool):
                Flag determining whether to show a plot. If `False`, the plot is
                kept in the background, which can be useful if it only needs to
                be written to a file.
        """
        self.grid = fields.grid
        self.quantities = self._prepare_quantities(fields, quantities, scale=scale)
        self.show = show

        # figure out whether plots are shown in jupyter notebook
        try:
            from ipywidgets import Output
        except ImportError:
            ipython_plot = False
        else:
            ipython_plot = self.show

        if ipython_plot:
            # plotting is done in an ipython environment using widgets
            from IPython.display import display

            self._ipython_out = Output()
            with self._ipython_out:
                self._initialize(fields, scale, fig, title, tight)
            display(self._ipython_out)

        else:
            # plotting is done using a simple matplotlib backend
            self._ipython_out = None
            self._initialize(fields, scale, fig, title, tight)

        if self.show:
            self._show()

    @classmethod
    @fill_in_docstring
    def from_storage(
        cls,
        storage: StorageBase,
        quantities=None,
        scale: ScaleData = "automatic",
        tight: bool = False,
        show: bool = True,
    ) -> "ScalarFieldPlot":
        """create ScalarFieldPlot from storage

        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                Instance of the storage class that contains the data
            quantities:
                {ARG_PLOT_QUANTITIES}
            scale (str, float, tuple of float):
                {ARG_PLOT_SCALE}
            tight (bool):
                Whether to call :func:`matplotlib.pyplot.tight_layout`. This
                affects the layout of all plot elements.
            show (bool):
                Flag determining whether to show a plot. If `False`, the plot is
                kept in the background.

        Returns:
            :class:`~pde.visualization.plotting.ScalarFieldPlot`
        """
        fields = storage[0]
        assert isinstance(fields, FieldBase)

        # prepare the data that needs to be plotted
        quantities = cls._prepare_quantities(fields, quantities=quantities, scale=scale)

        # resolve automatic scaling
        for quantity_row in quantities:
            for quantity in quantity_row:
                if quantity.get("scale", "automatic") == "automatic":
                    vmin, vmax = +np.inf, -np.inf
                    for data in storage:
                        field = extract_field(data, quantity.get("source"))
                        img = field.get_image_data()
                        vmin = np.nanmin([np.nanmin(img["data"]), vmin])
                        vmax = np.nanmax([np.nanmax(img["data"]), vmax])
                    quantity["scale"] = (vmin, vmax)

        # actually setup
        return cls(  # lgtm [py/call-to-non-callable]
            fields, quantities, tight=tight, show=show
        )

    @staticmethod
    @fill_in_docstring
    def _prepare_quantities(
        fields: FieldBase, quantities, scale: ScaleData = "automatic"
    ) -> List[List[Dict[str, Any]]]:
        """internal method to prepare quantities

        Args:
            fields (:class:`~pde.fields.base.FieldBase`):
                The field containing the data to show
            quantities (dict):
                {ARG_PLOT_QUANTITIES}
            scale (str, float, tuple of float):
                {ARG_PLOT_SCALE}

        Returns:
            list of list of dict: a 2d arrangements of panels that define what
                quantities are shown.
        """
        if quantities is None:
            # show each component by default
            if isinstance(fields, FieldCollection):
                quantities = []
                for i, field in enumerate(fields):
                    title = field.label if field.label else f"Field {i + 1}"
                    quantity = {"title": title, "source": i, "scale": scale}
                    quantities.append(quantity)
            else:
                quantities = [{"title": "Concentration", "source": None}]

        # make sure panels is a 2d array
        if isinstance(quantities, dict):
            quantities = [[quantities]]
        elif isinstance(quantities[0], dict):
            quantities = [quantities]

        # set the scaling for quantities where it is not yet set
        for row in quantities:
            for col in row:
                col.setdefault("scale", scale)

        return quantities  # type: ignore

    def __del__(self):
        try:
            if hasattr(self, "fig") and self.fig:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
        except Exception:
            pass  # can occur when shutting down python interpreter

    @fill_in_docstring
    def _initialize(
        self,
        fields: FieldBase,
        scale: ScaleData = "automatic",
        fig=None,
        title: Optional[str] = None,
        tight: bool = False,
    ):
        """initialize the plot creating the figure and the axes

        Args:
            fields (:class:`~pde.fields.base.FieldBase`):
                Collection of fields
            scale (str, float, tuple of float):
                {ARG_PLOT_SCALE}
            fig (:class:`matplotlib.figure.Figure):
                Figure to be used for plotting. If `None`, a new figure is
                created.
            title (str):
                Title of the plot.
            tight (bool):
                Whether to call :func:`matplotlib.pyplot.tight_layout`. This
                affects the layout of all plot elements.
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        num_rows = len(self.quantities)
        num_cols = max(len(p) for p in self.quantities)

        example_image = fields.get_image_data()

        # set up the figure
        if fig is None:
            figsize = (4 * num_cols, 4 * num_rows)
            self.fig, self.axes = plt.subplots(
                num_rows, num_cols, sharey=True, squeeze=False, figsize=figsize
            )
        else:
            self.fig = fig
            self.axes = fig.axes

        if title is None:
            title = ""
        self.sup_title = plt.suptitle(title)

        # set up all images
        empty = np.zeros_like(example_image["data"])

        self.images: List[List[Any]] = []
        for i, panel_row in enumerate(self.quantities):
            img_row = []
            for j, panel in enumerate(panel_row):
                # determine scale of the panel
                panel_scale = panel.get("scale", scale)
                if panel_scale == "automatic":
                    vmin, vmax = None, None
                elif panel_scale == "unity":
                    vmin, vmax = 0, 1
                elif panel_scale == "symmetric":
                    vmin, vmax = -1, 1
                else:
                    try:
                        vmin, vmax = panel_scale
                    except TypeError:
                        vmin, vmax = 0, panel_scale

                # determine colormap
                cmap = panel.get("cmap")
                if cmap is None:
                    if vmin is None or vmax is None:
                        cmap = cm.viridis
                    elif np.isclose(-vmin, vmax):
                        cmap = cm.coolwarm
                    elif np.isclose(vmin, 0):
                        cmap = cm.gray
                    else:
                        cmap = cm.viridis

                # initialize image
                ax = self.axes[i, j]
                img = ax.imshow(
                    empty,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    extent=example_image["extent"],
                    origin="lower",
                    interpolation="quadric",
                )
                _add_horizontal_colorbar(img, ax)
                ax.set_title(panel.get("title", ""))
                ax.set_xlabel(example_image["label_x"])
                ax.set_ylabel(example_image["label_y"])

                # store the default value alongside
                img._default_vmin_vmax = vmin, vmax
                img_row.append(img)
            self.images.append(img_row)

        if tight:
            # adjust layout and leave some room for title
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _update_data(self, fields: FieldBase, title: Optional[str] = None) -> None:
        """update the fields in the current plot

        Args:
            fields (:class:`~pde.fields.base.FieldBase`):
                The field or field collection of which the defined quantities
                are shown.
            title (str, optional):
                The title of this view. If `None`, the current title is not
                changed.
        """
        assert isinstance(fields, FieldBase)

        if title:
            self.sup_title.set_text(title)

        # iterate over all panels and update their content
        for i, panel_row in enumerate(self.quantities):
            for j, panel in enumerate(panel_row):
                # obtain image data
                field = extract_field(fields, panel.get("source"))
                img_data = field.get_image_data()

                # set the data in the correct panel
                img = self.images[i][j]
                img.set_data(img_data["data"])
                vmin, vmax = img._default_vmin_vmax
                if vmin is None:
                    vmin = img_data["data"].min()
                if vmax is None:
                    vmax = img_data["data"].max()
                img.set_clim(vmin, vmax)

    def _show(self):
        """ show the updated plot """
        if self._ipython_out:
            # seems to be in an ipython instance => update widget
            from IPython.display import clear_output, display

            with self._ipython_out:
                clear_output(wait=True)
                display(self.fig)

            # add a small pause to allow the GUI to run it's event loop
            time.sleep(0.01)

        else:
            # seems to be in a normal matplotlib window => update it
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # add a small pause to allow the GUI to run it's event loop
                import matplotlib.pyplot as plt

                plt.pause(0.01)

    def update(self, fields: FieldBase, title: Optional[str] = None) -> None:
        """update the plot with the given fields

        Args:
            fields:
                The field or field collection of which the defined quantities
                are shown.
            title (str, optional):
                The title of this view. If `None`, the current title is not
                changed.
        """
        self._update_data(fields, title)
        if self.show:
            self._show()

    def savefig(self, path: str, **kwargs):
        """save plot to file

        Args:
            path (str):
                The path to the file where the image is written. The file
                extension determines the image format
            **kwargs:
                Additional arguments are forwarded to
                :meth:`matplotlib.figure.Figure.savefig`.
        """
        self.fig.savefig(path, **kwargs)

    def make_movie(
        self, storage: StorageBase, filename: str, progress: bool = True
    ) -> None:
        """make a movie from the data stored in storage

        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                The storage instance that contains all the data for the movie
            filename (str):
                The filename to which the movie is written. The extension
                determines the format used.
            progress (bool):
                Flag determining whether the progress of making the movie is
                shown.
        """
        from ..visualization.movies import Movie

        # create the iterator over the data
        if progress:
            data_iter = display_progress(storage.items(), total=len(storage))
        else:
            data_iter = storage.items()

        with Movie(filename=filename) as movie:
            # iterate over all time steps
            for t, data in data_iter:
                self.update(data, title=f"Time {t:g}")
                movie.add_figure(self.fig)


@plot_on_axes()
@fill_in_docstring
def plot_magnitudes(
    storage: StorageBase, quantities=None, ax=None, **kwargs
) -> PlotReference:
    r"""plot spatially averaged quantities as a function of time

    For scalar fields, the default is to plot the average value while the averaged norm
    is plotted for vector fields.

    Args:
        storage:
            Instance of :class:`~pde.storage.base.StorageBase` that contains
            the simulation data that will be plotted
        quantities:
            {ARG_PLOT_QUANTITIES}
        {PLOT_ARGS}
        \**kwargs:
            All remaining parameters are forwarded to the `ax.plot` method

    Returns:
        :class:`~pde.tools.plotting.PlotReference`: The reference to the plot
    """
    if quantities is None:
        fields = storage[0]
        assert isinstance(fields, FieldBase)
        if fields.label:
            label_base = fields.label
        else:
            label_base = kwargs.get("label", "Field")

        if isinstance(fields, FieldCollection):
            quantities = []
            for i, field in enumerate(fields):
                if field.label:
                    label = field.label
                else:
                    label = label_base + f" {i + 1}"
                quantities.append({"label": label, "source": i})
        else:
            quantities = [{"label": label_base, "source": None}]

    _logger.debug("Quantities: %s", quantities)

    # prepare data field
    data = [
        {"label": quantity.get("label", ""), "values": []} for quantity in quantities
    ]

    # calculate the data
    for fields in storage:
        for i, quantity in enumerate(quantities):
            source = quantity["source"]
            if callable(source):
                value = source(fields)
            elif source is None:
                value = fields.magnitude  # type: ignore
            else:
                value = fields[source].magnitude  # type: ignore
            value *= quantity.get("scale", 1)
            data[i]["values"].append(value)

    # plot the data
    lines = []
    check_complex = True
    for d in data:
        kwargs["label"] = d["label"]

        # warn if there is an imaginary part
        if check_complex and np.any(np.iscomplex(d["values"])):
            _logger.warning("Only the real part of the complex data is shown")
            check_complex = False  # only warn once

        # actually plot the data
        (l,) = ax.plot(storage.times, np.real(d["values"]), **kwargs)
        lines.append(l)

    ax.set_xlabel("Time")
    if len(data) == 1:
        ax.set_ylabel(kwargs["label"])

    if len(data) > 1:
        ax.legend(loc="best")

    return PlotReference(ax, lines)


def _plot_kymograph(
    img_data: Dict[str, Any],
    ax,
    colorbar: bool = True,
    transpose: bool = False,
    **kwargs,
) -> PlotReference:
    r"""plots a simple kymograph from given data

    Args:
        img_data (dict):
            Contains the kymograph data
        ax (:class:`~matplotlib.axes.Axes`):
            The axes to which the plot is added
        colorbar (bool):
            Whether to show a colorbar or not
        transpose (bool):
            Determines whether the transpose of the data should is plotted
        \**kwargs:
            Additional keyword arguments are passed to
            :func:`matplotlib.pyplot.imshow`.

    Returns:
        :class:`~pde.tools.plotting.PlotReference`: The reference to the plot
    """
    # transpose data if requested
    if transpose:
        # avoid changing the img_data that was passed into the function
        extent = np.r_[img_data["extent_y"], img_data["extent_x"]]
        img_data = {
            "data": img_data["data"].T,
            "label_x": img_data["label_y"],
            "label_y": img_data["label_x"],
        }
    else:
        extent = np.r_[img_data["extent_x"], img_data["extent_y"]]

    # warn if there is an imaginary part
    if np.any(np.iscomplex(img_data["data"])):
        _logger.warning("Only the real part of the complex data is shown")

    # create the actual plot
    axes_image = ax.imshow(
        img_data["data"].real, extent=extent, origin="lower", **kwargs
    )

    # adjust some settings
    ax.set_xlabel(img_data["label_x"])
    ax.set_ylabel(img_data["label_y"])
    ax.set_aspect("auto")

    if colorbar:
        import matplotlib.pyplot as plt

        plt.colorbar(axes_image, ax=ax)
        # Note that we here do not use the method `add_scaled_colorbar` from the
        # `pde.tools.plotting` module since this does not play well with axes with an
        # explicit aspect ratio.

    return PlotReference(ax, axes_image)


@plot_on_axes()
def plot_kymograph(
    storage: StorageBase,
    field_index: int = None,
    scalar: str = "auto",
    extract: str = "auto",
    colorbar: bool = True,
    transpose: bool = False,
    ax=None,
    **kwargs,
) -> PlotReference:
    r"""plots a single kymograph from stored data

    The kymograph shows line data stacked along time. Consequently, the
    resulting image shows space along the horizontal axis and time along the
    vertical axis.

    Args:
        storage (:class:`~droplets.simulation.storage.StorageBase`):
            The storage instance that contains all the data
        field_index (int):
            An index to choose a single field out of many in a collection
            stored in `storage`. This option should not be used if only a single
            field is stored in a collection.
        scalar (str):
            The method for extracting scalars as described in
            :meth:`DataFieldBase.to_scalar`.
        extract (str):
            The method used for extracting the line data. See the docstring
            of the grid method `get_line_data` to find supported values.
        colorbar (bool):
            Whether to show a colorbar or not
        transpose (bool):
            Determines whether the transpose of the data should is plotted
        {PLOT_ARGS}
        \**kwargs:
            Additional keyword arguments are passed to
            :func:`matplotlib.pyplot.imshow`.

    Returns:
        :class:`~pde.tools.plotting.PlotReference`: The reference to the plot
    """
    if len(storage) == 0:
        raise RuntimeError("Storage is empty")
    line_data_args: Dict[str, Any] = {"scalar": scalar, "extract": extract}

    if field_index is not None:
        if storage.has_collection:
            line_data_args["index"] = field_index
        else:
            _logger.warning("`field_index` should only be set for FieldCollections")

    # prepare the image data for one kymograph
    image = []
    for _, field in storage.items():
        img_data = field.get_line_data(**line_data_args)
        image.append(img_data["data_y"])

    img_data["data"] = np.array(image)
    img_data["extent_y"] = (storage.times[0], storage.times[-1])
    img_data["label_y"] = "Time"

    return _plot_kymograph(
        img_data, ax, colorbar=colorbar, transpose=transpose, **kwargs
    )


@plot_on_figure()
def plot_kymographs(
    storage: StorageBase,
    scalar: str = "auto",
    extract: str = "auto",
    colorbar: bool = True,
    transpose: bool = False,
    resize_fig: bool = True,
    fig=None,
    **kwargs,
) -> List[PlotReference]:
    r"""plots kymographs for all fields stored in `storage`

    The kymograph shows line data stacked along time. Consequently, the
    resulting image shows space along the horizontal axis and time along the
    vertical axis.

    Args:
        storage (:class:`~droplets.simulation.storage.StorageBase`):
            The storage instance that contains all the data
        scalar (str):
            The method for extracting scalars as described in
            :meth:`DataFieldBase.to_scalar`.
        extract (str):
            The method used for extracting the line data. See the docstring
            of the grid method `get_line_data` to find supported values.
        colorbar (bool):
            Whether to show a colorbar or not
        transpose (bool):
            Determines whether the transpose of the data should is plotted
        resize_fig (bool):
            Whether to resize the figure to adjust to the number of panels
        {PLOT_ARGS}
        \**kwargs:
            Additional keyword arguments are passed to the calls to
            :func:`matplotlib.pyplot.imshow`.

    Returns:
        list of :class:`~pde.tools.plotting.PlotReference`: The references to
        all plots
    """
    if len(storage) == 0:
        raise RuntimeError("Storage is empty")
    line_data_args = {"scalar": scalar, "extract": extract}
    if storage.has_collection:
        num_fields = len(storage[0])  # type: ignore
    else:
        num_fields = 1

    # prepare the image data for one kymograph
    images = []
    for _, field in storage.items():
        if storage.has_collection:
            image_lines = []
            for f_id in range(num_fields):
                img_data = field.get_line_data(  # type: ignore
                    index=f_id,
                    **line_data_args,
                )
                image_lines.append(img_data["data_y"])
        else:
            img_data = field.get_line_data(**line_data_args)
            image_lines = [img_data["data_y"]]
        images.append(image_lines)

    img_arr = np.array(images)
    img_data["extent_y"] = (storage.times[0], storage.times[-1])
    img_data["label_y"] = "Time"

    # disable interactive plotting temporarily
    # create a plot with all the panels
    if resize_fig:
        fig.set_size_inches((4 * num_fields, 3), forward=True)
    axs = fig.subplots(1, num_fields, squeeze=False)

    # iterate over all axes and plot the kymograph
    refs = []
    for i, ax in enumerate(axs[0]):
        img_data["data"] = img_arr[:, i]
        ref = _plot_kymograph(
            img_data, ax, colorbar=colorbar, transpose=transpose, **kwargs
        )
        if storage.has_collection:
            ax.set_title(storage._field[i].label)  # type: ignore
        refs.append(ref)

    return refs


def plot_interactive(
    storage: StorageBase,
    time_scaling: str = "exact",
    viewer_args: Dict[str, Any] = None,
    **kwargs,
):
    r"""plots stored data interactively using the `napari <https://napari.org>`_ viewer

    Args:
        storage (:class:`~droplets.simulation.storage.StorageBase`):
            The storage instance that contains all the data
        time_scaling (str):
            Defines how the time axis is scaled. Possible options are "exact" (the
            actual time points are used), or "scaled" (the axis is scaled so that it has
            similar dimension to the spatial axes). Note that the spatial axes will
            never be scaled.
        viewer_args (dict):
            Arguments passed to :class:`napari.viewer.Viewer` to affect the viewer.
        **kwargs:
            Extra arguments passed to the plotting function
    """
    if not module_available("napari"):
        raise ImportError("Require the `napari` module for interactive plotting")

    if len(storage) < 1:
        raise ValueError("Storage is empty")

    grid = storage.grid
    if grid is None:
        raise RuntimeError("Storage did not contain information about the grid")

    # collect data from all time points
    timecourse: Dict[str, List[np.ndarray]] = dict()
    for field in storage:
        layer_data = field._get_napari_data(**kwargs)

        if timecourse:
            for key, field_data in layer_data.items():
                timecourse[key].append(field_data["data"])
        else:
            for key, field_data in layer_data.items():
                timecourse[key] = [field_data["data"]]

    # replace the data in the layer_data
    for key, field_data in layer_data.items():
        field_data["data"] = np.array(timecourse[key])
        if "scale" in field_data:
            if time_scaling == "exact":
                dt = np.diff(storage.times)[0] if len(storage) > 1 else 1
            elif time_scaling == "scaled":
                length_scale = grid.volume ** (1 / grid.dim)
                dt = length_scale / len(storage)
            else:
                raise ValueError(f"Unknown time scaling `{time_scaling}`")
            field_data["scale"] = np.r_[dt, field_data["scale"]]

    if viewer_args is None:
        viewer_args = {}
    viewer_args.setdefault("axis_labels", ["Time"] + grid.axes)

    # actually display the data using napari
    with napari_viewer(grid, **viewer_args) as viewer:
        napari_add_layers(viewer, layer_data)
