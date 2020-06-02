'''
Functions and classes for plotting simulation data

.. autosummary::
   :nosignatures:

   ScalarFieldPlot
   plot_magnitudes
   plot_kymograph
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import contextlib
import logging
import warnings
import time
from typing import (Union, Callable, Optional, Any, Dict, List, Tuple)

import numpy as np
import matplotlib.pyplot as plt

from ..grids.base import GridBase
from ..fields import FieldCollection
from ..fields.base import FieldBase, DataFieldBase
from ..storage.base import StorageBase
from ..tools.misc import display_progress
from ..tools.docstrings import fill_in_docstring


ScaleData = Union[str, float, Tuple[float, float]]



def _add_horizontal_colorbar(im, ax, num_loc: int = 5) -> None:
    """ adds a horizontal colorbar for image `im` to the axis `ax`
    
    Args:
        im: The result of calling :func:`matplotlib.pyplot.imshow`
        ax: The matplotlib axes to which the colorbar is added
        num_loc (int): Number of ticks
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MaxNLocator
    
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.1, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.locator = MaxNLocator(num_loc)
    cb.update_ticks()



def finalize_plot(fig_or_ax=None,
                  title: str = None,
                  filename: str = None,
                  show: bool = False,
                  close_figure: bool = False) -> Tuple[Any, Any]:
    r""" finalizes a figure by adjusting relevant parameters
    
    Args:
        fig_or_ax:
            The figure or axes that are affected. If `None`, the current figure
            is used.
        title (str):
            Determines the title of the figure
        filename (str):
            If given, the resulting image is written to this file.
        show (bool):
            Flag determining whether :func:`matplotlib.pyplot.show` is called
        close_figure (bool):
            Whether the figure should be closed in the end
            
    Returns:
        tuple: The figure and the axes that were used to finalize the plot
    """
    # determine which figure to modify    
    if fig_or_ax is None:
        fig = plt.gcf()  # current figure
        ax = fig.gca()
    elif hasattr(fig_or_ax, 'savefig'):
        fig = fig_or_ax  # figure is given
        ax = fig.gca()
    else:
        ax = fig_or_ax  # assume that axes are given
        try:
            fig = fig_or_ax.figure
        except AttributeError:
            fig = fig_or_ax.fig
    
    if title is not None:
        ax.set_title(title)
    
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    if close_figure:
        plt.close(fig)
        
    return fig, ax
    


def extract_field(fields: FieldBase,
                  source: Union[None, int, Callable] = None,
                  check_rank: Optional[int] = None) -> DataFieldBase:
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
            raise TypeError(f'Cannot extract component {source} from instance '
                            f'of {fields.__class__.__name__}')
        
    if isinstance(field, FieldCollection):
        raise RuntimeError('Extract field is a collection')
    
    # test whether the rank is as expected
    if check_rank is not None and field.rank != check_rank:  # type: ignore
        raise RuntimeError(f'Rank of extracted field is not {check_rank}')
        
    return field  # type: ignore



class ScalarFieldPlot():
    """ class managing compound plots of scalar fields """
    
    
    @fill_in_docstring
    def __init__(self,                 
                 fields: FieldBase,
                 quantities=None,
                 scale: ScaleData = 'automatic',
                 fig=None,
                 title: Optional[str] = None,
                 tight: bool = False,
                 show: bool = True):
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
        self.quantities = self._prepare_quantities(fields, quantities,
                                                   scale=scale)
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
    def from_storage(cls, storage: StorageBase,
                     quantities=None,
                     scale: ScaleData = 'automatic',
                     tight: bool = False,
                     show: bool = True) -> "ScalarFieldPlot":
        """ create ScalarFieldPlot from storage
        
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
        fields = storage.get_field(0)
        quantities = cls._prepare_quantities(fields, quantities=quantities,
                                             scale=scale)
        
        # resolve automatic scaling
        for quantity_row in quantities:
            for quantity in quantity_row:
                if quantity.get('scale', 'automatic') == 'automatic':
                    vmin, vmax = +np.inf, -np.inf
                    for data in storage:
                        field = extract_field(data, quantity.get('source'))
                        img = field.get_image_data()
                        vmin = np.nanmin([np.nanmin(img['data']), vmin])
                        vmax = np.nanmax([np.nanmax(img['data']), vmax])
                    quantity['scale'] = (vmin, vmax)
            
        # actually setup 
        return cls(fields,  # lgtm [py/call-to-non-callable]
                   quantities,
                   tight=tight, show=show)
        
        
    @staticmethod
    @fill_in_docstring
    def _prepare_quantities(fields: FieldBase,
                            quantities,
                            scale: ScaleData = 'automatic') \
            -> List[List[Dict[str, Any]]]:
        """ internal method to prepare quantities
        
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
                    title = field.label if field.label else f'Field {i + 1}'
                    quantity = {'title': title, 'source': i, 'scale': scale}
                    quantities.append(quantity)
            else:
                quantities = [{'title': 'Concentration', 'source': None}]
                
        # make sure panels is a 2d array
        if isinstance(quantities, dict):
            quantities = [[quantities]]
        elif isinstance(quantities[0], dict):
            quantities = [quantities]
            
        # set the scaling for quantities where it is not yet set
        for row in quantities:
            for col in row:
                col.setdefault('scale', scale)
            
        return quantities  # type: ignore
        
        
    def __del__(self):
        try:
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
        except Exception:
            pass  # can occur when shutting down python interpreter

    
    @fill_in_docstring
    def _initialize(self, fields: FieldBase,
                    scale: ScaleData = 'automatic',
                    fig=None,
                    title: Optional[str] = None,
                    tight: bool = False):         
        """ initialize the plot creating the figure and the axes
        
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
        
        num_rows = len(self.quantities)
        num_cols = max(len(p) for p in self.quantities)
             
        example_image = fields.get_image_data()
        
        # set up the figure
        if fig is None:
            figsize = (4 * num_cols, 4 * num_rows)
            self.fig, self.axes = plt.subplots(num_rows, num_cols, sharey=True,
                                               squeeze=False, figsize=figsize)
        else:
            self.fig = fig
            self.axes = fig.axes
            
            
        if title is None:
            title = ""        
        self.sup_title = plt.suptitle(title)
        
        # set up all images
        empty = np.zeros_like(example_image['data'])

        self.images: List[List[Any]] = []
        for i, panel_row in enumerate(self.quantities):
            img_row = []
            for j, panel in enumerate(panel_row):
                # determine scale of the panel
                panel_scale = panel.get('scale', scale)
                if panel_scale == 'automatic':
                    vmin, vmax = None, None
                elif panel_scale == 'unity':
                    vmin, vmax = 0, 1
                elif panel_scale == 'symmetric':
                    vmin, vmax = -1, 1
                else:
                    try:
                        vmin, vmax = panel_scale
                    except TypeError:
                        vmin, vmax = 0, panel_scale
                    
                # determine colormap
                cmap = panel.get('cmap')
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
                img = ax.imshow(empty, vmin=vmin, vmax=vmax, cmap=cmap,
                                extent=example_image['extent'], origin='lower',
                                interpolation='quadric')
                _add_horizontal_colorbar(img, ax)
                ax.set_title(panel.get('title', ''))
                ax.set_xlabel(example_image['label_x'])
                ax.set_ylabel(example_image['label_y'])
                
                # store the default value alongside
                img._default_vmin_vmax = vmin, vmax
                img_row.append(img)
            self.images.append(img_row)
            
        if tight:
            # adjust layout and leave some room for title
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            
    def _update_data(self, fields: FieldBase,
                     title: Optional[str] = None) -> None:
        """ update the fields in the current plot
        
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
                field = extract_field(fields, panel.get('source'))
                img_data = field.get_image_data()
                
                # set the data in the correct panel
                img = self.images[i][j]
                img.set_data(img_data['data'])
                vmin, vmax = img._default_vmin_vmax
                if vmin is None:
                    vmin = img_data['data'].min()
                if vmax is None:
                    vmax = img_data['data'].max()
                img.set_clim(vmin, vmax)
                    
            
    def _show(self):
        """ show the updated plot """
        if self._ipython_out:
            # seems to be in an ipython instance => update widget
            from IPython.display import display, clear_output
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
                plt.pause(0.01)
        
        
    def update(self, fields: FieldBase, title: Optional[str] = None) -> None:
        """ update the plot with the given fields
        
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
        """ save plot to file 
        
        Args:
            path (str):
                The path to the file where the image is written. The file
                extension determines the image format
            **kwargs:
                Additional arguments are forwarded to 
                :meth:`matplotlib.figure.Figure.savefig`.
        """
        self.fig.savefig(path, **kwargs)
        
    
    def make_movie(self, storage: StorageBase,
                   filename: str,
                   progress: bool = True) -> None:
        """ make a movie from the data stored in storage
        
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
            
        with Movie(filename=filename, verbose=False) as movie:
            # iterate over all time steps
            for t, data in data_iter:
                self.update(data, title=f'Time {t:g}')
                movie.add_figure(self.fig)



@fill_in_docstring
def plot_magnitudes(storage: StorageBase,
                    filename: str = None,
                    quantities=None,
                    ax=None,
                    title: str = None,
                    show: bool = False,
                    **kwargs) -> None:
    r""" create a plot of spatially integrated quantities from a given storage.
    
    Note that the plot will not be displayed when a filename is given and no
    plot axes are specified.
    
    Args:
        storage:
            Instance of :class:`~pde.storage.base.StorageBase` that contains
            the simulation data that will be plotted
        filename (str):
            If given, the resulting image is written to this file.
        quantities:
            {ARG_PLOT_QUANTITIES}
        ax:
            Matplotlib figure axes to be used for plotting. If `None`, a new
            figure is created
        title (str):
            Determines the title of the figure
        show (bool):
            Flag determining whether :func:`matplotlib.pyplot.show` is called
        \**kwargs:
            All remaining parameters are forwarded to the `ax.plot` method
    """
    if quantities is None:
        fields = storage.get_field(0)
        if fields.label:
            label_base = fields.label 
        else:
            label_base = kwargs.get('label', 'Field')
        
        if isinstance(fields, FieldCollection):
            quantities = []
            for i, field in enumerate(fields):
                if field.label:
                    label = field.label 
                else:
                    label = label_base + f' {i + 1}'
                quantities.append({'label': label, 'source': i})
        else:
            quantities = [{'label': label_base, 'source': None}]
            
    logging.getLogger(__name__).debug('Quantities: %s', quantities)
    
    # prepare data field
    data = [{'label': quantity['label'], 'values': []}
            for quantity in quantities]
    
    # calculate the data
    for fields in storage:
        for i, quantity in enumerate(quantities):
            source = quantity['source']
            if callable(source):
                value = source(fields)
            elif source is None:
                value = fields.magnitude  # type: ignore
            else:
                value = fields[source].magnitude  # type: ignore
            value *= quantity.get('scale', 1)
            data[i]['values'].append(value)
    
    close_figure = (bool(filename) and ax is None)
    if ax is None:
        # create new figure
        ax = plt.figure().gca()
    
    # plot the data
    for d in data:
        kwargs['label'] = d['label']
        ax.plot(storage.times, d['values'], **kwargs)
        
    ax.set_xlabel('Time')
    if len(data) == 1:
        ax.set_ylabel(kwargs['label'])
        
    if len(data) > 1:
        ax.legend(loc='best')
    
    finalize_plot(ax, title=title, filename=filename, show=show,
                  close_figure=close_figure)
            


def plot_kymograph(storage: StorageBase,
                   filename: str = None,
                   scalar: str = 'auto',
                   extract: str = 'auto',
                   colorbar: bool = True,
                   transpose: bool = False,
                   title: str = None,
                   show: bool = False,
                   ax=None,
                   **kwargs):
    r""" plots a simple kymograph from stored data
    
    The kymograph shows line data stacked along time. Consequently, the
    resulting image shows space along the horizontal axis and time along the
    vertical axis.
    
    Note that the plot will not be displayed when a filename is given and no
    plot axes are specified.
    
    Args:
        storage (:class:`~droplets.simulation.storage.StorageBase`):
            The storage instance that contains all the data for the movie
        filename (str):
            If given, the resulting image is written to this file.
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
        title (str):
            Determines the title of the figure
        show (bool):
            Flag determining whether :func:`matplotlib.pyplot.show` is called
        ax: Figure axes to be used for plotting. If `None`, a new figure is
            created
        \**kwargs:
            Additional keyword arguments are passed to
            :func:`matplotlib.pyplot.imshow`.
            
    Returns:
        Result of :func:`matplotlib.pyplot.imshow`
    """
    close_figure = (bool(filename) and ax is None)
    if ax is None:
        ax = plt.figure().gca()  # create new figure

    full_data = []
    for _, data in storage.items():
        img_data = data.get_line_data(scalar=scalar, extract=extract)
        full_data.append(img_data['data_y'])
        
    full_data = np.array(full_data)
    extent = np.r_[img_data['extent_x'], storage.times[0], storage.times[-1]]
    
    if transpose:
        full_data = full_data.T  # type: ignore
        label_x, label_y = 'Time', img_data['label_x']
    else:
        label_x, label_y = img_data['label_x'], 'Time'
    
    res = ax.imshow(full_data, extent=extent, origin='lower', **kwargs)
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect('auto') 

    if colorbar:
        from ..tools.misc import add_scaled_colorbar
        add_scaled_colorbar(res, ax=ax)
        
    if title is None:
        title = img_data['label_y']

    finalize_plot(ax, title=title, filename=filename, show=show,
                  close_figure=close_figure)
    
    return res



@contextlib.contextmanager
def napari_viewer(grid: GridBase, **kwargs):
    """ creates an napari viewer for interactive plotting
    
    Args:
        grid (:class:`pde.grids.base.GridBase`): The grid defining the space
        **kwargs: Extra arguments are passed to :class:`napari.Viewer`
    """
    import napari
    
    if grid.num_axes == 1:
        raise RuntimeError('Interactive plotting only works for data with '
                           'at least 2 dimensions')
    
    viewer_args = kwargs
    viewer_args.setdefault('axis_labels', grid.axes)
    viewer_args.setdefault('ndisplay', 3 if grid.num_axes >= 3 else 2)
    
    with napari.gui_qt():  # create Qt GUI context
        yield napari.Viewer(**viewer_args)
    