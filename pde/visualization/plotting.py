'''
Functions and classes for plotting simulation data

.. autosummary::
   :nosignatures:

   ScalarFieldPlot
   plot_magnitudes
   plot_kymograph
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import logging
import warnings
from typing import Union, Callable, Optional, Any, List  # @UnusedImport

import numpy as np

from ..fields import FieldCollection
from ..fields.base import FieldBase, DataFieldBase
from ..storage.base import StorageBase
from ..tools.misc import display_progress



def _add_horizontal_colorbar(im, ax, num_loc: int = 5):
    """ adds a horizontal colorbar for image `im` to the axis `ax` """
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
                  close_figure: bool = False):
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
        Result of :func:`matplotlib.pyplot.imshow`
    """
    import matplotlib.pyplot as plt
    
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
    
    if title:
        ax.set_title(title)
    
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    if close_figure:
        plt.close(fig)
        
    return fig
    


def extract_field(fields: FieldBase,
                  source: Union[None, int, Callable] = None,
                  check_rank: Optional[int] = None) -> DataFieldBase:
    """Extracts a single field from a possible collection.
    
    Args:
        fields: An instance of :class:`~pde.fields.FieldBase`.
        source (int or callable, optional): Determines how a field is extracted
            from `fields`. If `None`, `fields` is passed as is, assuming it is
            already a scalar field. This works for the simple, standard case
            where only a single ScalarField is treated. Alternatively, `source`
            can be an integer, indicating which field is extracted from an
            instance of :class:`~pde.fields.FieldCollection`.
            Lastly, `source` can be a function that takes `fields` as an
            argument and returns the desired field.
        check_rank (int, optional): Can be given to check whether the extracted
            field has the correct rank (0 = ScalarField, 1 = VectorField, ...).
            
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
    
    def __init__(self, fields: FieldBase, quantities=None, show: bool = True):
        """
        Args:
            fields: Collection of fields
            quantities: |Args_plot_quantities|
            show (bool): Flag determining whether to show a plot. If `False`,
                the plot is kept in the background, which can be useful if
                it only needs to be written to a file
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        self.grid = fields.grid
        example_image = fields.get_image_data()
        self.quantities = self._prepare_quantities(fields, quantities)
        self.show = show
             
        num_rows = len(self.quantities)
        num_cols = max(len(p) for p in self.quantities)
             
        # set up the figure
        self.fig, self.axes = plt.subplots(num_rows, num_cols,
                                           sharey=True, squeeze=False,
                                           figsize=(4 * num_cols, 4 * num_rows))
        self.sup_title = plt.suptitle('')
        
        # set up all images
        empty = np.zeros_like(example_image['data'])

        self.images: List[List[Any]] = []
        for i, panel_row in enumerate(self.quantities):
            img_row = []
            for j, panel in enumerate(panel_row):
                # determine scale of the panel
                scale = panel.get('scale', 1)
                try:
                    vmin, vmax = scale
                except TypeError:
                    vmin, vmax = 0, scale
                    
                # determine colormap
                cmap = panel.get('cmap')
                if cmap is None:
                    if np.isclose(-vmin, vmax):
                        cmap = cm.coolwarm
                    elif np.isclose(vmin, 0):
                        cmap = cm.gray
                    else:
                        cmap = cm.PiYG

                # initialize image    
                ax = self.axes[i, j]
                img = ax.imshow(empty, vmin=vmin, vmax=vmax, cmap=cmap,
                                extent=example_image['extent'], origin='lower',
                                interpolation='quadric')
                _add_horizontal_colorbar(img, ax)
                ax.set_title(panel.get('title', ''))
                ax.set_xlabel(example_image['label_x'])
                ax.set_ylabel(example_image['label_y'])
#                 ax.axison = False
                img_row.append(img)
            self.images.append(img_row)
            
        self.show_data(fields)    
        
    
    @classmethod
    def from_storage(cls, storage: StorageBase, quantities=None):
        """ create ScalarFieldPlot from storage
        
        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                Instance of the storage class that contains the data
            quantities: |Args_plot_quantities|
            
        Returns:
            :class:`~pde.visualization.plotting.ScalarFieldPlot`
        """
        fields = storage.get_field(0)
        quantities = cls._prepare_quantities(fields, quantities, 
                                             default_scale='automatic')
        
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
        return cls(fields, quantities=quantities)
        
        
    @staticmethod
    def _prepare_quantities(fields: FieldBase, quantities, default_scale=None):
        """ internal method to prepare quantities """
        if quantities is None:
            # show each component by default
            if isinstance(fields, FieldCollection):
                quantities = []
                for i, field in enumerate(fields):
                    title = field.label if field.label else f'Field {i + 1}'
                    quantity = {'title': title, 'source': i}
                    if default_scale:
                        quantity['scale'] = default_scale
                    quantities.append(quantity)
            else:
                quantities = [{'title': 'Concentration', 'source': None}]
            
        # make sure panels is a 2d array
        if isinstance(quantities, dict):
            quantities = [[quantities]]
        elif isinstance(quantities[0], dict):
            quantities = [quantities]
        return quantities
        
        
    def __del__(self):
        try:
            import matplotlib.pyplot as plt
        except TypeError:
            pass  # can occur when shutting down python interpreter
        else:
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
        
            
    def show_data(self, fields: FieldBase, title: Optional[str] = None) -> None:
        """ show the given fields in the current view
        
        Args:
            fields: The field or field collection of which the defined 
                quantities are shown.
            title (str, optional): The title of this view. If `None`, the
                current title is not changed. 
        """
        import matplotlib.pyplot as plt
        assert isinstance(fields, FieldBase)
        
        if title:
            self.sup_title.set_text(title)
        
        for i, panel_row in enumerate(self.quantities):
            for j, panel in enumerate(panel_row):
                field = extract_field(fields, panel.get('source'))
                img = field.get_image_data()
                self.images[i][j].set_data(img['data'])
                
        # add a small pause to allow the GUI to run it's event loop
        if self.show:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.pause(0.01)
                
    
    def make_movie(self, storage: StorageBase,
                   filename: str,
                   progress: bool = True) -> None:
        """ make a movie from the data stored in storage
        
        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                The storage instance that contains all the data for the movie
            filename (str): The filename to which the movie is written. The
                extension determines the format used.
            progress (bool): Flag determining whether the progress of making
                the movie is shown.
        """
        from ..visualization.movies import Movie

        data_iter = storage.items()
        if progress:
            data_iter = display_progress(data_iter, total=len(storage))
            
        with Movie(filename=filename, verbose=False) as movie:
            # iterate over all time steps
            for t, data in data_iter:
                self.show_data(data, title=f'Time {t:g}')
                movie.add_figure(self.fig)



def plot_magnitudes(storage: StorageBase,
                    filename: str = None,
                    quantities=None,
                    ax=None,
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
        quantities: |Args_plot_quantities|
        ax:
            Matplotlib figure axes to be used for plotting. If `None`, a new
            figure is created
        show (bool):
            Flag determining whether :func:`matplotlib.pyplot.show` is called
        \**kwargs:
            All remaining parameters are forwarded to the `ax.plot` method
    """
    import matplotlib.pyplot as plt
    
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
    
    close_figure = (filename and ax is None)
    if ax is None:
        # create new figure
        ax = plt.figure().gca()
    
    # plot the data
    for d in data:
        kwargs['label'] = d['label']
        ax.plot(storage.times, d['values'], **kwargs)
        
    if len(data) > 1:
        ax.legend(loc='best')
    
    fig = ax.figure
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    if close_figure:
        plt.close(fig)
            


def plot_kymograph(storage: StorageBase,
                   filename: str = None,
                   extract: str = 'auto',
                   colorbar: bool = True,
                   transpose: bool = False,
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
        extract (str):
            The method used for extracting the line data. See the docstring
            of the grid method `get_line_data` to find supported values.
        colorbar (bool):
            Whether to show a colorbar or not
        transpose (bool):
            Determines whether the transpose of the data should is plotted
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
    import matplotlib.pyplot as plt

    close_figure = (filename != '' and ax is None)
    if ax is None:
        # create new figure
        ax = plt.figure().gca()

    full_data = []
    for _, data in storage.items():
        img_data = data.get_line_data(extract=extract)
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

    finalize_plot(ax, title=img_data['label_y'], filename=filename, show=show,
                  close_figure=close_figure)
    
    return res
