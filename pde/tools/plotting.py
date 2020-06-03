'''
Tools for plotting and controlling plot output using context managers

.. autosummary::
   :nosignatures:

   finalize_plot
   disable_interactive
   plot_on_axes
   plot_on_figure
   PlotReference
   BasicPlottingContext
   JupyterPlottingContext
   get_plotting_context
   napari_viewer

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import re
import contextlib
import logging
import warnings
from typing import Type, Dict, Tuple, Any, TYPE_CHECKING  # @UnusedImport

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes

if TYPE_CHECKING:
    from ..grids.base import GridBase  # @UnusedImport

 
 
_IS_PLOTTING = False  # global flag to detect nested plotting calls



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
        fig = ax.get_figure()
    
    if title is not None:
        ax.set_title(title)
    
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    if close_figure:
        plt.close(fig)
        
    return fig, ax
    
    

@contextlib.contextmanager
def disable_interactive():
    """ context manager disabling the interactive mode of matplotlib
    
    This context manager restores the previous state after it is done. Details
    of the interactive mode are described in :func:`matplotlib.interactive`.
    """
    if plt.isinteractive():
        # interactive mode is enabled => disable it temporarily
        plt.interactive(False) 
        yield
        plt.interactive(True)
        
    else:
        # interactive mode is already disabled => do nothing
        yield



class plot_on_axes:
    """ wrapper for a plot method that fills an axes
    
    Example:
        The following example illustrates how the classes in this module can be
        used to implement plotting on a class. In particular, supplying the
        `update_method` will allow efficient dynamical plotting::
    
            class State:
                def __init__(self):
                    self.data = np.arange(8)
    
                def _update_plot(self, reference):
                    reference.element.set_ydata(self.data)
    
                @plot_on_axes(update_method='_update_plot')
                def plot(self, ax):
                    line, = ax.plot(np.arange(8), self.data)
                    return PlotReference(ax, line)
    """
    
    
    def __init__(self, update_method=None):
        """
        Args:
            update_method (callable):
                Method to call to update the plot. The argument of the new
                method will be the result of the initial call of the wrapped
                method.
        """
        # initialize the decorator
        if callable(update_method):
            raise RuntimeError('Wrapper needs to be called with brackets')
        self.update_method = update_method
        
        
    def __call__(self, method):
        """ apply the actual decorator """
        
        def wrapper(obj, *,
                    title: str = None,
                    filename: str = None,
                    show: bool = None,
                    close_figure: bool = False,
                    ax=None,
                    **kwargs):
            """
            Args:
                title (str):
                    Title of the plot. If omitted, the title might be chosen
                    automatically.
                filename (str, optional):
                    If given, the plot is written to the specified file.
                show (bool):
                    Flag setting whether :func:`matplotlib.pyplot.show` is
                    called. The value `None` sets show to `True` by default, but
                    disables it for nested calls.                    
                close_figure (bool):
                    Flag setting whether the figure is closed (after it was
                    shown)
            """
            # some logic to check for nested plotting calls:
            global _IS_PLOTTING
            root_plotting_call = not _IS_PLOTTING
            if show is None:
                show = root_plotting_call  # only call show outer routine
            _IS_PLOTTING = True
            
            # disable interactive plotting temporarily
            with disable_interactive():
                
                if ax is None:
                    # create new figure
                    backend = mpl.get_backend()
                    if 'backend_inline' in backend or 'nbAgg' == backend:
                        plt.close('all')  # close figures that were left over
                    fig, ax = plt.subplots()
                else:
                    fig = ax.get_figure()
            
                # call the actual plotting function
                reference = method(obj, ax=ax, **kwargs)
                
                # finishing touches...            
                if title is not None:
                    reference.ax.set_title(title)
                if filename:
                    fig.savefig(filename)
                    
            # decide what to do with the final plot
            if show:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.show()
            if close_figure:
                plt.close(fig)
                
            if root_plotting_call:
                _IS_PLOTTING = False
                
            return reference
            
        wrapper.__name__ = method.__name__        
        wrapper.__module__ = method.__module__
        
        doc_fragment = wrapper.__doc__.splitlines()[:-1]
        doc_fragment = '\n'.join(line[4:] for line in doc_fragment)
        wrapper.__doc__ = re.sub(r'Args:\s*$', doc_fragment, method.__doc__,
                                 flags=re.MULTILINE)
        wrapper.__dict__.update(method.__dict__)

        wrapper.mpl_class = 'axes'
        wrapper.update_method = self.update_method
            
        return wrapper
    


class plot_on_figure:
    """ wrapper for a plot method that fills an entire figure """
    
    
    def __init__(self, update_method=None):
        """
        Args:
            update_method (callable):
                Method to call to update the plot. The argument of the new
                method will be the result of the initial call of the wrapped
                method.
        """
        # initialize the decorator
        if callable(update_method):
            raise RuntimeError('Wrapper needs to be called with brackets')
        self.update_method = update_method
        
        
    def __call__(self, method):
        """ apply the actual decorator """
        
        def wrapper(obj, *,
                    title: str = None,
                    constrained_layout: bool = True,
                    filename: str = None,
                    show: bool = None,
                    close_figure: bool = False,
                    fig=None,
                    **kwargs):
            """
            Args:
                title (str):
                    Title of the plot. If omitted, the title might be chosen
                    automatically. This is shown above all panels.
                constrained_layout (bool):
                    Whether to use `constrained_layout` in 
                    :func:`matplotlib.pyplot.figure` call to create a figure. 
                    This affects the layout of all plot elements.
                filename (str, optional):
                    If given, the figure is written to the specified file.
                show (bool):
                    Flag setting whether :func:`matplotlib.pyplot.show` is
                    called. The value `None` sets `show` to `True` by default,
                    but disables it for nested calls.                  
                close_figure (bool):
                    Flag setting whether the figure is closed (after it was
                    shown).
            """
            # some logic to check for nested plotting calls:
            global _IS_PLOTTING
            root_plotting_call = not _IS_PLOTTING
            if show is None:
                show = root_plotting_call  # only call show outer routine
            _IS_PLOTTING = True

            # disable interactive plotting temporarily
            with disable_interactive():
                
                if fig is None:
                    # create new figure
                    backend = mpl.get_backend()
                    if 'backend_inline' in backend or 'nbAgg' == backend:
                        plt.close('all')  # close figures that were left over
                    fig = plt.figure(constrained_layout=constrained_layout)
            
                # call the actual plotting function
                reference = method(obj, fig=fig, **kwargs)
                
                # finishing touches...            
                if title is not None:
                    fig.suptitle(title)
                if filename:
                    fig.savefig(filename)
                    
            # decide what to do with the final plot
            if show:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.show()
                    
            if close_figure:
                plt.close(fig)    
                
            if root_plotting_call:
                _IS_PLOTTING = False
                
            return reference    
            
        wrapper.__name__ = method.__name__        
        wrapper.__module__ = method.__module__
        
        doc_fragment = wrapper.__doc__.splitlines()[:-1]
        doc_fragment = '\n'.join(line[4:] for line in doc_fragment)
        wrapper.__doc__ = re.sub(r'Args:\s*$', doc_fragment, method.__doc__,
                                 flags=re.MULTILINE)
        wrapper.__dict__.update(method.__dict__)

        wrapper.mpl_class = 'figure'
        wrapper.update_method = self.update_method
        
        return wrapper



class PlotReference():
    """ contains all information to update a plot element """
    
    __slots__ = ['ax', 'element', 'parameters']
    
    def __init__(self, ax, element: Any, parameters: Dict[str, Any] = None):
        """
        Args:
            ax (:class:`matplotlib.axes.Axes`): The axes of the element
            element (:class:`matplotlib.artist.Artist`): The actual element 
            parameters (dict): Parameters to recreate the plot element
        """
        self.ax = ax
        self.element = element
        self.parameters = {} if parameters is None else parameters



class PlottingContextBase(object):
    """ base class of the plotting contexts
    
    Example:
        The context wraps calls to the :mod:`matplotlib.pyplot` interface::
    
            context = PlottingContext()
            with context:
                plt.plot(...)
                plt.xlabel(...)
    """
    
    supports_update: bool = True
    """ flag indicating whether the context supports that plots can be updated
    with out redrawing the entire plot """
    
    
    def __init__(self,
                 title: str = None,
                 show: bool = True):
        """ 
        Args:
            title (str): The shown in the plot
            show (bool): Flag determining whether plots are actually shown
        """
        self.title = title
        self.show = show
        
        self.initial_plot = True
        self.fig = None
        self._logger = logging.getLogger(__name__)
        self._logger.info(f'Initialize {self.__class__.__name__}')
    

    def __enter__(self):
        # start the plotting process
        if self.fig is not None:
            plt.figure(self.fig.number)
    

    def __exit__(self, *exc): 
        if self.initial_plot or not self.supports_update:
            # recreate the entire figure
            self.fig = plt.gcf()
            if len(self.fig.axes) == 0:
                # The figure seems to be empty, which must be a mistake
                raise RuntimeError('Plot figure does not contain axes')
            
            elif len(self.fig.axes) == 1:
                # The figure contains only a single axis, indicating that it is
                # composed of a single panel
                self._title = plt.title(self.title)
                
            else:
                # The figure contains multiple axes. This is an indication that
                # the figure consists of multiple panels, although insets and
                # colorbars also count as additional axes
                self._title = plt.suptitle(self.title)
                
            self.initial_plot = False
            
        else:
            # update the old figure
            self._title.set_text(self.title)


    def close(self):
        """ close the plot """
        # close matplotlib figure
        if self.fig is not None:
            plt.close(self.fig)



class BasicPlottingContext(PlottingContextBase):
    """ basic plotting using just matplotlib """
    
    def __init__(self,
                 fig_or_ax=None,
                 title: str = None,
                 show: bool = True):
        """
        Args:
            fig_or_ax:
                If axes are given, they are used. If a figure is given, it is 
                set as active.
            title (str):
                The shown in the plot
            show (bool):
                Flag determining whether plots are actually shown
        """
        super().__init__(title=title, show=show)
        
        # determine which figure to modify
        if isinstance(fig_or_ax, mpl_axes.Axes):
            self.fig = fig_or_ax.get_figure()  # assume that axes are given
        elif isinstance(fig_or_ax, mpl.figure.Figure):
            self.fig = fig_or_ax
    

    def __exit__(self, *exc): 
        super().__exit__(*exc)
        if self.show:
            self.fig.canvas.draw()  # required for display in nbagg backend
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # add a small pause to allow the GUI to run it's event loop
                plt.pause(1e-3)



class JupyterPlottingContext(PlottingContextBase):
    """ plotting in a jupyter widget using the `inline` backend """
    
    supports_update = False
    """ flag indicating whether the context supports that plots can be updated
    with out redrawing the entire plot. The jupyter backend (`inline`) requires
    replotting of the entire figure, so an update is not supported."""
    
    def __enter__(self):
        from IPython.display import display
        from ipywidgets import Output
        
        if self.initial_plot:
            # close all previous plots
            plt.close('all')
            
            # create output widget for capturing all plotting
            self._ipython_out = Output()
            
            if self.show:
                # only show the widget if necessary
                display(self._ipython_out)
            
        # capture plots in the output widget
        self._ipython_out.__enter__()
    
    
    def __exit__(self, *exc):
        # finalize plot
        super().__exit__(*exc)
        
        if self.show:
            # show the plot, but ...
            plt.show()  # show the figure to make sure it can be captured
        # ... also clear it the next time something is done        
        self._ipython_out.clear_output(wait=True)
            
        # stop capturing plots in the output widget
        self._ipython_out.__exit__(*exc)

        # close the figure, so figure windows do not accumulate
        plt.close(self.fig)
 

    def close(self):
        """ close the plot """
        super().close()
        # close ipython output
        try:
            self._ipython_out.close()
        except Exception:
            pass


        
def get_plotting_context(context=None,
                         title: str = None,
                         show: bool = True) -> PlottingContextBase:
    """ returns a suitable plotting context
    
    Args:
        context:
            An instance of :class:`PlottingContextBase` or an instance of
            :class:`matplotlib.axes.Axes` or :class:`matplotlib.figure.Figure`
            to determine where the plotting will happen. If omitted, the context
            is determined automatically.
        title (str):
            The title shown in the plot
        show (bool):
            Determines whether the plot is shown while the simulation is
            running. If `False`, the files are created in the background.
            
    Returns:
        :class:`PlottingContextBase`: The plotting context
    """
    if context is None:
        # figure out whether plots are shown in jupyter notebook
        
        if 'backend_inline' in mpl.get_backend():
            # special context to support the `inline` backend
            try:
                from IPython.display import display  # @UnusedImport
                from ipywidgets import Output  # @UnusedImport
            except ImportError:
                context_class: Type[PlottingContextBase] = BasicPlottingContext
            else:
                context_class = JupyterPlottingContext
                
        else:
            # standard context for all other backends
            context_class = BasicPlottingContext
        
        return context_class(title=title, show=show)    
    
    elif isinstance(context, PlottingContextBase):
        # re-use an existing context
        context.title = title
        context.show = show
        return context
    
    elif isinstance(context, (mpl_axes.Axes, mpl.figure.Figure)):
        # create a basic context based on the given axes or figure
        return BasicPlottingContext(fig_or_ax=context, title=title, show=show)
    
    else:
        raise RuntimeError(f'Unknown plotting context `{context}`')
        
        

@contextlib.contextmanager
def napari_viewer(grid: 'GridBase', **kwargs):
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
        