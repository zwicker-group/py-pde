'''
Classes for controlling plot output using context managers

.. autosummary::
   :nosignatures:

   disable_interactive
   BasicPlottingContext
   JupyterPlottingContext
   get_plotting_context

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import contextlib
import logging
import warnings
from typing import Type  # @UnusedImport

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes



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



class PlottingContextBase(object):
    """ base class of the plotting contexts
    
    Example:
        context = PlottingContext()
        
        with context:
            plt.plot(...)
            plt.title(...)
    """
    
    supports_update: bool = True
    """ flag indicating whether the context supports that plots can be updated
    with out redrawing the entire plot """
    
    
    def __init__(self,
                 title: str = None,
                 filename: str = None,
                 show: bool = True):
        """ initialize the plotting context 
        
        Args:
            title (str): The shown in the plot
            filename (str): The filename to which the file is written
            show (bool): Flag determining whether plots are actually shown
        """
        self.title = title
        self.filename = filename
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
                raise RuntimeError('Plot figure does not contain axes')
            elif len(self.fig.axes) == 1:
                self._title = plt.title(self.title)
            else:
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
                 filename: str = None,
                 show: bool = True):
        """
        Args:
            fig_or_ax:
                If axes are given, they are used. If a figure is given, it is 
                set as active.
            title (str):
                The shown in the plot
            filename (str):
                The filename to which the file is written
            show (bool):
                Flag determining whether plots are actually shown
        """
        super().__init__(title=title, filename=filename, show=show)
        
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
    """ plotting in a jupyter widget """
    
    supports_update = False
    
    def __enter__(self):
        from IPython.display import display
        from ipywidgets import Output
        
        if self.initial_plot:
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
            plt.show()
            # ... also clear it the next time something is done        
            self._ipython_out.clear_output(wait=True)
            
        # stop capturing plots in the output widget
        self._ipython_out.__exit__(*exc)

 

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
                         filename: str = None,
                         show: bool = True) -> PlottingContextBase:
    """
    Args:
        context:
            An instance of :class:`PlottingContextBase` or an instance of
            :class:`mpl_axes.Axes` or :class:`mpl.figure.Figure` to determine
            where the plotting will happen. If omitted, the context is
            determined automatically.
        title (str):
            The shown in the plot
        filename (str):
            The filename to which the file is written
        show (bool):
            Determines whether the plot is shown while the simulation is
            running. If `False`, the files are created in the background.
            
    Returns:
        :class:`PlottingContextBase`: The plotting context
    """
    if context is None:
        # figure out whether plots are shown in jupyter notebook
        
        if 'ipykernel' in mpl.get_backend():
            try:
                from IPython.display import display  # @UnusedImport
                from ipywidgets import Output  # @UnusedImport
            except ImportError:
                context_class: Type[PlottingContextBase] = BasicPlottingContext
            else:
                context_class = JupyterPlottingContext
        else:
            context_class = BasicPlottingContext
        
        return context_class(title=title, filename=filename, show=show)    
    
    elif isinstance(context, PlottingContextBase):
        context.title = title
        context.filename = filename
        context.show = show
        return context
    
    elif isinstance(context, (mpl_axes.Axes, mpl.figure.Figure)):
        return BasicPlottingContext(fig_or_ax=context,
                                    title=title,
                                    filename=filename,
                                    show=show)
    
    raise RuntimeError(f'Unknown plotting context `{context}`')


        