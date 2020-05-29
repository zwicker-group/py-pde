'''
Classes for controlling plot output using context managers

.. autosummary::
   :nosignatures:

   BasicPlottingContext
   JupyterPlottingContext
   get_plotting_context

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import logging
import time
import warnings
from typing import Type  # @UnusedImport

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes



class PlottingContextBase(object):
    """ base class of the plotting contexts
    
    Example:
        context = PlottingContext()
        
        with context:
            plt.plot(...)
            plt.title(...)
    """
    
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
        self._logger = logging.getLogger(__name__)
    

    def __enter__(self):
        # start the plotting process
        if hasattr(self, 'fig'):
            plt.figure(self.fig.number)
    

    def __exit__(self, *exc): 
        if self.initial_plot:
            self.fig = plt.gcf()
            if len(self.fig.axes) > 1:
                self._title = plt.suptitle(self.title)
            else:
                self._title = plt.title(self.title)
            self.initial_plot = False
        else:
            self._title.set_text(self.title)
        
        
    def _show(self):
        """ show the updated plot """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # add a small pause to allow the GUI to run it's event loop
            plt.pause(0.001)


    def close(self):
        """ close the plot """
        # close matplotlib figure
        if hasattr(self, 'fig'):
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
            self._show()



class JupyterPlottingContext(PlottingContextBase):
    """ plotting in a jupyter widget """
    
    def __enter__(self):
        from ipywidgets import Output
        
        if self.initial_plot:
            self._logger.info('Initialize jupyter plotting context')
            self._ipython_out = Output()
            self._ipython_out.__enter__()
    
    
    def __exit__(self, *exc): 
        # need to copy the initial plot state, since it can be overwritten by
        # the call to super().__exit__
        initial_plot = self.initial_plot
        # finalize plot and set self.initial_plot = False
        super().__exit__(*exc)  
        if initial_plot:
            self._ipython_out.__exit__(*exc)
        if self.show:
            self._show(initial_plot)
            
        
    def _show(self, initial_plot):
        """ show the updated plot """
        from IPython.display import display

        if initial_plot:
            display(self._ipython_out)
        else:
            with self._ipython_out:
                display(self.fig)
        self._ipython_out.clear_output(wait=True)

        # add a small pause to allow the GUI to run it's event loop
        time.sleep(0.001)
        

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
                         show: bool = True):
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
        
        if show and 'ipykernel' in mpl.get_backend():
            try:
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
        