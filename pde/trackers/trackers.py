"""
Module defining classes for tracking results from simulations.

The trackers defined in this module are:

.. autosummary::
   :nosignatures:

   CallbackTracker
   ProgressTracker
   PrintTracker
   PlotTracker
   DataTracker
   SteadyStateTracker
   RuntimeTracker
   ConsistencyTracker
   MaterialConservationTracker

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from datetime import timedelta
import inspect
import sys
import os.path
import time
from typing import (Callable, Optional, Union, IO, List, Any,  # @UnusedImport
                    Dict, TYPE_CHECKING)

import numpy as np

from .base import TrackerBase, InfoDict, FinishedSimulation, Real
from .intervals import IntervalData, RealtimeIntervals
from ..fields.base import FieldBase
from ..fields import FieldCollection
from ..tools.parse_duration import parse_duration
from ..tools.misc import get_progress_bar_class
from ..tools.docstrings import fill_in_docstring



if TYPE_CHECKING:
    import pandas  # @UnusedImport



class CallbackTracker(TrackerBase):
    """ Tracker that calls a function periodically """
    
    @fill_in_docstring
    def __init__(self, func: Callable,
                 interval: IntervalData = 1):
        """ 
        Args:
            func: The function to call periodically. The function signature
                should be `(state)` or `(state, time)`, where `state` contains
                the current state as an instance of
                :class:`~pde.fields.FieldBase` and `time` is a
                float value indicating the current time. Note that only a view
                of the state is supplied, implying that a copy needs to be made
                if the data should be stored.
            interval:
                {ARG_TRACKER_INTERVAL}
        """
        super().__init__(interval=interval)
        self._callback = func
        self._num_args = len(inspect.signature(func).parameters)
        if not 0 < self._num_args < 3:
            raise ValueError('`func` must be a function accepting one or two '
                             f'arguments, not {self._num_args}') 
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
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
    """ Tracker that shows the progress of the simulation """
            
    name = 'progress'


    @fill_in_docstring            
    def __init__(self, interval: IntervalData = None,
                 ndigits: int = 5, leave: bool = True):
        """
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
                The default value `None` updates the progress bar approximately
                every (real) second.
            ndigits (int): The number of digits after the decimal point that are
                shown maximally.
            leave (bool): Whether to leave the progress bar after the simulation
                has finished (default: True)
        """ 
        if interval is None:
            # print every second by default
            interval = RealtimeIntervals(duration=1)
        
        super().__init__(interval=interval)
        self.ndigits = ndigits
        self.leave = leave
        

    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """ initialize the tracker with information about the simulation
        
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
        controller_info = {} if info is None else info.get('controller', {})
        
        # initialize the progress bar
        pb_cls = get_progress_bar_class()
        self.progress_bar = pb_cls(total=controller_info.get('t_end'),
                                   initial=controller_info.get('t_start', 0),
                                   leave=self.leave)
        self.progress_bar.set_description('Initializing')

        return result
            
            
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
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
        self.progress_bar.set_description('')
                
        
    def finalize(self, info: InfoDict = None) -> None:
        """ finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation        
        """
        super().finalize(info)
        self.progress_bar.set_description('')

        # limit progress bar to 100%
        controller_info = {} if info is None else info.get('controller', {}) 
        t_final = controller_info.get('t_final', -np.inf)
        t_end = controller_info.get('t_end', -np.inf)
        if t_final >= t_end and self.progress_bar.total:
            self.progress_bar.n = self.progress_bar.total
            self.progress_bar.refresh()
        
        if (controller_info.get('successful', False) and self.leave and
                hasattr(self.progress_bar, 'sp')):
            # show progress bar in green if simulation was successful. We
            # need to overwrite the default behavior (and disable the
            # progress bar) since reaching steady state means the simulation
            # was successful even though it did not reach t_final
            try:
                self.progress_bar.sp(bar_style='success')
            except TypeError:
                self.progress_bar.close()
            else:
                self.disable = True
        else:
            self.progress_bar.close()
            
            
    def __del__(self):
        if hasattr(self, 'progress_bar') and not self.progress_bar.disable:
            self.progress_bar.close()



class PrintTracker(TrackerBase):
    """ Tracker that prints data to a stream (default: stdout) """
    
    name = 'print'
    
    
    @fill_in_docstring
    def __init__(self, interval: IntervalData = 1,
                 stream: IO[str] = sys.stdout):
        """
        
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
            stream:
                The stream used for printing
        """
        super().__init__(interval=interval)
        self.stream = stream
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
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
    """ Tracker that plots data on screen, to files, or writes a movie """
     
    name = 'plot'
    
    @fill_in_docstring
    def __init__(self, interval: IntervalData = 1,
                 title: str = '',
                 output_file: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 movie_file: Optional[str] = None,
                 show: bool = True,
                 close_final: bool = False,
                 plot_arguments: Dict[str, Any] = None):
        """
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
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
            plot_arguments (dict):
                Extra arguments supplied to the plot call
        """
        super().__init__(interval=interval)
        self.title = title
        self.output_file = output_file
        self.output_folder = output_folder
        self.show = show
        self.close_final = close_final
        self.plot_arguments = {} if plot_arguments is None else plot_arguments
        
        if movie_file is not None or output_folder is not None:
            from ..visualization.movies import Movie
            movie = Movie(filename=movie_file, image_folder=output_folder)
            self.movie: Optional[Movie] = movie
            self.movie._start()  # initialize movie
        else:
            self.movie = None
         
     
    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """ initialize the tracker with information about the simulation
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation
                
        Returns:
            float: The first time the tracker needs to handle data
        """
        # initialize the plotting context
        from ..visualization.contexts import get_plotting_context
        title = self.title + 'Initializing...'
        self._context = get_plotting_context(title=title, show=self.show)
        
        self.plot_arguments['show'] = False  # this is handled by the context
        
        # do the actual plotting
        with self._context:
            self._plot_reference = field.plot(**self.plot_arguments)

        # determine whether this plot can be updated in the following
        self._update_plot = (self._context.supports_update and 
                             hasattr(field, 'update_plot') and
                             self._plot_reference is not None)
            
        return super().initialize(field, info=info)
        
         
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        self._context.title = f'{self.title}Time: {t:g}'
        
        # update the plot in the correct plotting context
        with self._context:
            if self._update_plot:
                field.update_plot(reference=self._plot_reference)
            else:
                field.plot(**self.plot_arguments)
                
        if self.output_file and self._context.fig is not None:
            self._context.fig.savefig(self.output_file)
        if self.movie:
            self.movie.add_figure(self._context.fig)
          

    def finalize(self, info: InfoDict = None) -> None:
        """ finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation        
        """
        super().finalize(info)
        if self.movie:
            if self.movie.filename:
                # write out movie file if requested
                self._logger.info(f'Writing movie to {self.movie.filename}...')
                self.movie.save()
            # finalize movie (e.g. delete temporary files)
            self.movie._end()
            
        if not self.show or self.close_final:
            self._context.close()
    
              
    
class DataTracker(CallbackTracker):
    """ Tracker that stores custom data obtained by calling a function
    
    Attributes:
        times (list):
            The time points at which the data is stored
        data (list):
            The actually stored data, which is a list of the objects returned by
            the callback function. 
    """
    
    @fill_in_docstring
    def __init__(self, func: Callable,
                 interval: IntervalData = 1,
                 filename: str = None):
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
            interval:
                {ARG_TRACKER_INTERVAL}
            filename (str):
                A path to a file to which the data is written at the end of the
                tracking. The data format will be determined by the extension
                of the filename. '.pickle' indicates a python pickle file
                storing a tuple `(self.times, self.data)`, whereas any other
                data format requires :mod:`pandas`.  
        """
        super().__init__(func=func, interval=interval)
        self.filename = filename
        self.times: List[float] = []
        self.data: List[Any] = []
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
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
        
        
    def finalize(self, info: InfoDict = None) -> None:
        """ finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation        
        """
        super().finalize(info)
        if self.filename:
            self.to_file(self.filename)
        
        
    @property
    def dataframe(self) -> "pandas.DataFrame":
        """ :class:`pandas.DataFrame`: the data in a dataframe
        
        If `func` returns a dictionary, the keys are used as column names.
        Otherwise, the returned data is enumerated starting with '0'. In any
        case the time point at which the data was recorded is stored in the
        column 'time'.
        """
        import pandas as pd
        df = pd.DataFrame(self.data)
        # insert the times and use them as an index
        df.insert(0, 'time', self.times)
        return df
    
    
    def to_file(self, filename: str, **kwargs):
        r""" store data in a file
        
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
        if extension == '.pickle':
            # default 
            import pickle
            with open(filename, "wb") as fp:
                pickle.dump((self.times, self.data), fp, **kwargs)
            
        elif extension == '.csv':
            self.dataframe.to_csv(filename, **kwargs)
        elif extension == '.json':
            self.dataframe.to_json(filename, **kwargs)
        elif extension in {'.xls', '.xlsx'}:
            self.dataframe.to_excel(filename, **kwargs)
        else:
            raise ValueError(f'Unsupported file extension `{extension}`')
            
            
            
class SteadyStateTracker(TrackerBase):
    """ Tracker that interrupts the simulation once steady state is reached
    
    Steady state is obtained when the state does not change anymore. This is the
    case when the derivative is close to zero.
    """

    name = 'steady_state'


    @fill_in_docstring
    def __init__(self, interval: IntervalData = None,
                 atol: float = 1e-8,
                 rtol: float = 1e-5):
        """
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
                The default value `None` checks for the steady state
                approximately every (real) second.
            atol (float): Absolute tolerance that must be reached to abort the
                simulation
            rtol (float): Relative tolerance that must be reached to abort the
                simulation
        """ 
        if interval is None:
            interval = RealtimeIntervals(duration=1)
        super().__init__(interval=interval)
        self.atol = atol 
        self.rtol = rtol
        self._last_data = None
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if self._last_data is not None:
            # scale with dt to make test independent of dt
            atol = self.atol * self.interval.dt
            rtol = self.rtol * self.interval.dt
            if np.allclose(self._last_data, field.data,
                           rtol=rtol, atol=atol, equal_nan=True):
                raise FinishedSimulation('Reached stationary state')
            
        self._last_data = field.data.copy()  # store data from last timestep
            


class RuntimeTracker(TrackerBase):
    """ Tracker that interrupts the simulation once a duration has passed """


    @fill_in_docstring
    def __init__(self, max_runtime: Union[Real, str],
                 interval: IntervalData = 1):  
        """
        Args:
            max_runtime (float or str):
                The maximal runtime of the simulation. If the runtime is
                exceeded, the simulation is interrupted. Values can be either
                given as a number (interpreted as seconds) or as a string, which
                is then parsed using the function
                :func:`~pde.tools.parse_duration.parse_duration`.
            interval:
                {ARG_TRACKER_INTERVAL}
        """
        super().__init__(interval=interval)
        
        try:
            self.max_runtime = float(max_runtime)
        except ValueError:
            td = parse_duration(str(max_runtime))
            self.max_runtime = td.total_seconds()


    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """ 
        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation        
                
        Returns:
            float: The first time the tracker needs to handle data
        """
        self.max_time = time.time() + self.max_runtime
        return super().initialize(field, info)
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if time.time() > self.max_time:
            dt = timedelta(seconds=self.max_runtime)
            raise FinishedSimulation(f'Reached maximal runtime of {str(dt)}')

            
            
class ConsistencyTracker(TrackerBase):
    """ Tracker that interrupts the simulation when the state is not finite """ 

    name = 'consistency'
        
    
    @fill_in_docstring
    def __init__(self, interval: IntervalData = None):
        """
        Args:
            interval: 
                {ARG_TRACKER_INTERVAL}
                The default value `None` checks for consistency approximately
                every (real) second.
        """ 
        if interval is None:
            interval = RealtimeIntervals(duration=1)
        super().__init__(interval=interval)
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        if not np.all(np.isfinite(field.data)):
            raise StopIteration('Field was not finite')
            
        self._last = field.data.copy()  # store data from last timestep
            


class MaterialConservationTracker(TrackerBase):
    """ Ensure that the amount of material is conserved """

    name = 'material_conservation'


    @fill_in_docstring
    def __init__(self, interval: IntervalData = 1,
                 atol: float = 1e-4,
                 rtol: float = 1e-4):
        """
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
            atol (float):
                Absolute tolerance for amount deviations
            rtol (float):
                Relative tolerance for amount deviations
        """
        super().__init__(interval=interval)
        self.atol = atol 
        self.rtol = rtol
        
        
    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
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
        """ handle data supplied to this tracker
        
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
                msg = f'Material of field {np.flatnonzero(~c)} is not conserved'
            else:
                msg = f'Material is not conserved'
            raise StopIteration(msg)
            
            
            
__all__ = ['CallbackTracker', 'ProgressTracker', 'PrintTracker', 'PlotTracker',
           'DataTracker', 'SteadyStateTracker', 'RuntimeTracker',
           'ConsistencyTracker', 'MaterialConservationTracker']
