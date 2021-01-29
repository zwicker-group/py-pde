"""
Classes for tracking simulation results in controlled intervals

Trackers are classes that periodically receive the state of the simulation to analyze,
store, or output it. The trackers defined in this module are:

.. autosummary::
   :nosignatures:

   ~trackers.CallbackTracker
   ~trackers.ProgressTracker
   ~trackers.PrintTracker
   ~trackers.PlotTracker
   ~trackers.LivePlotTracker
   ~trackers.DataTracker
   ~trackers.SteadyStateTracker
   ~trackers.RuntimeTracker
   ~trackers.ConsistencyTracker
   ~interactive.InteractivePlotTracker
   
Multiple trackers can be collected in a :class:`~base.TrackerCollection`, which provides
methods for handling them efficiently. Moreover, custom trackers can be implemented by
deriving from :class:`~.trackers.base.TrackerBase`. Note that trackers generally receive
a view into the current state, implying that they can adjust the state by modifying it
in-place. Moreover, trackers can interrupt the simulation by raising the special
exception :class:`StopIteration`.


For each tracker, the interval at which it is called can be decided using one
of the following classes:

.. autosummary::
   :nosignatures:

   ~intervals.ConstantIntervals
   ~intervals.LogarithmicIntervals
   ~intervals.RealtimeIntervals
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from .interactive import InteractivePlotTracker
from .intervals import ConstantIntervals, LogarithmicIntervals, RealtimeIntervals
from .trackers import *
