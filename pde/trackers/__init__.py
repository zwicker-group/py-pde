"""
Classes for tracking simulation results in controlled intervals

Trackers are classes that periodically receive the state of the simulation to
analyze, store, or output it.  The trackers defined in this module are:

.. autosummary::
   :nosignatures:

   ~trackers.CallbackTracker
   ~trackers.ProgressTracker
   ~trackers.PrintTracker
   ~trackers.PlotTracker
   ~trackers.DataTracker
   ~trackers.SteadyStateTracker
   ~trackers.RuntimeTracker
   ~trackers.ConsistencyTracker
   
Multiple trackers can be collected in a :class:`~base.TrackerCollection`,
which provides methods for handling them efficiently.

For each tracker, the interval at which it is called can be decided using one
of the following classes:

.. autosummary::
   :nosignatures:

   ~intervals.ConstantIntervals
   ~intervals.LogarithmicIntervals
   ~intervals.RealtimeIntervals
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from .intervals import (ConstantIntervals, LogarithmicIntervals,
                        RealtimeIntervals)
from .trackers import *
