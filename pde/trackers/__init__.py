"""Classes for tracking simulation results in controlled interrupts.

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

Some trackers can also be referenced by name for convenience when using them in
simulations. The lit of supported names is returned by
:func:`~pde.trackers.base.get_named_trackers`.

Multiple trackers can be collected in a :class:`~base.TrackerCollection`, which provides
methods for handling them efficiently. Moreover, custom trackers can be implemented by
deriving from :class:`~.trackers.base.TrackerBase`. Note that trackers generally receive
a view into the current state, implying that they can adjust the state by modifying it
in-place. Moreover, trackers can abort the simulation by raising the special exception
:class:`StopIteration`.


For each tracker, the time at which the simulation is interrupted can be decided using
one of the following classes:

.. autosummary::
   :nosignatures:

   ~interrupts.FixedInterrupts
   ~interrupts.ConstantInterrupts
   ~interrupts.LogarithmicInterrupts
   ~interrupts.GeometricInterrupts
   ~interrupts.RealtimeInterrupts

In particular, interrupts can be specified conveniently using
:func:`~interrupts.parse_interrupt`.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .base import get_named_trackers
from .interactive import InteractivePlotTracker
from .interrupts import (
    ConstantInterrupts,
    FixedInterrupts,
    LogarithmicInterrupts,
    RealtimeInterrupts,
    parse_interrupt,
)
from .trackers import *
