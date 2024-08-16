#!/usr/bin/env python3 -m modelrunner.run --output data.hdf5 --method foreground
"""
Using :mod:`modelrunner`
========================

This example shows how `py-pde` can be combined with :mod:`modelrunner`. The magic first
line allows running this example as a script using :code:`./py_modelrunner.py`, which
runs the function defined below and stores all results in the file `data.hdf5`.

The results can be read by the following code

.. code-block:: python

    from modelrunner import Result

    r = Result.from_file("data.hdf5")
    r.result.plot()  # plots the final state
    r.storage["trajectory"]  # allows accessing the stored trajectory
"""

from pde import DiffusionPDE, ModelrunnerStorage, ScalarField, UnitGrid


def run(storage, diffusivity=0.1):
    """Function that runs the model.

    Args:
        storage (:mod:`~modelrunner.storage.group.StorageGroup`):
            Automatically supplied storage, to which extra data can be written
        diffusivity (float):
            Example for a parameter used in the model
    """
    # initialize the model
    state = ScalarField.random_uniform(UnitGrid([64, 64]), 0.2, 0.3)
    storage["initia_state"] = state  # store initial state with simulation
    eq = DiffusionPDE(diffusivity=diffusivity)

    # store trajectory in storage
    tracker = ModelrunnerStorage(storage, loc="trajectory").tracker(1)
    final_state = eq.solve(state, t_range=5, tracker=tracker)

    # returns the final state as the result, which will be stored by modelrunner
    return final_state
