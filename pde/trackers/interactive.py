"""
Special module for defining an interactive tracker that uses napari to display fields

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import multiprocessing as mp
import platform
import queue
import signal
import time
from typing import Any, Dict, Optional

from ..fields.base import FieldBase
from ..tools.docstrings import fill_in_docstring
from ..tools.plotting import napari_add_layers
from .base import InfoDict, TrackerBase
from .intervals import IntervalData


def napari_process(
    data_channel: mp.Queue,
    initial_data: Dict[str, Dict[str, Any]],
    t_initial: float = None,
    viewer_args: Dict[str, Any] = None,
):
    """:mod:`multiprocessing.Process` running `napari <https://napari.org>`__

    Args:
        data_channel (:class:`multiprocessing.Queue`):
            queue instance to receive data to view
        initial_data (dict):
            Initial data to be shown by napari. The layers are named according to
            the keys in the dictionary. The associated value needs to be a tuple,
            where the first item is a string indicating the type of the layer and
            the second carries the associated data
        t_initial (float):
            Initial time
        viewer_args (dict):
            Additional arguments passed to the napari viewer
    """
    logger = logging.getLogger(__name__ + ".napari_process")

    try:
        import napari
        from napari.qt import thread_worker
    except ModuleNotFoundError:
        logger.error(
            "The `napari` python module could not be found. This module needs to be "
            "installed to use the interactive tracker."
        )
        return

    logger.info("Start napari process")

    # ignore keyboard interrupts in this process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if viewer_args is None:
        viewer_args = {}

    # start napari Qt GUI
    with napari.gui_qt():

        # create and initialize the viewer
        viewer = napari.Viewer(**viewer_args)
        napari_add_layers(viewer, initial_data)

        # add time if given
        if t_initial is not None:
            from qtpy.QtWidgets import QLabel

            label = QLabel()
            label.setText(f"Time: {t_initial}")
            viewer.window.add_dock_widget(label)
        else:
            label = None

        def check_signal(msg: Optional[str]):
            """helper function that processes messages by the listener thread"""
            if msg is None:
                return  # do nothing
            elif msg == "close":
                viewer.close()
            else:
                raise RuntimeError(f"Unknown message from listener: {msg}")

        @thread_worker(connect={"yielded": check_signal})
        def update_listener():
            """helper thread that listens to the data_channel """
            logger.info("Start napari thread to receive data")

            # infinite loop waiting for events in the queue
            while True:
                # get all items from the queue and display the last update
                update_data = None  # nothing to update yet
                while True:
                    time.sleep(0.02)  # read queue with 50 fps
                    try:
                        action, data = data_channel.get(block=False)
                    except queue.Empty:
                        break

                    if action == "close":
                        logger.info("Forced closing of napari...")
                        yield "close"  # signal to napari process to shut down
                        break
                    elif action == "update":
                        update_data = data
                        # continue running until the queue is empty
                    else:
                        logger.warning(f"Unexpected action: {action}")

                # update napari view when there is data
                if update_data is not None:
                    logger.debug(f"Update napari layer...")
                    layer_data, t = update_data
                    if label is not None:
                        label.setText(f"Time: {t}")
                    for name, layer_data in layer_data.items():
                        viewer.layers[name].data = layer_data["data"]

                yield

        # start worker thread that listens to the data_channel
        update_listener()

    logger.info("Shutting down napari process")


class NapariViewer:
    """allows viewing and updating data in a separate napari process"""

    def __init__(self, state: FieldBase, t_initial: float = None):
        """
        Args:
            state (:class:`pde.fields.base.FieldBase`): The initial state to be shown
            t_initial (float): The initial time. If `None`, no time will be shown.
        """
        self._logger = logging.getLogger(__name__)

        # pick a suitable multiprocessing
        if platform.system() == "Darwin":
            context: mp.context.BaseContext = mp.get_context("spawn")
        else:
            context = mp.get_context()

        # create process that runs napari
        self.data_channel = context.Queue()
        initial_data = state._get_napari_data()
        viewer_args = {
            "axis_labels": state.grid.axes,
            "ndisplay": 3 if state.grid.dim >= 3 else 2,
        }
        args = (self.data_channel, initial_data, t_initial, viewer_args)

        self.proc = context.Process(target=napari_process, args=args)

        # start the process in the background
        try:
            self.proc.start()
        except RuntimeError:
            print()
            print("=" * 80)
            print(
                "It looks as if the main program did not use the multiprocessing "
                "safe-guard, which is necessary on some platforms. Please protect the "
                "main code of your program in the following way:"
            )
            print("")
            print("    if __name__ == '__main__':")
            print("        code ...")
            print("")
            print("The interactive Napari viewer could not be launched.")
            print("=" * 80)
            print()
            self._logger.exception("Could not launch napari process")

    def update(self, state: FieldBase, t: float):
        """update the state in the napari viewer

        Args:
            state (:class:`pde.fields.base.FieldBase`): The new state
            t (float): Current time
        """
        if self.proc.is_alive():
            try:
                data = (state._get_napari_data(), t)
                self.data_channel.put(("update", data), block=False)
            except queue.Full:
                pass  # could not write data
        else:
            try:
                self.data_channel.get(block=False)
            except queue.Empty:
                pass

    def close(self, force: bool = True):
        """closes the napari process

        Args:
            force (bool):
                Whether to force closing of the napari program. If this is `False`, this
                method blocks until the user closes napari manually.
        """
        if self.proc.is_alive() and force:
            # signal to napari process that it should be closed
            try:
                self.data_channel.put(("close", None))
            except RuntimeError:
                pass

        self.data_channel.close()
        self.data_channel.join_thread()

        if self.proc.is_alive():
            self.proc.join()


class InteractivePlotTracker(TrackerBase):
    """Tracker that shows the state live in an interactive napari instance

    Note:
        The interactive tracker uses the python :mod:`multiprocessing` module to run
        `napari <http://napari.org/>`__ externally. The multiprocessing module
        has limitations on some platforms, which requires some care when writing your
        own programs. In particular, the main method needs to be safe-guarded so that
        the main module can be imported again after spawning a new process. An
        established pattern that works is to introduce a function `main` in your code,
        which you call using the following pattern

        .. code-block:: python

            def main():
                # here goes your main code

            if __name__ == "__main__":
                main()

        The last two lines ensure that the `main` function is only called when the
        module is run initially and not again when it is re-imported.

    """

    name = "interactive"

    @fill_in_docstring
    def __init__(
        self,
        interval: IntervalData = "0:01",
        close: bool = True,
        show_time: bool = False,
    ):
        """
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
            close (bool):
                Flag indicating whether the napari window is closed automatically at the
                end of the simulation. If `False`, the tracker blocks when `finalize` is
                called until the user closes napari manually.
            show_time (bool):
                Whether to indicate the time
        """
        # initialize the tracker
        super().__init__(interval=interval)
        self.close = close
        self.show_time = show_time

    def initialize(self, state: FieldBase, info: InfoDict = None) -> float:
        """initialize the tracker with information about the simulation

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        if self.show_time:
            t_initial = 0 if info is None else info.get("t_start", 0)
        else:
            t_initial = None

        self._viewer = NapariViewer(state, t_initial=t_initial)
        return super().initialize(state, info=info)

    def handle(self, state: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        self._viewer.update(state, t)

    def finalize(self, info: InfoDict = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        self._viewer.close(force=self.close)
