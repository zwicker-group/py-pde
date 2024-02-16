"""
Python functions for handling output

.. autosummary::
   :nosignatures:

   get_progress_bar_class
   display_progress
   in_jupyter_notebook
   BasicOutput
   JupyterOutput

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import sys
from abc import ABCMeta, abstractmethod
from typing import List, Type  # @UnusedImport

import tqdm


def get_progress_bar_class(fancy: bool = True):
    """returns a class that behaves as progress bar.

    This either uses classes from the optional `tqdm` package or a simple version that
    writes dots to stderr, if the class it not available.

    Args:
        fancy (bool):
            Flag determining whether a fancy progress bar should be used in jupyter
            notebooks (if :mod:`ipywidgets` is installed)
    """
    if fancy:
        # try using notebook progress bar
        try:
            # check whether progress bar can use a widget
            import ipywidgets  # @UnusedImport
        except ImportError:
            # widgets are not available => use standard tqdm
            progress_bar_class = tqdm.tqdm
        else:
            # use the fancier version of the progress bar in jupyter
            from tqdm.auto import tqdm as progress_bar_class  # type: ignore
    else:
        # only import text progress bar
        progress_bar_class = tqdm.tqdm

    return progress_bar_class


def display_progress(iterator, total=None, enabled=True, **kwargs):
    r"""
    displays a progress bar when iterating

    Args:
        iterator (iter): The iterator
        total (int): Total number of steps
        enabled (bool): Flag determining whether the progress is display
        **kwargs: All extra arguments are forwarded to the progress bar class

    Returns:
        A class that behaves as the original iterator, but shows the progress
        alongside iteration.
    """
    if not enabled:
        return iterator

    return get_progress_bar_class()(iterator, total=total, **kwargs)


class OutputBase(metaclass=ABCMeta):
    """base class for output management"""

    @abstractmethod
    def __call__(self, line: str):
        """add a line of text

        Args:
            line (str): The text line
        """

    @abstractmethod
    def show(self):
        """shows the actual text"""


class BasicOutput(OutputBase):
    """class that writes text line to stdout"""

    def __init__(self, stream=sys.stdout):
        """
        Args:
            stream: The stream where the lines are written
        """
        self.stream = stream

    def __call__(self, line: str):
        self.stream.write(line + "\n")

    def show(self):
        self.stream.flush()


class JupyterOutput(OutputBase):
    """class that writes text lines as html in a jupyter cell"""

    def __init__(self, header: str = "", footer: str = ""):
        """
        Args:
            header (str): The html code written before all lines
            footer (str): The html code written after all lines
        """
        self.header = header
        self.footer = footer
        self.lines: list[str] = []

    def __call__(self, line: str):
        self.lines.append(line)

    def show(self):
        try:
            from IPython.display import HTML, display
        except ImportError:
            print("\n".join(self.lines))
        else:
            html = self.header + "".join(self.lines) + self.footer
            display(HTML(html))


def in_jupyter_notebook() -> bool:
    """checks whether we are in a jupyter notebook"""
    try:
        from IPython import display, get_ipython  # @UnusedImport
    except ImportError:
        return False

    try:
        ipython_config = get_ipython().config
    except AttributeError:
        return False

    return "IPKernelApp" in ipython_config
