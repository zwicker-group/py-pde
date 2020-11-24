"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools import output


def test_progress_bars():
    """ test progress bars """
    for pb_cls in [output.MockProgress, output.get_progress_bar_class()]:
        tot = 0
        for i in pb_cls(range(4)):
            tot += i
        assert tot == 6


def test_in_jupyter_notebook():
    """ test the function in_jupyter_notebook """
    assert isinstance(output.in_jupyter_notebook(), bool)


def test_display_progress(capsys):
    """ test whether this works """
    for _ in output.display_progress(range(2)):
        pass
    out, err = capsys.readouterr()
    assert out == ""
    assert len(err) > 0
