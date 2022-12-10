"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools.parse_duration import parse_duration


def test_parse_duration():
    """test function signature checks"""

    def p(value):
        return parse_duration(value).total_seconds()

    assert p("0") == 0
    assert p("1") == 1
    assert p("1:2") == 62
    assert p("1:2:3") == 3723
