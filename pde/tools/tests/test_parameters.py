"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools
import logging
import pickle

import numpy as np
import pytest
from pde.tools.parameters import (
    DeprecatedParameter,
    HideParameter,
    Parameter,
    Parameterized,
    get_all_parameters,
    sphinx_display_parameters,
)


def test_parameters():
    """ test mixing Parameterized """

    param = Parameter("a", 1, int, "help", extra={"b": 3})
    assert isinstance(str(param), str)

    p_string = pickle.dumps(param)
    param_new = pickle.loads(p_string)
    assert param.__dict__ == param_new.__dict__
    assert param is not param_new
    assert param_new.extra["b"] == 3

    class Test1(Parameterized):
        parameters_default = [param]

    t = Test1()
    assert t.parameters["a"] == 1
    assert t.get_parameter_default("a") == 1

    t = Test1(parameters={"a": 2})
    assert t.parameters["a"] == 2
    assert t.get_parameter_default("a") == 1

    with pytest.raises(ValueError):
        t = Test1(parameters={"b": 3})
    t = Test1()
    ps = t._parse_parameters({"b": 3}, check_validity=False)
    assert ps["a"] == 1
    assert ps["b"] == 3

    class Test2(Test1):
        # also test conversion of default parameters
        parameters_default = [Parameter("b", "2", int, "help")]

    t = Test2()
    assert t.parameters["a"] == 1
    assert t.parameters["b"] == 2

    t = Test2(parameters={"a": 10, "b": 20})
    assert t.parameters["a"] == 10
    assert t.parameters["b"] == 20
    assert t.get_parameter_default("a") == 1
    assert t.get_parameter_default("b") == "2"
    with pytest.raises(KeyError):
        t.get_parameter_default("c")

    class Test3(Test2):
        # test overwriting defaults
        parameters_default = [Parameter("a", 3), Parameter("c", 4)]

    t = Test3()
    assert t.parameters["a"] == 3
    assert t.get_parameter_default("a") == 3
    assert set(t.parameters.keys()) == {"a", "b", "c"}

    # test get_all_parameters function after having used Parameters
    p1 = get_all_parameters()
    for key in ["value", "description"]:
        p2 = get_all_parameters(key)
        assert set(p1) == p2.keys()

    # test whether sphinx_display_parameters runs
    lines = [":param parameters:"]
    sphinx_display_parameters(None, "class", "Test1", Test1, None, lines)
    assert len(lines) > 1


def test_parameters_simple():
    """ test adding parameters using a simple dictionary """

    class Test(Parameterized):
        parameters_default = {"a": 1}

    t = Test()
    assert t.parameters["a"] == 1


def test_parameter_help(monkeypatch, capsys):
    """ test how parameters are shown """

    class Test1(Parameterized):
        parameters_default = [DeprecatedParameter("a", 1, int, "random string")]

    class Test2(Test1):
        parameters_default = [Parameter("b", 2, int, "another word")]

    t = Test2()
    for in_jupyter in [False, True]:
        monkeypatch.setattr("pde.tools.output.in_jupyter_notebook", lambda: in_jupyter)

        for flags in itertools.combinations_with_replacement([True, False], 3):
            Test2.show_parameters(*flags)
            o1, e1 = capsys.readouterr()
            t.show_parameters(*flags)
            o2, e2 = capsys.readouterr()
            assert o1 == o2
            assert e1 == e2 == ""


def test_hidden_parameter():
    """ test how hidden parameters are handled """

    class Test1(Parameterized):
        parameters_default = [Parameter("a", 1), Parameter("b", 2)]

    assert Test1().parameters == {"a": 1, "b": 2}

    class Test2(Test1):
        parameters_default = [HideParameter("b")]

    class Test2a(Parameterized):
        parameters_default = [Parameter("a", 1), Parameter("b", 2, hidden=True)]

    for t_class in [Test2, Test2a]:
        assert "b" not in t_class.get_parameters()
        assert len(t_class.get_parameters()) == 1
        assert len(t_class.get_parameters(include_hidden=True)) == 2
        t2 = t_class()
        assert t2.parameters == {"a": 1, "b": 2}
        assert t2.get_parameter_default("b") == 2
        with pytest.raises(ValueError):
            t2._parse_parameters({"b": 2}, check_validity=True, allow_hidden=False)

    class Test3(Test1):
        parameters_default = [Parameter("b", 3)]

    t3 = Test3()
    assert t3.parameters == {"a": 1, "b": 3}
    assert t3.get_parameter_default("b") == 3


def test_convert_default_values(caplog):
    """ test how default values are handled """

    class Test1(Parameterized):
        parameters_default = [Parameter("a", 1, float)]

    with caplog.at_level(logging.WARNING):
        t1 = Test1()
    assert "Default value" not in caplog.text
    assert isinstance(t1.parameters["a"], float)

    class Test2(Parameterized):
        parameters_default = [Parameter("a", np.arange(3), np.array)]

    t2 = Test2()
    np.testing.assert_equal(t2.parameters["a"], np.arange(3))

    class Test3(Parameterized):
        parameters_default = [Parameter("a", [0, 1, 2], np.array)]

    t3 = Test3()
    np.testing.assert_equal(t3.parameters["a"], np.arange(3))

    class Test4(Parameterized):
        parameters_default = [Parameter("a", 1, str)]

    with caplog.at_level(logging.WARNING):
        t4 = Test4()
    assert "Default value" in caplog.text
    np.testing.assert_equal(t4.parameters["a"], "1")
