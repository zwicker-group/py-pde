'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import itertools
import pickle

import pytest

from ..parameters import (Parameter, DeprecatedParameter, HideParameter,
                          Parameterized, get_all_parameters)



def test_parameters():
    """ test mixing Parameterized """
        
    param = Parameter('a', 1, int, "help")
    assert isinstance(str(param), str)
    
    p_string = pickle.dumps(param)
    param_new = pickle.loads(p_string)
    assert param.__dict__ == param_new.__dict__
    assert param is not param_new
        
    class Test1(Parameterized):
        parameters_default = [param]

        
    t = Test1()
    assert t.parameters['a'] == 1
    assert t.get_parameter_default('a') == 1

    t = Test1(parameters={'a': 2})
    assert t.parameters['a'] == 2
    assert t.get_parameter_default('a') == 1
    
    with pytest.raises(ValueError):
        t = Test1(parameters={'b': 3})
    t = Test1()
    ps = t._parse_parameters({'b': 3}, check_validity=False)
    assert ps['a'] == 1
    assert ps['b'] == 3
    
    
    class Test2(Test1):
        # also test conversion of default parameters
        parameters_default = [Parameter('b', '2', int, "help")]
        
    t = Test2()
    assert t.parameters['a'] == 1
    assert t.parameters['b'] == 2
    
    t = Test2(parameters={'a': 10, 'b': 20})
    assert t.parameters['a'] == 10
    assert t.parameters['b'] == 20
    assert t.get_parameter_default('a') == 1
    assert t.get_parameter_default('b') == '2'
    with pytest.raises(KeyError):
        t.get_parameter_default('c')
        
    class Test3(Test2):
        # test overwriting defaults
        parameters_default = [Parameter('a', 3), Parameter('c', 4)]
        
    t = Test3()
    assert t.parameters['a'] == 3
    assert t.get_parameter_default('a') == 3
    assert set(t.parameters.keys()) == {'a', 'b', 'c'}
        
        
        
def test_parameters_simple():
    """ test adding parameters using a simple dictionary """
    
    class Test(Parameterized):
        parameters_default = {'a': 1}
        
    t = Test()
    assert t.parameters['a'] == 1
        
        
        
def test_parameter_help(capsys):
    """ test how parameters are shown """
    class Test1(Parameterized):
        parameters_default = [DeprecatedParameter('a', 1, int, "random string")]
        
    class Test2(Test1):
        parameters_default = [Parameter('b', 2, int, "another word")]
        
    t = Test2()
    for flags in itertools.combinations_with_replacement([True, False], 3):
        Test2.show_parameters(*flags)
        o1, e1 = capsys.readouterr()
        t.show_parameters(*flags)
        o2, e2 = capsys.readouterr()
        assert o1 == o2
        assert e1 == e2 == ''
    
    
    
def test_hidden_parameter():
    """ test how hidden parameters are handled """
    class Test1(Parameterized):
        parameters_default = [Parameter('a', 1), Parameter('b', 2)]
         
    assert Test1().parameters == {'a': 1, 'b': 2}
    
    class Test2(Test1):
        parameters_default = [HideParameter('b')]
        
    class Test2a(Parameterized):
        parameters_default = [Parameter('a', 1), Parameter('b', 2, hidden=True)]
        
    for t_class in [Test2, Test2a]:
        assert 'b' not in t_class.get_parameters()
        assert len(t_class.get_parameters()) == 1
        assert len(t_class.get_parameters(include_hidden=True)) == 2
        t2 = t_class()
        assert t2.parameters == {'a': 1, 'b': 2}
        assert t2.get_parameter_default('b') == 2
        with pytest.raises(ValueError):
            t2._parse_parameters({'b': 2}, check_validity=True,
                                 allow_hidden=False)
        
    class Test3(Test1):
        parameters_default = [Parameter('b', 3)]
    
    t3 = Test3()
    assert t3.parameters == {'a': 1, 'b': 3}
    assert t3.get_parameter_default('b') == 3
    
    
    
def test_get_all_parameters():
    """ test the get_all_parameters function """
    p1 = get_all_parameters()
    for key in ['value', 'description']:
        p2 = get_all_parameters(key)
        assert set(p1) == p2.keys()
    
    