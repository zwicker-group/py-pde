'''
Check the parameters module

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from ..parameters import Parameter, Parameterized



def test_parameter_class():
    """ test Parameter class """
    p = Parameter(name='int', default_value=1, cls=int, description='test',
                  deprecated=False)
    
    assert p.name == 'int'
    assert p.convert(1.1) == 1
    
    q = Parameter('new')
    q.__setstate__(p.__getstate__())
    assert p.__dict__ == q.__dict__
    
    

def test_parameterized_class(capsys):
    """ test Parameter class """
    
    class Test(Parameterized):
        parameters_default = [Parameter(name='int', default_value=1, cls=int,
                                        description='test', deprecated=False)]
    
    Test.show_parameters(description=True)
    captured = capsys.readouterr()
    assert 'int' in captured.out
    assert '1' in captured.out
    
    t = Test()
    assert t.parameters['int'] == 1
    
    t = Test(parameters={'int': 2})
    assert t.parameters['int'] == 2
    assert t.get_parameter_default('int') == 1
    t.show_parameters(description=True)
    captured = capsys.readouterr()
    assert 'int' in captured.out
    assert '2' in captured.out
    
    