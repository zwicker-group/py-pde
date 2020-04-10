'''
This module provides infrastructure for managing classes with parameters. One
aim is to allow easy management of inheritance of parameters.

.. autosummary::
   :nosignatures:

   Parameter
   DeprecatedParameter
   ObsoleteParameter
   Parameterized
   get_all_parameters

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


import logging
from collections import OrderedDict
from typing import Sequence, Dict, Any, Set  # @UnusedImport

from pde.tools.misc import import_class, hybridmethod, classproperty



class Parameter():
    """ class representing a single parameter """
    
    def __init__(self, name: str,
                 default_value=None,
                 cls=object,
                 description: str = ''):
        """ initialize a parameter 
        
        Args:
            name (str):
                The name of the parameter
            default_value:
                The default value
            cls:
                The type of the parameter, which is used for conversion
            description (str):
                A string describing the impact of this parameter. This
                description appears in the parameter help
        """
        self.name = name
        self.default_value = default_value
        self.cls = cls
        self.description = description
        if cls is not object and cls(default_value) != default_value:
            logging.warning('Default value `%s` does not seem to be of type '
                            '`%s`', name, cls.__name__)
    
        
    def __repr__(self):
        template = ('{class_name}(name="{name}", default_value='
                    '"{default_value}", cls="{cls_name}", '
                    'description="{description}")')
        return template.format(class_name=self.__class__.__name__,
                               cls_name=self.cls.__name__,
                               **self.__dict__)
    __str__ = __repr__
    
    
    def __getstate__(self):
        # replace the object class by its class path 
        return {'name': str(self.name),
                'default_value': self.convert(),
                'cls': object.__module__ + '.' + self.cls.__name__,
                'description': self.description}
    
        
    def __setstate__(self, state):
        # restore the object from the class path
        state['cls'] = import_class(state['cls'])
        # restore the state
        self.__dict__.update(state)
        
        
    def convert(self, value=None):
        """ converts a `value` into the correct type for this parameter. If
        `value` is not given, the default value is converted.
        
        Note that this does not make a copy of the values, which could lead to
        unexpected effects where the default value is changed by an instance.
        
        Args:
            value: The value to convert
        
        Returns:
            The converted value, which is of type `self.cls`
        """ 
        if value is None:
            value = self.default_value
            
        if self.cls is object:
            return value
        else:
            return self.cls(value)



class DeprecatedParameter(Parameter):
    """ a parameter that can still be used normally but is deprecated """
    pass



class ObsoleteParameter():
    """ parameter that was defined on a parent class, but does not apply to
    child class anymore. This class can be used on child classes to signal that
    a parameter from the parent class must not be inherited. """
    
    
    def __init__(self, name: str):
        """ 
        Args:
            name (str):
                The name of the parameter
        """
        self.name = name



class Parameterized():
    """ a mixin that manages the parameters of a class """

    parameters_default: Sequence[Parameter] = []
    _subclasses: Dict[str, 'Parameterized'] = {}


    def __init__(self, parameters: Dict[str, Any] = None,
                 check_validity: bool = True):
        """ initialize the object with optional parameters that overwrite the
        default behavior
        
        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. Call
                :meth:`~Parameterized.show_parameters` for details.
            check_validity (bool):
                Determines whether a `ValueError` is raised if there are keys in
                parameters that are not in the defaults. If `False`, additional
                items are simply stored in `self.parameters`
        """
        # initialize a logger that can be used in this instance
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # initialize parameters with default ones from all parent classes
        self.parameters: Dict[str, Any] = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'parameters_default'):
                for p in cls.parameters_default:  # type: ignore
                    if isinstance(p, ObsoleteParameter):
                        del self.parameters[p.name]
                    else:
                        self.parameters[p.name] = p.convert()
                
        # update parameters with the supplied ones
        if parameters is not None:
            if check_validity and any(key not in self.parameters
                                      for key in parameters):
                for key in parameters:
                    if key not in self.parameters:
                        raise ValueError(f'Parameter `{key}` was provided in '
                                         'instance specific parameters but is '
                                         'not defined for the class '
                                         f'`{self.__class__.__name__}`.')
            
            self.parameters.update(parameters)


    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclasses to reconstruct them later """
        # normalize the parameters_default attribute
        if (hasattr(cls, 'parameters_default') and
                isinstance(cls.parameters_default, dict)):
            # default parameters are given as a dictionary
            cls.parameters_default = \
                [Parameter(*args) for args in cls.parameters_default.items()]
        
        # register the subclasses
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
            
            
    @classproperty
    def _obsolete_parameters(cls) -> Set[str]:  # @NoSelf
        """ a list of obsolete parameters """
        obsolete = set() 
        for cls in reversed(cls.__mro__):  # type: ignore
            if hasattr(cls, 'parameters_default'):
                for p in cls.parameters_default:
                    if isinstance(p, ObsoleteParameter):
                        obsolete.add(p.name)
                    else:
                        obsolete.discard(p.name)
        return obsolete
            

    def get_parameter_default(self, name):
        """ return the default value for the parameter with `name` 
        
        Args:
            name (str): The parameter name
        """
        if name in self._obsolete_parameters:
            raise KeyError(f'Parameter `{name}` is obsolete')
            
        for cls in self.__class__.__mro__:
            if hasattr(cls, 'parameters_default'):
                for p in cls.parameters_default:
                    if p.name == name:
                        return p.default_value

        raise KeyError(f'Parameter `{name}` is not defined')
        

    @classmethod        
    def _get_parameters(cls, sort: bool = True, incl_deprecated: bool = False):
        parameters: Dict[str, Any] = {}
        for cls in reversed(cls.__mro__):
            if hasattr(cls, 'parameters_default'):
                for p in cls.parameters_default:
                    if isinstance(p, ObsoleteParameter):
                        del parameters[p.name]
                    elif (incl_deprecated or
                            not isinstance(p, DeprecatedParameter)):
                        parameters[p.name] = p
                        
        if sort:
            parameters = OrderedDict(sorted(parameters.items()))
        return parameters
        
        
    @classmethod
    def _show_parameters(cls, description: bool = False,
                         sort: bool = False,
                         show_deprecated: bool = False,
                         parameter_values: Dict[str, Any] = None):
        """ private method showing all parameters in human readable format
        
        Args:
            description (bool):
                Flag determining whether the parameter description is shown
            sort (bool):
                Flag determining whether the parameters are sorted
            show_deprecated (bool):
                Flag determining whether deprecated parameters are shown
            parameter_values (dict):
                A dictionary with values to show. Parameters not in this
                dictionary are shown with their default value.
        
        All flags default to `False`.
        """
        # set the templates for displaying the data 
        if description:
            template = '{name}: {type} = {value!r} ({description})'
            template_object = '{name} = {value!r} ({description})'
        else:
            template = '{name}: {type} = {value!r}'
            template_object = '{name} = {value!r}'
            
        # iterate over all parameters
        params = cls._get_parameters(sort=sort, incl_deprecated=show_deprecated)
        for param in params.values():
            # initialize the data to show
            data = {'name': param.name,
                    'type': param.cls.__name__,
                    'description': param.description}
            
            # determine the value to show
            if parameter_values is None:
                data['value'] = param.default_value
            else:
                data['value'] = parameter_values[param.name]
            
            # print the data
            if param.cls is object:
                print((template_object.format(**data)))
            else:
                print((template.format(**data)))
            

    @hybridmethod
    def show_parameters(cls, description: bool = False,  # @NoSelf
                        sort: bool = False,
                        show_deprecated: bool = False):
        """ show all parameters in human readable format
        
        Args:
            description (bool):
                Flag determining whether the parameter description is shown
            sort (bool):
                Flag determining whether the parameters are sorted
            show_deprecated (bool):
                Flag determining whether deprecated parameters are shown
        
        All flags default to `False`.
        """
        cls._show_parameters(description, sort, show_deprecated)    


    @show_parameters.instancemethod  # type: ignore
    def show_parameters(self, description: bool = False,  # @NoSelf
                        sort: bool = False,
                        show_deprecated: bool = False,
                        default_value: bool = False):
        """ show all parameters in human readable format
        
        Args:
            description (bool):
                Flag determining whether the parameter description is shown
            sort (bool):
                Flag determining whether the parameters are sorted
            show_deprecated (bool):
                Flag determining whether deprecated parameters are shown
            default_value (bool):
                Flag determining whether the default values or the current
                values are shown
        
        All flags default to `False`.
        """
        self._show_parameters(description, sort, show_deprecated,
                              None if default_value else self.parameters)
        
        

def get_all_parameters(data: str = None) -> Dict[str, Any]:
    """ get a dictionary with all parameters of all registered classes """
    result = {}
    for cls_name, cls in Parameterized._subclasses.items():
        if data is None:
            parameters = set(cls._get_parameters().keys())
        elif data == 'value':
            parameters = {k: v.default_value  # type: ignore
                          for k, v in cls._get_parameters().items()}
        elif data == 'description':
            parameters = {k: v.description  # type: ignore
                          for k, v in cls._get_parameters().items()}
        else:
            raise ValueError(f'Cannot interpret data `{data}`')
        
        result[cls_name] = parameters
    return result
        
                