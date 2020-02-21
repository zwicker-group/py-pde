'''
Helper classes for dealing with instance parameters

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


import logging
from typing import Sequence, Dict, Any  # @UnusedImport

from ..tools.misc import import_class, hybridmethod



class Parameter():
    """ class representing information about parameters """
    
    def __init__(self, name: str, default_value=None,
                 cls=object, description: str = '',
                 deprecated: bool = False):
        """ initialize a single parameter
        
        Args:
            name (str): The name of the parameter
            default_value: The default value
            cls: The type of the parameter also used to convert it
            description (str): A description of the parameter
            bool (flag): Flag stating whether this parameter is deprecated
        """
        self.name = name
        self.default_value = default_value
        self.cls = cls
        self.description = description
        self.deprecated = deprecated
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
    
    
    def __hash__(self):
        """ custom hash function """
        return hash((str(self.name), self.convert(), self.cls.__name__,
                     self.description))
 
    
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
    
    
    def __lt__(self, other):
        """ allow comparison of parameter objects """
        return self.name.upper() < other.name.upper()
        
        
    def convert(self, value=None):
        """ converts a value into the correct type for this parameter.
        
        If value is not given, the default value is converted. Note that this
        does not necessarily copy the values, which could lead to unexpected
        effects where the default value is changed by an instance.
        
        Args:
            value: The value which will be converted
        """ 
        if value is None:
            value = self.default_value
            
        if self.cls is object:
            return value
        else:
            return self.cls(value)



class Parameterized():
    """ manages a dictionary of parameters assigned to classes """

    parameters_default: Sequence[Parameter] = []
    _subclasses: Dict[str, Any] = {}


    def __init__(self, parameters=None, check_validity=True):
        """ initialize the object with optional parameters that overwrite the
        default behavior
        
        Args:
            parameters (dict):
                A dictionary of parameters overwriting the defaults
            check_validity (bool):
                Determines whether an error is raised if there are keys in
                parameters that are not in the defaults
        """
        # initialize a logger that can be used in this instance
        self._logger = logging.getLogger(self.__class__.__module__)
        
        # initialize parameters with default ones from all parent classes
        self.parameters = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'parameters_default'):
                for p in cls.parameters_default:
                    self.parameters[p.name] = p.convert()
                
        # update parameters with the supplied ones
        if parameters is not None:
            if check_validity and any(key not in self.parameters
                                      for key in parameters):
                for key in parameters:
                    if key not in self.parameters:
                        raise ValueError('Parameter `{}` was provided in '
                                         'instance specific parameters but is '
                                         'not defined for the class `{}`'
                                         .format(key, self.__class__.__name__))
            
            self.parameters.update(parameters)


    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
            

    def get_parameter_default(self, name):
        """ return the default value for the parameter with `name` """
        if isinstance(self.parameters_default, dict):
            # assume that this is a simple dictionary for names and values
            # without further information on type and a description
            return self.parameters_default[name]
        
        else:
            for p in self.parameters_default:
                if p.name == name:
                    return p.default_value
            raise KeyError('Parameter `name` is not defined')
        
        
    @classmethod
    def _show_parameters(cls, description=False, sort=False,
                         show_deprecated=False, parameter_values=None):
        """ private method showing all parameters in human readable format
        
        Args:
            description (bool):
                Determines whether the parameter description is shown
            sort (bool):
                Determines whether all parameters are sorted
            show_deprecated (bool):
                Determines whether deprecated parameters are shown
            parameter_values (dict)
                A dictionary with values to show
        
        All flags default to `False`.
        """
        # set the templates for displaying the data 
        if description:
            template = '{name}: {type} = {value!r} ({description})'
            template_object = '{name} = {value!r} ({description})'
        else:
            template = '{name}: {type} = {value!r}'
            template_object = '{name} = {value!r}'
            
        # obtain the iterator containing the data
        params = cls.parameters_default
        if sort:
            params = sorted(params)
            
        # iterate over all parameters
        for param in params:
            if param.deprecated and not show_deprecated:
                continue  # skip deprecated parameter
            
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
    def show_parameters(cls, description=False, sort=False,  # @NoSelf
                        show_deprecated=False):
        """ private method showing all parameters in human readable format
        
        Args:
            description (bool):
                Determines whether the parameter description is shown
            sort (bool):
                Determines whether all parameters are sorted
            show_deprecated (bool):
                Determines whether deprecated parameters are shown
            parameter_values (dict)
                A dictionary with values to show
        
        All flags default to `False`.
        """
        cls._show_parameters(description, sort, show_deprecated)    


    @show_parameters.instancemethod  # type: ignore
    def show_parameters(self, description=False, sort=False,
                        show_deprecated=False, default_value=False):
        """ private method showing all parameters in human readable format
        
        Args:
            description (bool):
                Determines whether the parameter description is shown
            sort (bool):
                Determines whether all parameters are sorted
            show_deprecated (bool):
                Determines whether deprecated parameters are shown
            parameter_values (dict)
                A dictionary with values to show
        
        All flags default to `False`.
        """
        self._show_parameters(description, sort, show_deprecated,
                              None if default_value else self.parameters)
        
                