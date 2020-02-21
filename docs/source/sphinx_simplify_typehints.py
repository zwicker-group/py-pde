"""
Simple sphinx plug-in that simplifies  type information in function signatures
"""

import collections
import re


# simple (literal) replacement rules
REPLACEMENTS = collections.OrderedDict([
    # Remove some unnecessary information in favor of a more compact style 
    ('Dict[KT, VT]', 'dict'),
    ('Dict[str, Any]', 'dict'),
    ('Optional[str]', 'str'),
    ('Optional[float]', 'float'),
    ('Optional[int]', 'int'),
    ('Optional[dict]', 'dict'),
    ('Optional[Dict[str, Any]]', 'dict'),
    # Complex types that can be represented by some descriptive name
    ('Union[dict, str, BCBase]', 'BoundaryData'),
    ('Union[Dict[str, BoundaryData], '
     'BoundaryData, Tuple[BoundaryData, BoundaryData]]', 'BoundaryPairData'),
    ('Union[Dict[str, BoundaryData], dict, str, BCBase, '
     'Tuple[BoundaryData, BoundaryData]]', 'BoundaryPairData'),
    ('Union[BoundaryPairData, Sequence[BoundaryPairData]]', 'BoundariesData'),
    ('Union[Dict[str, BoundaryData], dict, str, BCBase, '
     'Tuple[BoundaryData, BoundaryData], Sequence[BoundaryPairData]]',
        'BoundariesData'),
    ('Union[dict, str, BCBase, Tuple[Union[dict, str, BCBase], '
     'Union[dict, str, BCBase]], Sequence[Union[dict, str, BCBase, '
     'Tuple[Union[dict, str, BCBase], Union[dict, str, BCBase]]]]]',
        'BoundaryConditionData'),
    ('Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, '
     'Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]], '
     'Sequence[Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, '
     'Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]]]]]',
        'BoundaryConditionData'),
    ('Union[dict, str, BCBase, Tuple[Union[dict, str, BCBase], '
     'Union[dict, str, BCBase]], Sequence[Union[dict, str, BCBase, '
     'Tuple[Union[dict, str, BCBase], Union[dict, str, BCBase]]]], '
     'Sequence[BoundaryConditionData]]',
        'BoundaryConditionData'),
    ('Union[Dict[str, BoundaryData], dict, str, BCBase, '
     'Tuple[BoundaryData, BoundaryData], '
     'Sequence[BoundaryPairData], Sequence[BoundariesData]]',
        'BoundariesDataList'),
    ('Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, '
     'Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]], '
     'Sequence[Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, '
     'Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]]]], '
     'Sequence[BoundaryConditionData]]',
        'BoundariesDataList'),
    ('Union[List[Union[TrackerBase, str]], TrackerBase, str, None]',
        'TrackerData'),
])


# replacement rules based on regular expressions
REPLACEMENTS_REGEX = {
    # remove full package path and only leave the module/class identifier
    'pde\.(\w+\.)*': '',
}



def process_signature(app, what: str, name: str, obj, options, signature,
                      return_annotation):
    """ Process signature by applying replacement rules """
    if signature is not None:
        for key, value in REPLACEMENTS.items():
            signature = signature.replace(key, value)
        for key, value in REPLACEMENTS_REGEX.items():
            signature = re.sub(key, value, signature)
    return signature, return_annotation



def setup(app):
    """ set up hooks for this sphinx plugin """ 
    app.connect('autodoc-process-signature', process_signature)
    