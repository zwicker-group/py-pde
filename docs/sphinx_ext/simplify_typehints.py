"""
Simple sphinx plug-in that simplifies  type information in function signatures
"""

import re

# simple (literal) replacement rules
REPLACEMENTS = [
    # numbers and numerical arrays
    ("Union[int, float, complex, numpy.generic, numpy.ndarray]", "NumberOrArray"),
    ("Union[int, float, complex, numpy.ndarray]", "NumberOrArray"),
    ("Union[int, float, complex]", "Number"),
    (
        "Optional[Union[_SupportsArray[dtype], _NestedSequence[_SupportsArray[dtype]], "
        "bool, int, float, complex, str, bytes, _NestedSequence[Union[bool, int, "
        "float, complex, str, bytes]]]]",
        "NumberOrArray",
    ),
    (
        "Union[dtype[Any], None, Type[Any], _SupportsDType[dtype[Any]], str, "
        "Tuple[Any, int], Tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], "
        "List[Any], _DTypeDict, Tuple[Any, Any]]",
        "DType",
    ),
    # Complex types describing the boundary conditions
    (
        "Dict[str, Dict | str | BCBase] | Dict | str | BCBase | "
        "Tuple[Dict | str | BCBase, Dict | str | BCBase] | BoundaryAxisBase | "
        "Sequence[Dict[str, Dict | str | BCBase] | Dict | str | BCBase | "
        "Tuple[Dict | str | BCBase, Dict | str | BCBase] | BoundaryAxisBase]",
        "BoundariesData",
    ),
    (
        "Dict[str, Dict | str | BCBase] | Dict | str | BCBase | "
        "Tuple[Dict | str | BCBase, Dict | str | BCBase] | BoundaryAxisBase",
        "BoundariesPairData",
    ),
    ("Dict | str | BCBase", "BoundaryData"),
    # Other compound data types
    ("Union[List[Union[TrackerBase, str]], TrackerBase, str, None]", "TrackerData"),
    (
        "Optional[Union[List[Union[TrackerBase, str]], TrackerBase, str]]",
        "TrackerData",
    ),
]


# replacement rules based on regular expressions
REPLACEMENTS_REGEX = {
    # remove full package path and only leave the module/class identifier
    r"pde\.(\w+\.)*": "",
    r"typing\.": "",
}


def process_signature(
    app, what: str, name: str, obj, options, signature, return_annotation
):
    """Process signature by applying replacement rules"""

    def process(sig_obj):
        """process the signature object"""
        if sig_obj is not None:
            for key, value in REPLACEMENTS_REGEX.items():
                sig_obj = re.sub(key, value, sig_obj)
            for key, value in REPLACEMENTS:
                sig_obj = sig_obj.replace(key, value)
        return sig_obj

    signature = process(signature)
    return_annotation = process(return_annotation)

    return signature, return_annotation


def process_docstring(app, what: str, name: str, obj, options, lines):
    """Process docstring by applying replacement rules"""
    for i, line in enumerate(lines):
        for key, value in REPLACEMENTS:
            line = line.replace(key, value)
        lines[i] = line


def setup(app):
    """set up hooks for this sphinx plugin"""
    app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-process-docstring", process_docstring)
