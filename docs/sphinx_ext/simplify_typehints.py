"""
Simple sphinx plug-in that simplifies  type information in function signatures
"""

import re

# simple (literal) replacement rules
REPLACEMENTS = [
    # numbers and numerical arrays
    (
        "Union[int, float, complex, numpy.generic, numpy.ndarray, "
        "Sequence[Union[int, float, complex, numpy.generic, numpy.ndarray]], "
        "Sequence[Sequence[Any]]]",
        "ArrayLike",
    ),
    (
        "Union[int, float, complex, numpy.ndarray, "
        "Sequence[Union[int, float, complex, numpy.ndarray]], "
        "Sequence[Sequence[Any]]]",
        "ArrayLike",
    ),
    ("Union[int, float, complex, numpy.generic, numpy.ndarray]", "NumberOrArray"),
    ("Union[int, float, complex, numpy.ndarray]", "NumberOrArray"),
    ("Union[int, float, complex]", "Number"),
    # Remove some unnecessary information in favor of a more compact style
    ("Dict[KT, VT]", "dict"),
    ("Dict[str, Any]", "dict"),
    ("Optional[str]", "str"),
    ("Optional[int]", "int"),
    ("Optional[float]", "float"),
    ("Optional[numpy.ndarray]", "numpy.ndarray"),
    ("Optional[dict]", "dict"),
    ("Optional[Dict[str, Any]]", "dict"),
    # Complex types describing the boundary conditions
    ("Union[dict, str, BCBase]", "BoundaryData"),
    (
        "Union[Dict[str, BoundaryData], "
        "BoundaryData, Tuple[BoundaryData, BoundaryData]]",
        "BoundaryPairData",
    ),
    (
        "Union[Dict[str, BoundaryData], dict, str, BCBase, "
        "Tuple[BoundaryData, BoundaryData]]",
        "BoundaryPairData",
    ),
    ("Union[BoundaryPairData, Sequence[BoundaryPairData]]", "BoundariesData"),
    (
        "Union[Dict[str, BoundaryData], dict, str, BCBase, "
        "Tuple[BoundaryData, BoundaryData], Sequence[BoundaryPairData]]",
        "BoundariesData",
    ),
    (
        "Union[dict, str, BCBase, Tuple[Union[dict, str, BCBase], "
        "Union[dict, str, BCBase]], Sequence[Union[dict, str, BCBase, "
        "Tuple[Union[dict, str, BCBase], Union[dict, str, BCBase]]]]]",
        "BoundaryConditionData",
    ),
    (
        "Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, "
        "Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]], "
        "Sequence[Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, "
        "Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]]]]]",
        "BoundaryConditionData",
    ),
    (
        "Union[dict, str, BCBase, Tuple[Union[dict, str, BCBase], "
        "Union[dict, str, BCBase]], Sequence[Union[dict, str, BCBase, "
        "Tuple[Union[dict, str, BCBase], Union[dict, str, BCBase]]]], "
        "Sequence[BoundaryConditionData]]",
        "BoundaryConditionData",
    ),
    (
        "Union[Dict[str, BoundaryData], dict, str, BCBase, "
        "Tuple[BoundaryData, BoundaryData], "
        "Sequence[BoundaryPairData], Sequence[BoundariesData]]",
        "BoundariesDataList",
    ),
    (
        "Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, "
        "Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]], "
        "Sequence[Union[Dict[str, Union[Dict, str, BCBase]], Dict, str, BCBase, "
        "Tuple[Union[Dict, str, BCBase], Union[Dict, str, BCBase]]]], "
        "Sequence[BoundaryConditionData]]",
        "BoundariesDataList",
    ),
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
    "pde\.(\w+\.)*": "",
    "typing\.": "",
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


def setup(app):
    """set up hooks for this sphinx plugin"""
    app.connect("autodoc-process-signature", process_signature)
