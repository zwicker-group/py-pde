#!/usr/bin/env python3
"""
This script generates a list of reserved symbols in sympy, which should not be used in
expressions in py-pde.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
from pathlib import Path
from typing import List

import sympy
from sympy.parsing import sympy_parser

PACKAGE_PATH = Path(__file__).resolve().parents[1]


def get_reserved_sympy_symbols() -> List[str]:
    """return a list of reserved sympy symbols"""
    # test all public objects in sympy
    forbidden = []
    for name in dir(sympy):
        if name.startswith("_"):
            continue  # private names are not forbidden

        try:
            expr = sympy_parser.parse_expr(name)
        except Exception:
            continue  # skip symbols that cannot be parsed

        try:
            free_count = len(expr.free_symbols)
        except Exception:
            continue  # skip symbols that do not seem to be proper expressions

        if free_count == 0:
            # sympy seemed to have recognized this symbols
            forbidden.append(name.lower())

    return forbidden


def write_reserved_sympy_symbols(path: str = None):
    """write the list of sympy symbols to a json file"""
    if path is None:
        path = (
            PACKAGE_PATH / "pde" / "tools" / "resources" / "reserved_sympy_symbols.json"
        )

    with open(path, "w") as f:
        json.dump(get_reserved_sympy_symbols(), f)


if __name__ == "__main__":
    write_reserved_sympy_symbols()
