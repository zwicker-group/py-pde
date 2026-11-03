#!/usr/bin/env python3
"""Create MovieStorage files for backward compatibility tests.

The files are written to `tests/resources/MovieStorage/current`. See the README in
`tests/resources/MovieStorage` for how to add them to the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))
sys.path.insert(0, str(PACKAGE_PATH / "tests" / "storage"))

from _compat_cases import RESOURCES_ROOT, write_cases


def main():
    """Create all MovieStorage resource files."""
    path = RESOURCES_ROOT / "MovieStorage" / "current"
    for file in write_cases("MovieStorage", path):
        print(f"Created {file}")
    print(
        f"\nTo include these files in the tests, copy them (including `.times` files) "
        f"to a version folder, e.g., {path.parent / 'v1.0'}"
    )


if __name__ == "__main__":
    main()
