#!/usr/bin/env python3
"""Create FileStorage files for backward compatibility tests.

The files are written to `tests/resources/FileStorage/current`. See the README in
`tests/resources/FileStorage` for how to add them to the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))
sys.path.insert(0, str(PACKAGE_PATH / "tests" / "storage"))

from _compat_cases import RESOURCES_ROOT, write_cases


def main():
    """Create all FileStorage resource files."""
    path = RESOURCES_ROOT / "FileStorage" / "current"
    for file in write_cases("FileStorage", path):
        print(f"Created {file}")
    print(
        f"\nTo include these files in the tests, copy them to a version folder, e.g., "
        f"{path.parent / 'v1.0'}"
    )


if __name__ == "__main__":
    main()
