#!/usr/bin/env python3
"""This scripts measures the total time it takes to import the module.

The total time should ideally be below 1 second.
"""

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))


from pyinstrument import Profiler

with Profiler() as profiler:
    import pde

print(profiler.open_in_browser())
