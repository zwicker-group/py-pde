#!/usr/bin/env python3
"""
This scripts measures the total time it takes to import the module. The total time
should ideally be below 1 second.
"""

import sys

sys.path.append("..")

from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

import pde  # @UnusedImport

profiler.stop()

print(profiler.output_text(unicode=True, color=True))
