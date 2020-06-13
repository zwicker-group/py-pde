#!/usr/bin/env python3

import sys
sys.path.append('..')

from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

import pde

profiler.stop()

print(profiler.output_text(unicode=True, color=True))
