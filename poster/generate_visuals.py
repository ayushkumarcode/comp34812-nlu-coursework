#!/usr/bin/env python3
"""Generate matplotlib visualizations for the poster."""

import sys
from pathlib import Path

VENV = Path('/tmp/poster_env')
if VENV.exists():
    for p in VENV.glob('lib/*/site-packages'):
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
