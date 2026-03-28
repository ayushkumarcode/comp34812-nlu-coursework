#!/usr/bin/env python3
"""
COMP34812 — Professional Academic Poster Generator (AV Track)
Group 34

Creates an A1-printable academic poster using python-pptx.
Single slide, 16:9 aspect ratio at 40" x 22.5" for high-res A1 printing.

Generates matplotlib visualizations and embeds them inline.

Usage:
    source /tmp/poster_env/bin/activate
    python poster/create_poster_av.py
"""

import os
import sys
from pathlib import Path

# Use poster venv if available
VENV = Path('/tmp/poster_env')
if VENV.exists():
    for p in VENV.glob('lib/*/site-packages'):
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ============================================================
# CONFIGURATION
# ============================================================

POSTER_DIR = Path(__file__).parent
