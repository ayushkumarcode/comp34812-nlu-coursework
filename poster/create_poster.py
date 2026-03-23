#!/usr/bin/env python3
"""
COMP34812 — Academic Poster Generator
Group 34

Creates an A1 academic poster using python-pptx.
Single slide, 16:9 aspect ratio.
Generates matplotlib visualizations and embeds them.

Usage: python poster/create_poster.py
"""

import os
import sys
from pathlib import Path

# Use poster venv if available
VENV = Path('/tmp/poster_env')
if VENV.exists():
    sys.path.insert(0, str(VENV / 'lib' / 'python3.14' / 'site-packages'))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import numpy as np

# ============================================================
