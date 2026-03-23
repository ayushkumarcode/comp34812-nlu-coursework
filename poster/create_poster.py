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
# CONFIGURATION
# ============================================================

SLIDE_WIDTH = Inches(40)
SLIDE_HEIGHT = Inches(22.5)

TITLE_BG = RGBColor(0x1B, 0x3A, 0x5C)
HEADER_BG = RGBColor(0x2E, 0x86, 0xAB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x2D, 0x2D, 0x2D)

TITLE_SIZE = Pt(48)
HEADER_SIZE = Pt(28)
BODY_SIZE = Pt(16)
