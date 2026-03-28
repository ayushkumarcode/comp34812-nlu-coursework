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
OUTPUT_PATH = POSTER_DIR / 'poster_av.pptx'

# Slide dimensions — 40" x 22.5" (16:9, A1-printable)
SLIDE_W = Inches(40)
SLIDE_H = Inches(22.5)

# Color palette
NAVY      = RGBColor(0x1B, 0x3A, 0x5C)   # Title background
TEAL      = RGBColor(0x2E, 0x86, 0xAB)   # Section headers
DARK_TEAL = RGBColor(0x1E, 0x6E, 0x8E)   # Accent
LIGHT_BG  = RGBColor(0xF5, 0xF7, 0xFA)   # Section body background
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
DARK      = RGBColor(0x2D, 0x2D, 0x2D)   # Body text
MID_GRAY  = RGBColor(0x66, 0x66, 0x66)   # Subtitle
LIGHT_GRAY = RGBColor(0xBB, 0xBB, 0xBB)
ACCENT_GREEN = RGBColor(0x27, 0xAE, 0x60)
ACCENT_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
TABLE_HEADER_BG = RGBColor(0x1B, 0x3A, 0x5C)
TABLE_ALT_BG = RGBColor(0xEB, 0xF0, 0xF5)
BORDER_COLOR = RGBColor(0xCC, 0xCC, 0xCC)

# Font sizes (calibrated for A1 print readability at 2m distance)
TITLE_FONT     = Pt(72)
SUBTITLE_FONT  = Pt(36)
HEADER_FONT    = Pt(36)
BODY_FONT      = Pt(24)
BODY_FONT_SM   = Pt(22)
SMALL_FONT     = Pt(20)
