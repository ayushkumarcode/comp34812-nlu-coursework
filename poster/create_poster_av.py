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
TINY_FONT      = Pt(18)

# Layout constants
MARGIN = Emu(600000)          # ~0.24 inches
COL_GAP = Emu(500000)         # ~0.2 inches
SECTION_GAP = Emu(400000)     # gap between sections vertically

# ============================================================
# CHART GENERATION
# ============================================================

def generate_f1_chart():
    """Generate F1 comparison bar chart with professional styling."""
    models = ['SVM\nBaseline', 'LSTM\nBaseline', 'BERT\nBaseline',
              'Sol 1\n(Cat A)', 'Sol 2\n(Cat B)']
    scores = [0.5610, 0.6226, 0.7854, 0.7340, 0.7422]

    fig, ax = plt.subplots(figsize=(14, 9))

    # Colors: gray baselines, green/orange for our solutions
    colors = ['#95a5a6', '#95a5a6', '#95a5a6', '#27AE60', '#E67E22']
    edge_colors = ['#7f8c8d', '#7f8c8d', '#7f8c8d', '#1E8449', '#CA6F1E']

    bars = ax.bar(models, scores, color=colors, edgecolor=edge_colors,
                  linewidth=2, width=0.65, zorder=3)

    # Value labels on top
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=22, fontweight='bold', color='#2D2D2D')

    # Styling
    ax.set_ylabel('Macro F1 Score', fontsize=24, fontweight='bold',
                  labelpad=15, color='#2D2D2D')
    ax.set_title('Authorship Verification - Model Comparison',
                 fontsize=28, fontweight='bold', pad=20, color='#1B3A5C')
    ax.set_ylim(0, 0.92)
    ax.set_xlim(-0.6, 4.6)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', labelsize=18, colors='#2D2D2D')
    ax.tick_params(axis='x', length=0)

    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')

    # Baseline reference line
    ax.axhline(y=0.5610, color='#BDC3C7', linestyle=':', linewidth=1.5,
               alpha=0.7, zorder=1)
    ax.text(4.5, 0.565, 'SVM Baseline', fontsize=14, color='#95a5a6',
            ha='right', style='italic')

    # Legend
    baseline_patch = mpatches.Patch(color='#95a5a6', label='Baselines')
    sol1_patch = mpatches.Patch(color='#27AE60', label='Our Sol 1 (Cat A)')
    sol2_patch = mpatches.Patch(color='#E67E22', label='Our Sol 2 (Cat B)')
    ax.legend(handles=[baseline_patch, sol1_patch, sol2_patch],
              fontsize=18, loc='upper left', framealpha=0.9,
              edgecolor='#CCCCCC')

    # Statistical significance annotations
    ax.annotate('', xy=(3, 0.74), xytext=(0, 0.57),
                arrowprops=dict(arrowstyle='->', color='#27AE60',
                               lw=2, connectionstyle='arc3,rad=0.15'))
    ax.text(1.5, 0.68, '+0.173***', fontsize=16, color='#27AE60',
            fontweight='bold', ha='center', rotation=15)

    ax.annotate('', xy=(4, 0.75), xytext=(1, 0.63),
                arrowprops=dict(arrowstyle='->', color='#E67E22',
                               lw=2, connectionstyle='arc3,rad=0.15'))
    ax.text(2.5, 0.72, '+0.120***', fontsize=16, color='#E67E22',
            fontweight='bold', ha='center', rotation=12)

    plt.tight_layout()
    path = POSTER_DIR / 'f1_chart_av.png'
    plt.savefig(str(path), dpi=250, bbox_inches='tight',
