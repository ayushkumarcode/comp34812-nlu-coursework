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


def add_section(slide, title, body, x, y, width, header_h=Emu(1500000)):
    """Add a section with colored header and body text."""
    header = slide.shapes.add_shape(1, x, y, width, header_h)
    header.fill.solid()
    header.fill.fore_color.rgb = HEADER_BG
    header.line.fill.background()

    txBox = slide.shapes.add_textbox(x + Emu(100000), y + Emu(150000),
                                      width - Emu(200000), header_h - Emu(300000))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = HEADER_SIZE
    p.font.color.rgb = WHITE
    p.font.bold = True

    body_y = y + header_h + Emu(100000)
    body_box = slide.shapes.add_textbox(x + Emu(100000), body_y,
                                         width - Emu(200000), Emu(15000000))
    bf = body_box.text_frame
    bf.word_wrap = True
    p = bf.paragraphs[0]
    p.text = body
    p.font.size = BODY_SIZE
    p.font.color.rgb = DARK


def create_poster(task='av'):
    """Create academic poster as PPTX."""
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Title banner
    shape = slide.shapes.add_shape(1, 0, 0, SLIDE_WIDTH, Emu(4000000))
    shape.fill.solid()
    shape.fill.fore_color.rgb = TITLE_BG
    shape.line.fill.background()

    txBox = slide.shapes.add_textbox(Emu(500000), Emu(500000),
                                      SLIDE_WIDTH - Emu(1000000), Emu(2500000))
    tf = txBox.text_frame
    tf.word_wrap = True
    task_name = 'Authorship Verification' if task == 'av' else 'NLI'
    p = tf.paragraphs[0]
    p.text = f'{task_name}: A Multi-Strategy Approach'
    p.font.size = TITLE_SIZE
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = 'COMP34812 — Group 34'
    p2.font.size = Pt(22)
    p2.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    p2.alignment = PP_ALIGN.CENTER

    prs.save('poster/poster.pptx')
    print("Poster saved to poster/poster.pptx")


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'av'
    create_poster(task=task)
