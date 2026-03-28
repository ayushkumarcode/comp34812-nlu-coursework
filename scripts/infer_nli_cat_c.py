"""NLI Category C — Test inference with DeBERTa cross-encoder."""
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.models.cat_c_deberta import NLIDeBERTaCrossEncoder
