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


def main():
    print("=" * 60)
    print("  NLI Cat C — Test Inference")
    print("=" * 60, flush=True)

    model_path = PROJECT_ROOT / 'models' / 'nli_cat_c_best.pt'
    test_path = PROJECT_ROOT / 'test_data_nli.csv'
    out_path = PROJECT_ROOT / 'predictions' / 'nli_test_C.csv'
    out_path.parent.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)

    # Load test data
    print("\n[1/4] Loading test data...", flush=True)
    t0 = time.time()
    df = pd.read_csv(test_path, quotechar='"', engine='python')
    df['premise'] = df['premise'].apply(lambda x: clean_text(x, lowercase=False))
    df['hypothesis'] = df['hypothesis'].apply(lambda x: clean_text(x, lowercase=False))
    df['premise'] = df['premise'].apply(lambda x: '.' if not x or x.isspace() else x)
    print(f"  Loaded {len(df)} pairs in {time.time()-t0:.1f}s", flush=True)

    # Tokenize
    print("\n[2/4] Tokenizing...", flush=True)
    t0 = time.time()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'microsoft/deberta-v3-base', use_fast=False
    )
    max_len = 256

    all_input_ids = []
    all_attention_masks = []
    for _, row in df.iterrows():
        encoded = tokenizer(
            row['premise'], row['hypothesis'],
            max_length=max_len, truncation=True, padding='max_length',
            return_tensors='pt',
        )
        all_input_ids.append(encoded['input_ids'].squeeze(0))
        all_attention_masks.append(encoded['attention_mask'].squeeze(0))

    input_ids = torch.stack(all_input_ids)
    attention_masks = torch.stack(all_attention_masks)
    print(f"  Tokenized {len(df)} pairs in {time.time()-t0:.1f}s", flush=True)

    # Load model
    print("\n[3/4] Loading model...", flush=True)
    model = NLIDeBERTaCrossEncoder(
        model_name='microsoft/deberta-v3-base', grl_lambda=0.1
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("  Model loaded.", flush=True)

    # Inference in batches
    print("\n[4/4] Running inference...", flush=True)
    t0 = time.time()
    batch_size = 32
    all_probs = []

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            ids = input_ids[start:end].to(device)
            mask = attention_masks[start:end].to(device)
            logits, _ = model(ids, mask)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
            if (start // batch_size) % 20 == 0:
                print(f"  Batch {start//batch_size}: {start}/{len(df)}", flush=True)

    all_probs = np.array(all_probs)
    print(f"  Inference time: {time.time()-t0:.1f}s", flush=True)
