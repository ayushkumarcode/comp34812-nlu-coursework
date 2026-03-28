"""AV Category B — Test inference with Siamese char-CNN+BiLSTM+GRL."""
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.models.av_cat_b_dataset import char_encode
from src.models.av_cat_b_model import AVCatBModel


def main():
    print("=" * 60)
    print("  AV Cat B — Test Inference")
    print("=" * 60, flush=True)

    model_path = PROJECT_ROOT / 'models' / 'av_cat_b_v3_best.pt'
    test_path = PROJECT_ROOT / 'test_data_av.csv'
    out_path = PROJECT_ROOT / 'predictions' / 'av_test_B.csv'
    out_path.parent.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)

    # Load test data
    print("\n[1/4] Loading test data...", flush=True)
    t0 = time.time()
    df = pd.read_csv(test_path, quotechar='"', engine='python')
    df['text_1'] = df['text_1'].apply(lambda x: clean_text(x, lowercase=False))
    df['text_2'] = df['text_2'].apply(lambda x: clean_text(x, lowercase=False))
    print(f"  Loaded {len(df)} pairs in {time.time()-t0:.1f}s", flush=True)

    # Encode characters
    print("\n[2/4] Encoding characters...", flush=True)
    t0 = time.time()
    max_len = 1500
    encoded_1 = [char_encode(t, max_len) for t in df['text_1']]
    encoded_2 = [char_encode(t, max_len) for t in df['text_2']]
    ids_1 = torch.tensor(np.array(encoded_1), dtype=torch.long)
    ids_2 = torch.tensor(np.array(encoded_2), dtype=torch.long)
    print(f"  Encoded in {time.time()-t0:.1f}s", flush=True)

    # Load model
    print("\n[3/4] Loading model...", flush=True)
    model = AVCatBModel(
        vocab_size=97, char_emb_dim=32, cnn_filters=128,
        lstm_hidden=128, proj_dim=128, num_topics=10,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("  Model loaded.", flush=True)

    # Inference in batches
    print("\n[4/4] Running inference...", flush=True)
    t0 = time.time()
    batch_size = 64
    all_preds = []

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            b1 = ids_1[start:end].to(device)
            b2 = ids_2[start:end].to(device)
            logits, _ = model(b1, b2)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            if (start // batch_size) % 20 == 0:
                print(f"  Batch {start//batch_size}: {start}/{len(df)}", flush=True)

    all_preds = np.array(all_preds)
    print(f"  Inference time: {time.time()-t0:.1f}s", flush=True)
    print(f"  Predictions: {len(all_preds)}", flush=True)
    print(f"  Class distribution: 0={sum(all_preds==0)}, 1={sum(all_preds==1)}", flush=True)
