"""AV Cat B iter3 — Char-CNN+BiLSTM+GRL with LR=2e-4, no contrastive.

Changes from v2 (0.7414):
- LR=2e-4 (was 5e-4) — even slower
- Disable contrastive loss entirely (only BCE + topic adversarial)
- patience=20 (was 15), max_epochs=120
"""
import sys, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import AVCharDataset, generate_topic_labels, VOCAB_SIZE
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics

def train_epoch(model, dl, opt, dev, bce_fn, topic_fn, t_weight=0.02):
    model.train()
    total, all_p, all_l = 0, [], []
    for b in dl:
        c1, c2, labels = b['char_ids_1'].to(dev), b['char_ids_2'].to(dev), b['label'].to(dev)
        opt.zero_grad()
        logits, tl, _ = model(c1, c2, return_embeddings=True)
        loss = bce_fn(logits.squeeze(-1), labels)
        if t_weight > 0 and 'topic' in b:
            loss = loss + t_weight * topic_fn(tl, b['topic'].to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
        all_p.extend((torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu().numpy())
        all_l.extend(labels.cpu().numpy())
    return total / len(dl), f1_score(all_l, all_p, average='macro', zero_division=0)

def evaluate(model, dl, dev):
    model.eval()
    all_p, all_pr, all_l = [], [], []
    with torch.no_grad():
        for b in dl:
            c1, c2 = b['char_ids_1'].to(dev), b['char_ids_2'].to(dev)
            logits, _ = model(c1, c2)
            pr = torch.sigmoid(logits.squeeze(-1))
            all_p.extend((pr > 0.5).long().cpu().numpy())
            all_pr.extend(pr.cpu().numpy())
            all_l.extend(b['label'].numpy())
    return np.array(all_p), np.array(all_pr), np.array(all_l)

def main():
    print("=" * 60)
    print("  AV Cat B v3 — LR=2e-4, no contrastive")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')
    all_texts = list(train_df['text_1']) + list(train_df['text_2'])
    topic_all = generate_topic_labels(all_texts, n_clusters=10)
    train_topic = topic_all[:len(train_df)]
    num_topics = int(topic_all.max()) + 1
    train_ds = AVCharDataset(train_df, max_len=1500, augment=True, topic_labels=train_topic)
    dev_ds = AVCharDataset(dev_df, max_len=1500, augment=False, topic_labels=None)
    dev_ds.labels = np.array(dev_labels, dtype=np.float32)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = AVCatBModel(vocab_size=VOCAB_SIZE, char_emb_dim=32, cnn_filters=128,
                         lstm_hidden=128, proj_dim=128, num_topics=num_topics,
                         grl_lambda=0.0).to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    topic_fn = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    best_f1, patience_cnt = 0.0, 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)
    for epoch in range(1, 121):
        if epoch <= 20: grl_lambda = 0.05 * epoch / 20
        else: grl_lambda = 0.05
        model.grl.lambda_val = grl_lambda
        t_weight = 0.02 if epoch >= 15 else 0.0
        loss, tf1 = train_epoch(model, train_dl, opt, device, bce_fn, topic_fn, t_weight)
        sched.step()
        dp, dpr, dt = evaluate(model, dev_dl, device)
        df1 = f1_score(dt, dp, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train F1: {tf1:.4f} | Dev F1: {df1:.4f}")
        if df1 > best_f1:
            best_f1, patience_cnt = df1, 0
            torch.save(model.state_dict(), save_dir / 'av_cat_b_v3_best.pt')
            np.save(save_dir / 'av_cat_b_v3_probs.npy', dpr)
            print(f"  -> New best (F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"\nBest F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_dir / 'av_cat_b_v3_best.pt', weights_only=True))
    dp, dpr, dt = evaluate(model, dev_dl, device)
    best_thresh, best_tf1 = 0.5, 0
    for t in np.arange(0.30, 0.80, 0.02):
        pt = (dpr >= t).astype(int)
        ft = f1_score(dt, pt, average='macro', zero_division=0)
        if ft > best_tf1: best_thresh, best_tf1 = t, ft
        print(f"  thresh={t:.2f}: F1={ft:.4f}")
    print(f"\nBest threshold: {best_thresh:.2f} -> F1={best_tf1:.4f}")
    fp = (dpr >= best_thresh).astype(int)
    m = compute_all_metrics(dt, fp)
    print_metrics(m, "AV Cat B v3 — Final")
    save_predictions(fp, PROJECT_ROOT / 'predictions' / 'av_Group_34_B_v3.csv')
    print("Done!")

if __name__ == '__main__': main()
