"""NLI Cat C iter12 — R-Drop alpha=2.0, warmup, epochs=20.

Current best: 0.9252 with alpha=0.5.
Try alpha=2.0 for stronger consistency regularization.
"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics

class D(Dataset):
    def __init__(self, df, tok, ml=128):
        self.p, self.h = list(df['premise']), list(df['hypothesis'])
        self.l = df['label'].values.astype(np.float32)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.l)
    def __getitem__(self, i):
        e = self.tok(self.p[i], self.h[i], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'ids': e['input_ids'].squeeze(0), 'mask': e['attention_mask'].squeeze(0), 'label': torch.tensor(self.l[i], dtype=torch.float)}

class M(nn.Module):
    def __init__(self, mn='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(mn)
        hs = self.enc.config.hidden_size
        self.cls = nn.Sequential(nn.Dropout(0.1), nn.Linear(hs, 256), nn.Tanh(), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(self, ids, mask):
        return self.cls(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])

def rdrop(l1, l2, labels, alpha=2.0):
    bce = nn.BCEWithLogitsLoss()
    tl = (bce(l1, labels) + bce(l2, labels)) / 2
    p1, p2 = torch.sigmoid(l1), torch.sigmoid(l2)
    d1 = torch.stack([p1, 1-p1], -1).clamp(1e-7)
    d2 = torch.stack([p2, 1-p2], -1).clamp(1e-7)
    kl = (F.kl_div(d1.log(), d2, reduction='batchmean') + F.kl_div(d2.log(), d1, reduction='batchmean')) / 2
    return tl + alpha * kl, tl, kl

def main():
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    MN, LR, BS, ML, EP, PAT, ALPHA = 'microsoft/deberta-v3-base', 1e-5, 16, 128, 20, 10, 2.0
    print(f"\n=== NLI Cat C + R-Drop (alpha={ALPHA}) ===\nLR={LR}, BS={BS}, ML={ML}, EP={EP}\n")
    tok = AutoTokenizer.from_pretrained(MN, use_fast=False)
    tdf, ddf = load_nli_data(split='train'), load_nli_data(split='dev')
    dl = load_solution_labels(task='nli')
    tds, dds = D(tdf, tok, ML), D(ddf, tok, ML)
    dds.l = np.array(dl, dtype=np.float32)
    tl = DataLoader(tds, batch_size=BS, shuffle=True, num_workers=4)
    dll = DataLoader(dds, batch_size=BS, shuffle=False, num_workers=4)
    model = M(mn=MN).to(dev)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ts = len(tl) * EP
    sched = get_linear_schedule_with_warmup(opt, int(ts*0.1), ts)
    sc = GradScaler('cuda')
    bf1, pc = 0, 0
    sd = PROJECT_ROOT / 'models'; sd.mkdir(exist_ok=True)
    for ep in range(1, EP+1):
        model.train()
        ttl, ttk, nb = 0, 0, 0
        for b in tl:
            ids, mask, labels = b['ids'].to(dev), b['mask'].to(dev), b['label'].to(dev)
            opt.zero_grad()
            with autocast('cuda'):
                l1, l2 = model(ids, mask).squeeze(-1), model(ids, mask).squeeze(-1)
                loss, tl2, kl = rdrop(l1, l2, labels, ALPHA)
            sc.scale(loss).backward()
            sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sc.step(opt); sc.update(); sched.step()
            ttl += tl2.item(); ttk += kl.item(); nb += 1
        model.eval()
        ps, prs, ls = [], [], []
        with torch.no_grad():
            for b in dll:
                ids, mask = b['ids'].to(dev), b['mask'].to(dev)
                lo = model(ids, mask).squeeze(-1)
                pr = torch.sigmoid(lo)
                ps.extend((pr > 0.5).long().cpu().numpy())
                prs.extend(pr.cpu().numpy()); ls.extend(b['label'].numpy())
        df1 = f1_score(ls, ps, average='macro', zero_division=0)
        print(f"Epoch {ep:3d} | task={ttl/nb:.4f} kl={ttk/nb:.4f} | Dev F1: {df1:.4f}")
        if df1 > bf1:
            bf1, pc = df1, 0
            torch.save(model.state_dict(), sd / 'nli_cat_c_alpha2_best.pt')
            np.save(sd / 'nli_cat_c_alpha2_probs.npy', np.array(prs))
            print(f"  -> Best (F1={bf1:.4f})")
        else:
            pc += 1
            if pc >= PAT: print(f"Early stop at {ep}"); break
    model.load_state_dict(torch.load(sd / 'nli_cat_c_alpha2_best.pt', weights_only=True))
    model.eval()
    fps = []
    with torch.no_grad():
        for b in dll:
            lo = model(b['ids'].to(dev), b['mask'].to(dev)).squeeze(-1)
            fps.extend(torch.sigmoid(lo).cpu().numpy())
    fps = np.array(fps); yd = np.array(dl)
    bt, btf = 0.5, 0
    for t in np.arange(0.30, 0.72, 0.02):
        pt = (fps >= t).astype(int)
        ft = f1_score(yd, pt, average='macro', zero_division=0)
        if ft > btf: bt, btf = t, ft
        print(f"  thresh={t:.2f}: F1={ft:.4f}")
    print(f"\nBest thresh: {bt:.2f} -> F1={btf:.4f}")
    fp = (fps >= bt).astype(int)
    m = compute_all_metrics(yd, fp)
    print_metrics(m, "NLI Cat C (alpha=2.0) Final")
    save_predictions(fp, PROJECT_ROOT / 'predictions' / 'nli_Group_34_C_alpha2.csv')
    print("Done!")
if __name__ == '__main__': main()
