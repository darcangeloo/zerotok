import sys
from pathlib import Path

# Permette di lanciare lo script anche direttamente: aggiungo la root al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.head import load_head, save_head
from scripts.train_head import make_batch, BATCH_SIZE, MAX_CHOICES

HEAD_PATH = "head/head.pt"
VAL_PATH = "head/cache_val.pt"
N_BINS = 15
GRID = torch.logspace(-0.30103, 2.30103, 60)  # da 0.5 a 200, passo geometrico


def val_logits(head, cache):
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i in range(0, len(cache), BATCH_SIZE):
            c, a, m, y = make_batch(cache[i:i + BATCH_SIZE])
            logits = head(c, a, m)
            # Porto tutti i batch a MAX_CHOICES colonne con -inf
            logits = F.pad(logits, (0, MAX_CHOICES - logits.shape[1]), value=float("-inf"))
            all_logits.append(logits)
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def nll(logits, y, t):
    return F.cross_entropy(logits / t, y).item()


def ece(logits, y, t):
    probs = torch.softmax(logits / t, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, N_BINS + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.float().mean() * (correct[in_bin].mean() - conf[in_bin].mean()).abs()
    return float(total)


def report(name, logits, y, t):
    conf = torch.softmax(logits / t, dim=-1).max(dim=-1).values.mean().item()
    acc = (logits.argmax(dim=-1) == y).float().mean().item()
    print(f"{name} (T={t:.3f}): acc {acc:.3f}, conf media {conf:.3f}, "
          f"NLL {nll(logits, y, t):.4f}, ECE {ece(logits, y, t):.4f}")


if __name__ == "__main__":
    head, _ = load_head(HEAD_PATH)
    cache = torch.load(VAL_PATH)
    logits, y = val_logits(head, cache)

    # Provo ogni T della griglia e tengo quello con la NLL piu bassa
    best = 0
    best_loss = float("inf")
    for i, t in enumerate(GRID):
        loss = nll(logits, y, t)
        if loss < best_loss:
            best, best_loss = i, loss
    t_opt = GRID[best].item()
    if best in (0, len(GRID) - 1):
        print(f"ATTENZIONE: T ottimale {t_opt:.3f} e all'estremo della griglia, allargala")

    report("prima ", logits, y, 1.0)
    report("dopo  ", logits, y, t_opt)

    save_head(head, HEAD_PATH, temperature=t_opt)
    print(f"salvato {HEAD_PATH} con temperature {t_opt:.3f}")
