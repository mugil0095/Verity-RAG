import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p > lo) & (p <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def best_threshold(p, y):
    ts = np.linspace(0.05, 0.95, 91)
    return float(max(ts, key=lambda t: precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2]))

def report(p, y, t):
    p, y = np.asarray(p), np.asarray(y)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)
    return {"auroc": round(float(roc_auc_score(y, p)), 4), "precision": round(float(pr), 4), "recall": round(float(rc), 4),
            "f1": round(float(f1), 4), "acc": round(float(((p >= t) == y).mean()), 4), "ece": round(ece(p, y), 4), "threshold": round(t, 2), "n": int(len(y))}
