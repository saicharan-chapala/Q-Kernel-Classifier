import time
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)

from .data import BENIGN, UNKNOWN


def known_metrics(y, p):
    labels = sorted(set(y))
    rec = recall_score(y, p, labels=labels, average=None, zero_division=0)
    out = {"macro_f1": f1_score(y, p, average="macro"),
           "balanced_acc": balanced_accuracy_score(y, p)}
    out.update({f"recall_{l}": r for l, r in zip(labels, rec)})
    return out


def unknown_metrics(g_known, g_unk, tau, y_known):
    """Unknown family = positive class. tau comes from BENIGN VALIDATION (never from test)."""
    y = np.r_[np.zeros(len(g_known)), np.ones(len(g_unk))]
    g = np.r_[g_known, g_unk]
    benign = y_known == BENIGN
    return {"auroc": roc_auc_score(y, g),
            "auprc": average_precision_score(y, g),
            "unk_recall": float((g_unk > tau).mean()),
            "benign_fpr": float((g_known[benign] > tau).mean()),
            "known_attack_flagged": float((g_known[~benign] > tau).mean())}


def openset_f1(y_known, p_known, g_known, p_unk, g_unk, tau):
    """Macro-F1 over {known classes + UNKNOWN}: flagged-as-unknown known flows count as errors."""
    pk = np.where(g_known > tau, UNKNOWN, p_known)
    pu = np.where(g_unk > tau, UNKNOWN, p_unk)
    y = np.r_[y_known, np.full(len(g_unk), UNKNOWN)]
    return f1_score(y, np.r_[pk, pu], average="macro")


def per_flow_latency_ms(fn, Z, n=40):
    n = min(n, len(Z))
    for i in range(min(3, n)):
        fn(Z[i:i + 1])
    t0 = time.perf_counter()
    for i in range(n):
        fn(Z[i:i + 1])
    return (time.perf_counter() - t0) / n * 1e3


def confusion_df(y, p):
    import pandas as pd
    labels = sorted(set(y) | set(p))
    return pd.DataFrame(confusion_matrix(y, p, labels=labels), index=labels, columns=labels)
