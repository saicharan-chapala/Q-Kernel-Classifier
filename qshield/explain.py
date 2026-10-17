"""Original-feature perturbation attribution.

For each raw feature j: replace it with a benign background median and measure how much a
score changes. score_fn maps RAW flows -> scalar (e.g. novelty score). Positive delta means
the feature pushes the score up (more novel / more attack-like).
"""
import numpy as np
import pandas as pd


def perturbation_attribution(score_fn, x_raw, background, feature_names, topk=6):
    x = np.asarray(x_raw, dtype=float).reshape(1, -1)
    base = float(score_fn(x)[0])
    X = np.repeat(x, len(feature_names), axis=0)
    for j in range(len(feature_names)):
        X[j, j] = background[j]
    deltas = base - np.asarray(score_fn(X), dtype=float)
    df = pd.DataFrame({"feature": feature_names, "value": x.ravel(),
                       "background": background, "delta_score": deltas})
    df["abs"] = df["delta_score"].abs()
    return base, df.sort_values("abs", ascending=False).head(topk).drop(columns="abs").reset_index(drop=True)
