"""Dataset loading, family mapping and leakage-safe splitting.

CICIoT2023 (https://www.unb.ca/cic/datasets/iotdataset-2023.html) ships flow-level
CSVs with a `label` column but no capture/session/device identifier, so splits here
are stratified by class only and may contain correlated flows. State this in your report.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

BENIGN = "Benign"
UNKNOWN = "UNKNOWN"

_PREFIX_TO_FAMILY = [
    ("benign", BENIGN), ("ddos", "DDoS"), ("dos", "DoS"), ("mirai", "Mirai"),
    ("recon", "Recon"), ("vulnerabilityscan", "Recon"),
    ("mitm", "Spoofing"), ("dns_spoofing", "Spoofing"),
    ("dictionarybruteforce", "BruteForce"),
    ("browserhijacking", "Web"), ("commandinjection", "Web"), ("sqlinjection", "Web"),
    ("xss", "Web"), ("backdoor", "Web"), ("uploading", "Web"),
]


def family_of(label: str) -> str:
    low = str(label).strip().lower()
    for prefix, fam in _PREFIX_TO_FAMILY:
        if low.startswith(prefix):
            return fam
    return str(label).strip()


def load_ciciot2023(data_dir, rows_per_file=3000, max_per_family=8000, seed=0):
    files = sorted(Path(data_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files under {data_dir}")
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        d = pd.read_csv(f)
        lab = next((c for c in d.columns if c.lower() == "label"), None)
        if lab is None:
            continue
        if len(d) > rows_per_file:
            d = d.sample(rows_per_file, random_state=int(rng.integers(1 << 31)))
        d["family"] = d[lab].map(family_of)
        parts.append(d.drop(columns=[lab]))
    df = pd.concat(parts, ignore_index=True)
    df = pd.concat([g.sample(min(len(g), max_per_family), random_state=seed)
                    for _, g in df.groupby("family")], ignore_index=True)
    return df


SYN_FEATURES = ["Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "syn_count",
                "ack_count", "rst_count", "Tot sum", "Min", "Max", "AVG", "Std", "IAT",
                "Number", "Variance"]
SYN_FAMILIES = [BENIGN, "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]


def make_synthetic(n_per_family=1500, seed=0):
    """Heavy-tailed, overlapping synthetic IoT-like flows. DEMO/CI ONLY -- not evidence."""
    geo = np.random.default_rng(12345)               # fixed family geometry
    rng = np.random.default_rng(seed)
    d_lat, F = 5, len(SYN_FEATURES)
    A = geo.normal(size=(d_lat, F)) / np.sqrt(d_lat)
    offs = geo.normal(scale=1.0, size=F)
    centers = {f: geo.normal(scale=1.5, size=d_lat) for f in SYN_FAMILIES}
    spreads = {f: (1.4 if f == BENIGN else geo.uniform(0.6, 1.1)) for f in SYN_FAMILIES}
    frames = []
    for f in SYN_FAMILIES:
        z = centers[f] + rng.normal(scale=spreads[f], size=(n_per_family, d_lat))
        x = np.exp(0.8 * (z @ A + 0.5 * rng.normal(size=(n_per_family, F))) + offs)
        d = pd.DataFrame(x, columns=SYN_FEATURES)
        d["family"] = f
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_dataset(data_dir=None, seed=0):
    if data_dir:
        return load_ciciot2023(data_dir, seed=seed), False
    return make_synthetic(seed=seed), True


@dataclass
class Split:
    withheld: str
    feature_names: list
    Xtr: pd.DataFrame
    ytr: np.ndarray
    Xva: pd.DataFrame
    yva: np.ndarray
    Xte: pd.DataFrame
    yte: np.ndarray
    Xun: pd.DataFrame          # complete withheld family, used ONLY at test time


def make_split(df, withheld, seed, cfg):
    """The withheld family is never present in train/val. Per-class sampling, no overlap."""
    if withheld not in set(df["family"]):
        raise ValueError(f"family {withheld!r} not in data: {sorted(set(df['family']))}")
    rng = np.random.default_rng(seed)
    feats = [c for c in df.columns if c != "family" and pd.api.types.is_numeric_dtype(df[c])]
    parts = {"tr": [], "va": [], "te": []}
    for fam, g in df[df["family"] != withheld].groupby("family"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        n = len(g)
        ntr = min(cfg.n_train_per_class, int(0.6 * n))
        nva = min(cfg.n_val_per_class, int(0.2 * n))
        nte = min(cfg.n_test_per_class, n - ntr - nva)
        parts["tr"].append(g.iloc[:ntr])
        parts["va"].append(g.iloc[ntr:ntr + nva])
        parts["te"].append(g.iloc[ntr + nva:ntr + nva + nte])
    tr, va, te = (pd.concat(parts[k], ignore_index=True) for k in ("tr", "va", "te"))
    un = df[df["family"] == withheld]
    un = un.sample(min(len(un), cfg.n_unknown), random_state=int(rng.integers(1 << 31)))
    return Split(withheld, feats, tr[feats], tr["family"].to_numpy(), va[feats],
                 va["family"].to_numpy(), te[feats], te["family"].to_numpy(), un[feats])
