import numpy as np
from qshield.config import Config
from qshield.data import BENIGN, make_split, make_synthetic
from qshield.experiment import run_one


def test_no_leakage_of_withheld_family():
    df, cfg = make_synthetic(400), Config(n_train_per_class=50, n_val_per_class=40, n_test_per_class=60)
    s = make_split(df, "Mirai", 0, cfg)
    for y in (s.ytr, s.yva, s.yte):
        assert "Mirai" not in set(y)
    assert BENIGN in set(s.ytr) and len(s.Xun) > 0
    idx = [set(x.index) for x in (s.Xtr, s.Xva, s.Xte)]
    # split frames are re-indexed, so check row-content disjointness instead
    keys = [set(map(tuple, np.round(x.to_numpy(), 6))) for x in (s.Xtr, s.Xva, s.Xte)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_run_one_smoke():
    cfg = Config(n_train_per_class=40, n_val_per_class=40, n_test_per_class=50, n_unknown=100,
                 n_ref=100, scale_grid=(0.4, 1.6), gamma_grid=(0.3, 1.0), C_grid=(1.0, 10.0))
    rows = run_one(make_synthetic(300), "DoS", 0, cfg)
    names = {r["method"] for r in rows}
    assert "QSVC + novelty gate" in names and "RBF-SVM + novelty gate" in names
    assert all(0 <= r["macro_f1"] <= 1 for r in rows)
