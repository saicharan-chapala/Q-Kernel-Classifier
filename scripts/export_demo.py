"""Fit Q-SHIELD for one withheld family and export everything the dashboard needs.

  python scripts/export_demo.py --withheld Mirai [--data-dir /path/to/CICIoT2023]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd
from qshield.config import Config
from qshield.data import BENIGN, load_dataset
from qshield.experiment import run_one

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir"); ap.add_argument("--withheld", default="Mirai")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="artifacts")
a = ap.parse_args()

cfg = Config(seed=a.seed)
df, synthetic = load_dataset(a.data_dir)
rows, o = run_one(df, a.withheld, a.seed, cfg, ablate=False, keep=True)
out = Path(a.out); out.mkdir(exist_ok=True)

split = o["split"]
background = split.Xtr[split.ytr == BENIGN].median().to_numpy()       # benign median for attribution
te = split.Xte.copy(); te["family"] = split.yte
te["role"] = np.where(te["family"] == BENIGN, "benign", "known attack")
un = split.Xun.copy(); un["family"] = split.withheld; un["role"] = "withheld (unknown)"
replay = pd.concat([te.groupby("family", group_keys=False).head(60), un.head(60)], ignore_index=True)
replay.to_csv(out / "replay.csv", index=False)

joblib.dump(dict(pre=o["pre"], qsvc=o["qsvc"], rbf=o["rbf"], rf=o["rf"], router=o["router"],
                 pre_classical=o["pre_classical"], features=split.feature_names, background=background,
                 withheld=split.withheld, synthetic=synthetic, cfg=cfg.to_dict(),
                 q_scale=o["q_scale"]), out / "qshield.joblib")
pd.DataFrame(rows).to_csv(out / "evidence.csv", index=False)
for k, v in o["confusion"].items():
    v.to_csv(out / f"confusion_{k}.csv")
print("exported to", out.resolve())
print(pd.DataFrame(rows)[["method", "macro_f1", "auroc", "unk_recall", "benign_fpr", "latency_ms"]].round(3).to_string())
