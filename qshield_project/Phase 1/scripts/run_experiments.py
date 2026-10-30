"""Run the full baseline / ablation grid and write results/*.csv and results/summary.md.

  python scripts/run_experiments.py                       # synthetic demo data
  python scripts/run_experiments.py --data-dir /path/to/CICIoT2023 --withheld Mirai Recon --seeds 5 --ablate
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from qshield.config import Config
from qshield.data import BENIGN, load_dataset
from qshield.experiment import run_one

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir"); ap.add_argument("--withheld", nargs="+", default=["Mirai"])
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--ablate", action="store_true")
ap.add_argument("--n-train-per-class", type=int, default=80)
ap.add_argument("--out", default="results")
a = ap.parse_args()

cfg = Config(n_train_per_class=a.n_train_per_class)
df, synthetic = load_dataset(a.data_dir)
fams = sorted(set(df["family"]) - {BENIGN}) if a.withheld == ["all"] else a.withheld
rows = []
for w in fams:
    for s in range(a.seeds):
        t0 = time.time()
        cfg.seed = s
        rows += run_one(df, w, s, cfg, ablate=a.ablate)
        print(f"withheld={w} seed={s} done in {time.time()-t0:.1f}s", flush=True)

raw = pd.DataFrame(rows)
out = Path(a.out); out.mkdir(exist_ok=True)
raw.to_csv(out / "raw.csv", index=False)
cols = ["macro_f1", "balanced_acc", "auroc", "auprc", "unk_recall", "benign_fpr",
        "known_attack_flagged", "openset_f1", "kernel_s", "latency_ms", "peak_mem_mb", "fast_path_frac"]
cols = [c for c in cols if c in raw]
g = raw.groupby("method", sort=False)[cols]
summ = g.mean().round(3).astype(str) + " +/- " + g.std().fillna(0).round(3).astype(str)
summ.insert(0, "runs", g.size())
summ.to_csv(out / "summary.csv")
(out / "summary.md").write_text(
    ("> SYNTHETIC DATA - demo only, not evidence.\n\n" if synthetic else "") + summ.to_markdown())
print(summ.to_string())
