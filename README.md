# Q-SHIELD — quantum-similarity-driven hybrid intrusion & emerging-threat detection

Pipeline: flows → leak-safe split → impute/log/scale/PCA(4) fit on train only → 4-qubit ZZ fidelity kernel →
precomputed-kernel SVC (QSVC) → calibrated novelty gate → known / benign / UNKNOWN → explanations + dashboard.

**Claims are deliberately narrow:** this is a system-level combination to be *tested*, not a proven quantum advantage
or a verified first-ever detector. Simulated quantum kernels are not expected to be faster than classical ones.

## Quick start
```bash
pip install -r requirements.txt
pytest -q                                        # kernel properties, Qiskit cross-check, leakage test
python scripts/run_experiments.py --seeds 3 --ablate          # synthetic demo data
python scripts/export_demo.py --withheld Mirai                # artifacts for the dashboard
streamlit run app.py
```
Real data (CICIoT2023 CSVs from https://www.unb.ca/cic/datasets/iotdataset-2023.html):
```bash
python scripts/run_experiments.py --data-dir /path/to/CICIoT2023 --withheld all --seeds 5 --ablate
python scripts/export_demo.py --data-dir /path/to/CICIoT2023 --withheld Mirai
```
Without `--data-dir` a **synthetic** dataset is used (clearly flagged in the UI/outputs). Never report it as evidence.

## Layout
| File | Role |
|---|---|
| `qshield/kernels.py` | Exact simulator of Qiskit `ZZFeatureMap` fidelity kernel (+ RBF, linear), cross-checked vs Qiskit in tests |
| `qshield/data.py` | CICIoT2023 loader, family mapping, split with withheld family never in train/val |
| `qshield/preprocess.py` | Train-only imputer / log / scaler / PCA / bandwidth scaling |
| `qshield/novelty.py` | Gate: `1 − mean top-k similarity` (or kernel one-class SVM); τ from benign validation only |
| `qshield/system.py` | `KernelSystem` (SVC + gate, any kernel), `RFSystem`, `HybridRouter` |
| `qshield/experiment.py` | Baselines + ablations, identical data for every method |
| `qshield/explain.py`, `viz.py` | Perturbation attribution, plots |
| `scripts/` | `run_experiments.py` (tables → `results/`), `export_demo.py` (→ `artifacts/`) |
| `app.py` | 5-screen Streamlit dashboard (replay, kernel, verdict, evidence, safety) |

## Hypotheses → where they are tested
- **H1** (quantum vs matched RBF): rows `QSVC only` vs `RBF-SVM` / `Linear SVM` (macro-F1, balanced acc, per-class recall columns in `raw.csv`).
- **H2** (novelty gate): `* (closed-set margin)` vs `* + novelty gate` — AUROC/AUPRC, unknown recall at the benign-FPR target. `RBF-SVM + novelty gate` isolates kernel vs gate engineering.
- **H3** (fallback): `Hybrid router` vs `QSVC` — macro-F1, `fast_path_frac`, per-flow latency.
- Ablations (`--ablate`): reps 1/3, linear/no entanglement, bandwidth ×½/×2, 3/5 qubits.
- Fair tuning: quantum bandwidth, RBF γ and C are each chosen on validation macro-F1 only.

## Known limitations (state these in the pitch)
- CICIoT2023 has no session/device IDs → correlated-flow leakage is possible; splits are stratified per class.
- Kernel cost is O(n²) in training size; defaults use ~100 flows/class.
- Latency is measured on a CPU simulator in a Python loop; the hybrid router showed no speed benefit in the synthetic smoke run, so report what you measure on real data rather than assuming H3.
- "UNKNOWN" = unlike the training data; benign distribution shift can trigger it (see `benign_fpr`, `known_attack_flagged`).
- Dashboard replay examples are illustrations, not statistical evidence. Alerts are advisory; nothing auto-blocks traffic.
- Check 2025–2026 literature before claiming novelty.
"# Q-Kernel-Classifier" 
