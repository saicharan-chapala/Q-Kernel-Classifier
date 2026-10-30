"""Q-SHIELD dashboard.   streamlit run app.py   (after: python scripts/export_demo.py)"""
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
from qshield.data import BENIGN
from qshield.explain import perturbation_attribution
from qshield.viz import attribution_bar, draw_feature_map, kernel_heatmap, score_gauge

st.set_page_config(page_title="Q-SHIELD", layout="wide")
ART = Path("artifacts")
if not (ART / "qshield.joblib").exists():
    st.error("No artifacts found. Run `python scripts/export_demo.py` first."); st.stop()


@st.cache_resource
def load():
    return joblib.load(ART / "qshield.joblib"), pd.read_csv(ART / "replay.csv"), pd.read_csv(ART / "evidence.csv")


M, replay, evidence = load()
q, pre, feats, bg, cfg = M["qsvc"], M["pre"], M["features"], M["background"], M["cfg"]

st.title("Q-SHIELD")
st.caption("Quantum-similarity-driven hybrid intrusion & emerging-threat detection  |  "
           "ADVISORY ONLY - experimental model, never auto-block real traffic")
if M["synthetic"]:
    st.warning("Running on SYNTHETIC demo data. Numbers here are not evidence about CICIoT2023.")

# ---- sidebar: replay controls ------------------------------------------------
st.sidebar.header("Traffic replay")
scenario = st.sidebar.radio("Scenario", ["Normal traffic", "Known attack", f"Withheld family ({M['withheld']})"])
role = {"Normal traffic": "benign", "Known attack": "known attack"}.get(scenario, "withheld (unknown)")
pool = replay[replay["role"] == role].reset_index(drop=True)
if role == "known attack":
    fam = st.sidebar.selectbox("Attack family", sorted(pool["family"].unique()))
    pool = pool[pool["family"] == fam].reset_index(drop=True)
idx = st.sidebar.slider("Flow #", 0, len(pool) - 1, 0)
flow = pool.iloc[idx]
x_raw = flow[feats].to_numpy(dtype=float).reshape(1, -1)
Z = pre.transform(x_raw)
out = q.predict_full(Z)
g, tau, pred = float(out["g_gate"][0]), float(out["tau_gate"]), str(out["pred"][0])
verdict = "UNKNOWN ALERT" if g > tau else ("BENIGN" if pred == BENIGN else f"KNOWN ATTACK: {pred}")

t1, t2, t3, t4, t5 = st.tabs(["1 Traffic replay", "2 Quantum kernel", "3 Threat verdict", "4 Evidence", "5 Safety"])

with t1:
    st.subheader("Selected flow")
    st.dataframe(flow[feats].to_frame("value").T, width="stretch")
    st.write(f"Ground truth (replay label): **{flow['family']}**")
    st.caption("Replay examples illustrate behaviour; they are not independent statistical evidence.")

with t2:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(draw_feature_map(cfg["n_qubits"], cfg["reps"], cfg["entanglement"], x=Z[0]))
        st.caption(f"Selected flow encoded into {cfg['n_qubits']} qubits via PCA features "
                   f"(bandwidth scale {M['q_scale']:.2f}). Exact simulator; no hardware.")
    with c2:
        samp = replay.groupby("family", group_keys=False).head(8).sort_values("family")
        Zs = pre.transform(samp[feats].to_numpy(dtype=float))
        st.pyplot(kernel_heatmap(q.kernel(Zs, Zs), samp["family"].to_numpy()))

with t3:
    (c1, c2), color = st.columns(2), ("red" if g > tau else "green")
    with c1:
        st.markdown(f"### :{color}[{verdict}]")
        st.write(f"Closed-set class: `{pred}`  |  novelty g(x) = **{g:.3f}**  |  threshold tau = **{tau:.3f}**")
        st.pyplot(score_gauge(g, tau))
        st.caption(f"tau is the {1 - cfg['target_fpr']:.0%} quantile of g on benign validation flows. "
                   "Advisory alert - requires analyst review.")
    with c2:
        _, attr = perturbation_attribution(lambda X: q.novelty_score(pre.transform(X)), x_raw, bg, feats)
        st.pyplot(attribution_bar(attr, "Features pushing novelty score (vs benign median)"))
        st.dataframe(attr.round(4), width="stretch")

with t4:
    st.subheader("Classical vs quantum on the same held-out test set")
    cols = [c for c in ["method", "macro_f1", "balanced_acc", "auroc", "auprc", "unk_recall", "benign_fpr",
                        "openset_f1", "kernel_s", "latency_ms", "peak_mem_mb"] if c in evidence]
    st.dataframe(evidence[cols].round(3), width="stretch")
    st.bar_chart(evidence.set_index("method")[["macro_f1", "auroc"]].dropna(how="all"))
    st.bar_chart(evidence.set_index("method")["latency_ms"])
    for name in ("QSVC", "RBF-SVM"):
        p = ART / f"confusion_{name}.csv"
        if p.exists():
            st.write(f"Confusion matrix - {name}"); st.dataframe(pd.read_csv(p, index_col=0))
    st.info("Single seed/family. For uncertainty use results/summary.md from scripts/run_experiments.py. "
            "No quantum speed-up is claimed: simulated quantum kernels are not faster than classical ones.")

with t5:
    st.markdown("""
- Every alert is **advisory**. Nothing in this app blocks, quarantines or modifies network traffic.
- "Unknown" means *unlike the training data* - it can be a benign distribution shift, not necessarily a new exploit.
- A withheld attack family is a test of generalisation, not proof of detecting real-world zero-days.
- Quantum component runs on a local exact simulator; hardware is optional future work.
""")
