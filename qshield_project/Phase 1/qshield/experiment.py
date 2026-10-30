"""One experiment = one withheld family x one seed. Every method sees identical data."""
import numpy as np
import pandas as pd

from .config import Config
from .data import BENIGN, make_split
from .evaluate import (confusion_df, known_metrics, openset_f1, per_flow_latency_ms,
                       unknown_metrics)
from .kernels import FidelityKernel, LinearKernel, RBFKernel
from .preprocess import Preprocessor
from .system import HybridRouter, KernelSystem, RFSystem


def _row(method, split, seed, sysm, Zte, yte, Zun, g_key, tau_key, gated_pred_key="pred", extra=None):
    kt = sysm.predict_full(Zte)
    ut = sysm.predict_full(Zun)
    tau = kt[tau_key]
    row = {"method": method, "withheld": split.withheld, "seed": seed}
    row.update(known_metrics(yte, kt["pred"]))
    row.update(unknown_metrics(kt[g_key], ut[g_key], tau, yte))
    row["openset_f1"] = openset_f1(yte, kt["pred"], kt[g_key], ut["pred"], ut[g_key], tau)
    row.update(fit_s=sysm.fit_time, kernel_s=sysm.kernel_time, peak_mem_mb=sysm.peak_mem_mb,
               latency_ms=per_flow_latency_ms(sysm.predict_full, Zte, 40))
    row.update(extra or {})
    return row


def run_one(df, withheld, seed, cfg: Config, ablate=False, keep=False):
    split = make_split(df, withheld, seed, cfg)
    pre0 = Preprocessor(cfg.n_qubits, cfg.scale_max).fit(split.Xtr)      # train only
    Z = {k: pre0.transform(getattr(split, k)) for k in ("Xtr", "Xva", "Xte", "Xun")}
    ytr, yva, yte = split.ytr, split.yva, split.yte
    rows, objs = [], {}

    def fit_kernel(kernel, use_gate, name):
        s = KernelSystem(kernel, cfg, use_gate=use_gate, name=name)
        return s.fit(Z["Xtr"], ytr, Z["Xva"], yva)

    # --- classical baselines -------------------------------------------------
    lin = fit_kernel(LinearKernel(), False, "Linear SVM")
    rows.append(_row("Linear SVM (closed-set margin)", split, seed, lin, Z["Xte"], yte, Z["Xun"],
                     "g_closed", "tau_closed"))

    rbfs = [fit_kernel(RBFKernel(g), True, "RBF-SVM") for g in cfg.gamma_grid]
    rbf = max(rbfs, key=lambda s: s.val_f1)                               # selected on validation
    rows.append(_row("RBF-SVM (closed-set margin)", split, seed, rbf, Z["Xte"], yte, Z["Xun"],
                     "g_closed", "tau_closed", extra={"kernel": rbf.kernel.name}))
    rows.append(_row("RBF-SVM + novelty gate", split, seed, rbf, Z["Xte"], yte, Z["Xun"],
                     "g_gate", "tau_gate", extra={"kernel": rbf.kernel.name}))

    # --- quantum (feature bandwidth chosen on validation, like RBF gamma) -------
    best_q = None
    for sc in cfg.scale_grid:
        pq = Preprocessor(cfg.n_qubits, sc).fit(split.Xtr)
        Zq_ = {k: pq.transform(getattr(split, k)) for k in ("Xtr", "Xva", "Xte", "Xun")}
        cand = KernelSystem(FidelityKernel(cfg.n_qubits, cfg.reps, cfg.entanglement), cfg, True, "QSVC")
        cand.fit(Zq_["Xtr"], ytr, Zq_["Xva"], yva)
        if best_q is None or cand.val_f1 > best_q[0].val_f1:
            best_q = (cand, pq, Zq_, sc)
    q, pre, Zq, q_scale = best_q
    qx = {"kernel": q.kernel.name, "scale": q_scale}
    rows.append(_row("QSVC only (closed-set margin)", split, seed, q, Zq["Xte"], yte, Zq["Xun"],
                     "g_closed", "tau_closed", extra=qx))
    rows.append(_row("QSVC + novelty gate", split, seed, q, Zq["Xte"], yte, Zq["Xun"],
                     "g_gate", "tau_gate", extra=qx))

    # --- hybrid fallback -------------------------------------------------------
    rf = RFSystem(cfg).fit(Zq["Xtr"], ytr)
    router = HybridRouter(rf, q, cfg.router_tol).fit(Zq["Xva"], yva)
    p_h, frac_fast = router.predict(Zq["Xte"])
    p_q, p_r = q.predict_known(Zq["Xte"]), rf.predict(Zq["Xte"])
    rows.append({"method": "Random Forest (known classes only)", "withheld": withheld, "seed": seed,
                 **known_metrics(yte, p_r), "fit_s": rf.fit_time,
                 "latency_ms": per_flow_latency_ms(rf.predict, Zq["Xte"], 40)})
    rows.append({"method": "Hybrid router (RF fast path -> QSVC)", "withheld": withheld, "seed": seed,
                 **known_metrics(yte, p_h), "fast_path_frac": frac_fast, "router_thr": router.thr,
                 "latency_ms": per_flow_latency_ms(lambda z: router.predict(z), Zq["Xte"], 40)})

    # --- quantum ablations (qubits / depth / entanglement / bandwidth) ---------
    if ablate:
        b = q_scale
        variants = [(4, 1, "full", b), (4, 3, "full", b), (4, 2, "linear", b), (4, 2, "none", b),
                    (4, 2, "full", b / 2), (4, 2, "full", b * 2), (3, 2, "full", b), (5, 2, "full", b)]
        for nq, reps, ent, sc in variants:
            p2 = Preprocessor(nq, sc).fit(split.Xtr)
            Zv = {k: p2.transform(getattr(split, k)) for k in ("Xtr", "Xva", "Xte", "Xun")}
            s = KernelSystem(FidelityKernel(nq, reps, ent), cfg, True).fit(Zv["Xtr"], ytr, Zv["Xva"], yva)
            tag = f"QSVC+gate [q={nq},reps={reps},ent={ent},scale={sc:.2f}]"
            rows.append(_row(tag, split, seed, s, Zv["Xte"], yte, Zv["Xun"], "g_gate", "tau_gate",
                             extra={"ablation": True}))

    if not keep:
        return rows
    objs = dict(split=split, pre=pre, pre_classical=pre0, Z=Zq, Z_classical=Z, qsvc=q, rbf=rbf, rf=rf,
                router=router, linear=lin, q_scale=q_scale,
                confusion={"QSVC": confusion_df(yte, p_q),
                           "RBF-SVM": confusion_df(yte, rbf.predict_known(Z["Xte"]))})
    return rows, objs
