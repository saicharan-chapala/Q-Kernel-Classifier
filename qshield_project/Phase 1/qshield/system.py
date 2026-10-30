"""Classifiers that share one interface so quantum / classical comparisons are matched."""
import time
import tracemalloc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from .data import BENIGN, UNKNOWN
from .novelty import NoveltyGate


class KernelSystem:
    """Precomputed-kernel SVC + (optional) kernel novelty gate. Works for any Kernel."""

    def __init__(self, kernel, cfg, use_gate=True, name=None):
        self.kernel, self.cfg, self.use_gate = kernel, cfg, use_gate
        self.name = name or kernel.name

    def fit(self, Ztr, ytr, Zva, yva):
        cfg = self.cfg
        tracemalloc.start()
        t0 = time.perf_counter()
        self.Etr = self.kernel.embed(Ztr)
        Ktr = self.kernel.gram(self.Etr, self.Etr)
        self.kernel_time = time.perf_counter() - t0
        Kva = self.kernel.gram(self.kernel.embed(Zva), self.Etr)

        best = None
        for C in cfg.C_grid:
            svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, ytr)
            f1 = f1_score(yva, svc.predict(Kva), average="macro")
            if best is None or f1 > best[0]:
                best = (f1, C, svc)
        self.val_f1, self.C, self.svc = best

        benign_val = yva == BENIGN
        self.tau_closed = float(np.quantile(self._closed(Kva)[benign_val], 1 - cfg.target_fpr))

        self.gate = None
        if self.use_gate:
            rng = np.random.default_rng(cfg.seed)
            n_ref = min(cfg.n_ref, len(Ztr))
            self.ref_idx = np.sort(rng.choice(len(Ztr), n_ref, replace=False))
            self.gate = NoveltyGate(cfg.gate_method, cfg.knn, cfg.ocsvm_nu, cfg.target_fpr)
            self.gate.fit(Ktr[np.ix_(self.ref_idx, self.ref_idx)])
            self.gate.calibrate(self.gate.score(Kva[benign_val][:, self.ref_idx]))
        self.fit_time = time.perf_counter() - t0
        self.peak_mem_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        return self

    def _closed(self, K):
        """Closed-set unknown score: negative best one-vs-rest margin."""
        return -np.max(self.svc.decision_function(K), axis=1)

    def predict_full(self, Z):
        K = self.kernel.gram(self.kernel.embed(Z), self.Etr)
        out = {"pred": self.svc.predict(K), "g_closed": self._closed(K), "tau_closed": self.tau_closed}
        if self.gate is not None:
            g = self.gate.score(K[:, self.ref_idx])
            out.update(g_gate=g, tau_gate=self.gate.tau)
            out["final"] = np.where(g > self.gate.tau, UNKNOWN, out["pred"])
        return out

    def predict_known(self, Z):
        return self.svc.predict(self.kernel.gram(self.kernel.embed(Z), self.Etr))

    def novelty_score(self, Z):
        return self.predict_full(Z)["g_gate"]


class RFSystem:
    def __init__(self, cfg, name="RandomForest"):
        self.cfg, self.name = cfg, name

    def fit(self, Ztr, ytr):
        t0 = time.perf_counter()
        self.rf = RandomForestClassifier(n_estimators=100, random_state=self.cfg.seed, n_jobs=1,
                                         class_weight="balanced").fit(Ztr, ytr)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict_proba(self, Z):
        return self.rf.predict_proba(Z)

    def predict(self, Z):
        return self.rf.predict(Z)


class HybridRouter:
    """Fast classical path first; escalate low-confidence flows to the quantum kernel path.

    The confidence threshold is the LOWEST value whose validation macro-F1 is within
    `tol` of the quantum-only system (tol is predeclared in Config), i.e. maximum fast-path use.
    """

    def __init__(self, fast: RFSystem, slow: KernelSystem, tol=0.01):
        self.fast, self.slow, self.tol = fast, slow, tol

    def _route(self, Z, thr):
        proba = self.fast.predict_proba(Z)
        conf = proba.max(1)
        pred = self.fast.rf.classes_[proba.argmax(1)].astype(object)
        slow_mask = conf < thr
        if slow_mask.any():
            pred[slow_mask] = self.slow.predict_known(Z[slow_mask])
        return pred.astype(str), 1.0 - slow_mask.mean()

    def fit(self, Zva, yva):
        f_slow = f1_score(yva, self.slow.predict_known(Zva), average="macro")
        self.thr = 1.01                                  # 1.01 == always escalate
        for thr in np.linspace(0.3, 0.99, 70):
            pred, _ = self._route(Zva, thr)
            if f1_score(yva, pred, average="macro") >= f_slow - self.tol:
                self.thr = float(thr)
                break
        return self

    def predict(self, Z):
        return self._route(Z, self.thr)
