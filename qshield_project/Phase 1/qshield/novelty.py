"""Novelty gate. Score g(x): higher = more unlike the training reference set.

  similarity : g = 1 - mean of the top-k kernel similarities to a reference set R
               (s(x) = max_r K(x, r) when k = 1)
  ocsvm      : g = -decision_function of a precomputed-kernel one-class SVM
Threshold tau = (1 - target_fpr) quantile of g on BENIGN VALIDATION flows only.
The withheld attack family is never used to set direction, kernel or threshold.
"""
import numpy as np
from sklearn.svm import OneClassSVM


class NoveltyGate:
    def __init__(self, method="similarity", k=5, nu=0.1, target_fpr=0.05):
        self.method, self.k, self.nu, self.target_fpr = method, k, nu, target_fpr
        self.tau = None

    def fit(self, K_ref_ref):
        if self.method == "ocsvm":
            self.oc = OneClassSVM(kernel="precomputed", nu=self.nu).fit(K_ref_ref)
        elif self.method != "similarity":
            raise ValueError(self.method)
        return self

    def score(self, K_x_ref):
        if self.method == "ocsvm":
            return -self.oc.decision_function(K_x_ref)
        k = min(self.k, K_x_ref.shape[1])
        top = np.partition(K_x_ref, -k, axis=1)[:, -k:]
        return 1.0 - top.mean(axis=1)

    def calibrate(self, g_benign_val):
        self.tau = float(np.quantile(g_benign_val, 1.0 - self.target_fpr))
        return self.tau
