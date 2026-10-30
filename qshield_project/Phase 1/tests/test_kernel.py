import numpy as np
import pytest
from qshield.kernels import FidelityKernel


@pytest.mark.parametrize("ent", ["full", "linear", "none"])
def test_kernel_properties(ent):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(30, 4))
    K = FidelityKernel(4, 2, ent)(X, X)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)
    assert np.allclose(K, K.T, atol=1e-9)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-9
    assert np.linalg.eigvalsh(K).min() > -1e-8          # positive semi-definite


@pytest.mark.parametrize("ent,reps", [("full", 2), ("linear", 1), ("full", 1)])
def test_matches_qiskit(ent, reps):
    qk = pytest.importorskip("qiskit")
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, size=(6, 4))
    fm = ZZFeatureMap(4, reps=reps, entanglement=ent)
    sv = [Statevector(fm.assign_parameters(x)).data for x in X]
    ref = np.abs(np.array(sv) @ np.array(sv).conj().T) ** 2
    assert np.allclose(FidelityKernel(4, reps, ent)(X, X), ref, atol=1e-9)
