"""Kernels with a shared interface: embed() -> gram().

FidelityKernel is an exact statevector simulation of a Qiskit-style ZZFeatureMap
(H, P(2x_i), then CX-P(2(pi-x_i)(pi-x_j))-CX per pair, repeated `reps` times).
Because every layer after H is diagonal in the computational basis, the whole
simulation is a few matrix products -- no circuit objects needed.
    K(x, z) = |<phi(x)|phi(z)>|^2
"""
import itertools
import numpy as np


class Kernel:
    name = "kernel"

    def embed(self, Z):
        return np.asarray(Z, dtype=float)

    def gram(self, EA, EB):
        raise NotImplementedError

    def __call__(self, A, B):
        return self.gram(self.embed(A), self.embed(B))


class LinearKernel(Kernel):
    name = "linear"

    def gram(self, EA, EB):
        return EA @ EB.T


class RBFKernel(Kernel):
    def __init__(self, gamma=1.0):
        self.gamma = float(gamma)
        self.name = f"rbf(gamma={self.gamma:g})"

    def gram(self, EA, EB):
        d2 = (EA ** 2).sum(1)[:, None] + (EB ** 2).sum(1)[None, :] - 2 * EA @ EB.T
        return np.exp(-self.gamma * np.maximum(d2, 0.0))


class FidelityKernel(Kernel):
    def __init__(self, n_qubits=4, reps=2, entanglement="full", chunk=4096):
        self.n, self.reps, self.entanglement, self.chunk = n_qubits, reps, entanglement, chunk
        self.name = f"fidelity(q={n_qubits},reps={reps},ent={entanglement})"
        dim = 2 ** n_qubits
        idx = np.arange(dim)
        # bit i of the basis index = state of qubit i
        self.bits = ((idx[:, None] >> np.arange(n_qubits)[None, :]) & 1).astype(float)
        self.pairs = self._pairs()
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H = np.array([[1.0]])
        for _ in range(n_qubits):
            H = np.kron(H, h)
        self.H = H

    def _pairs(self):
        n = self.n
        if self.entanglement == "full":
            return list(itertools.combinations(range(n), 2))
        if self.entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if self.entanglement == "none":
            return []
        raise ValueError(f"unknown entanglement {self.entanglement!r}")

    def _states(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n:
            raise ValueError(f"expected {self.n} features, got {X.shape[1]}")
        B = self.bits
        phase = 2.0 * X @ B.T                                   # P(2 x_i) on every qubit
        for i, j in self.pairs:                                  # CX - P(.) - CX == ZZ phase
            coef = 2.0 * (np.pi - X[:, i]) * (np.pi - X[:, j])
            parity = np.abs(B[:, i] - B[:, j])
            phase += coef[:, None] * parity[None, :]
        D = np.exp(1j * phase)
        psi = np.full(D.shape, 1.0 / np.sqrt(2 ** self.n), dtype=complex) * D   # H|0..0>, then diag
        for _ in range(1, self.reps):
            psi = (psi @ self.H.T) * D
        return psi

    def embed(self, Z):
        Z = np.asarray(Z, dtype=float)
        return np.vstack([self._states(Z[i:i + self.chunk]) for i in range(0, len(Z), self.chunk)])

    def gram(self, EA, EB):
        return np.abs(EA @ EB.conj().T) ** 2
