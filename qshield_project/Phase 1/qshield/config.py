from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Config:
    # quantum encoder
    n_qubits: int = 4
    reps: int = 2
    entanglement: str = "full"          # full | linear | none
    scale_max: float = float(np.pi)     # feature range used by the classical baselines
    scale_grid: tuple = (0.4, 0.8, 1.6, 3.14)   # quantum feature bandwidth, tuned on validation

    # sampling budget (kernel cost is quadratic in training size)
    n_train_per_class: int = 100
    n_val_per_class: int = 150
    n_test_per_class: int = 200
    n_unknown: int = 600

    # novelty gate
    n_ref: int = 300                    # reference-set size R
    knn: int = 5                        # s(x) = mean of top-k similarities
    target_fpr: float = 0.05            # benign validation false-positive target
    gate_method: str = "similarity"     # similarity | ocsvm
    ocsvm_nu: float = 0.1

    # hyper-parameter grids (selected on validation only)
    C_grid: tuple = (0.5, 5.0, 50.0)
    gamma_grid: tuple = (0.1, 0.3, 1.0, 3.0)

    # fallback router
    router_tol: float = 0.01            # predeclared max val macro-F1 loss
    latency_flows: int = 40
    seed: int = 0

    def to_dict(self):
        return asdict(self)
