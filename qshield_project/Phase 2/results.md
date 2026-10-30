# QFF2026 QML Track — Results Report

## 1. Project overview

This project implements a quantum-kernel classifier and compares it with
a classical RBF Support Vector Machine (SVM). It also evaluates the effect
of a simulated noise model and reports the official Task A and Task B
benchmarks.

## 2. Q2: Quantum-kernel classifier

### Method

- Dataset: 800 labeled training samples with 50 input features.
- Test set: 200 samples without available labels.
- Feature reduction: selected/reduced the input to 4 dimensions using
  training data only.
- Quantum circuit: 4-qubit ZZFeatureMap, reps=1, linear entanglement.
- Kernel: fidelity-based quantum kernel.
- Classical baseline: RBF-SVM, with C=10 and gamma='scale'.
- Predictions: saved in `predictions_q2.csv` and passed the official
  prediction-file validator.

### Validation results

| Model/condition | Balanced accuracy |
|---|---:|
| Classical RBF-SVM | 71.07% |
| Ideal quantum kernel | 52.68% |
| Noisy quantum kernel | 49.11% |

The quantum experiment used a small training subset of 20 samples and
a validation subset of 30 samples. The classical baseline used more
training data, so these results are not a perfectly matched comparison.
The measurements do not demonstrate a quantum advantage. The noisy
quantum result was lower than the ideal quantum result.

### Circuit and kernel cost

- Qubits: 4
- Circuit depth: 20
- Two-qubit gates: 10 CX gates
- Training subset: 20 samples
- Validation subset: 30 samples
- Training kernel matrix: 20 x 20
- Validation-to-training kernel matrix: 30 x 20
- Unique off-diagonal training pairs: 190
- Validation-to-training pairs: 600
- Total unique pair evaluations counted: 790
- At 1,024 shots per pair: 808,960 shots

The circuit depth and gate count refer to the measured circuit configuration
from the experiment. Pair counts exclude diagonal entries, which were set
to one, and exploit symmetry in the training kernel matrix.

### Noise analysis

The experiment compared ideal and noisy quantum-kernel evaluations using
the organizer's QFF2026-v2 noise model. The noisy model includes single-
and two-qubit depolarizing errors and readout bit-flip errors.

Validation balanced accuracy decreased from 52.68% in the ideal condition
to 49.11% in the noisy condition, a decrease of 3.57 percentage points.
This indicates that noise affected the downstream classifier in this run.
No exact kernel-matrix difference statistic is reported here because it
has not been recorded in this report's source measurements.

## 3. Task A: Bell-state benchmark

Noise model version: QFF2026-v2
Shots: 8,192

The Bell-state fidelity is calculated as:

F = (1 + <XX> - <YY> + <ZZ>) / 4

| Measurement | Ideal | Noisy |
|---|---:|---:|
| XX expectation | 1.000000 | 0.819092 |
| YY expectation | -1.000000 | -0.827881 |
| ZZ expectation | 1.000000 | 0.873535 |
| Fidelity | 1.000000 | 0.880127 |
| Fidelity standard error | 0.000000 | 0.002592 |

The ideal simulation produced fidelity 1.0. Under the organizer's noise
model, the estimated fidelity was 0.880127 with standard error 0.002592.
This demonstrates the effect of simulated noise on the Bell-state
measurement.

## 4. Task B: Processor D and QFT-3 benchmark

Noise model version: QFF2026-v2

| Metric | Official result |
|---|---:|
| QFT-3 depth before transpilation | 6 |
| QFT-3 depth after transpilation | 22 |
| Two-qubit gate count | 12 |
| SWAP overhead | 1 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity | 0.603585 |

The Processor D design passed validation. Its configured four-qubit
linear connectivity and native gate set are intended to reflect the
small quantum circuit used in the Q2 experiment.

The transpiled QFT-3 circuit had greater depth than the original logical
circuit, reflecting the effects of mapping the circuit to the processor's
gate and connectivity constraints. The noisy fidelity of 0.603585 was
below the ideal fidelity of 1.0, showing sensitivity to the simulated
noise assumptions.

## 5. Limitations and conclusion

The Q2 quantum-kernel classifier did not outperform the classical
RBF-SVM in the recorded validation experiment. The quantum experiment
used a small subset of training data, and the validation set was small.
Therefore, the results should be interpreted as an implementation and
noise-sensitivity study, not evidence of quantum advantage.

Task A and Task B demonstrate how the organizer's noise model affects
quantum benchmark fidelity. Reproducibility depends on preserving the
official benchmark scripts, noise model, configuration, package versions,
and notebook execution order.

The test predictions were validated for file format and sample IDs.
This validation does not establish test-set predictive accuracy because
test labels were not available.
