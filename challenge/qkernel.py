"""8-qubit fidelity quantum kernel + SVM, its entanglement-free ablation and the matched RBF control.

All three share one train-fitted encoder: StandardScaler -> PCA(8) -> StandardScaler on the 13 features.
"""
from time import perf_counter
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ('alfalfa', 'maize', 'wheat')
QUBITS = 8


def encoder():
    return make_pipeline(StandardScaler(), PCA(QUBITS, svd_solver='full'), StandardScaler())


def feature_map(a, gamma, reps=1):
    """H . RZ(2a x_i) . RZZ(2 gamma x_i x_{i+1}) on a linear chain, repeated `reps` times. No trainable angles."""
    x = ParameterVector('x', QUBITS)
    qc = QuantumCircuit(QUBITS)
    for _ in range(reps):
        qc.h(range(QUBITS))
        for i in range(QUBITS):
            qc.rz(2 * a * x[i], i)
        if gamma:
            for i in range(QUBITS - 1):
                qc.rzz(2 * gamma * x[i] * x[i + 1], i, i + 1)
    return qc, x


def statevectors(X, a, gamma, reps=1):
    qc, x = feature_map(a, gamma, reps)
    return np.array([Statevector(qc.assign_parameters(dict(zip(x, row)))).data for row in np.asarray(X)])


def fidelity(S1, S2):
    """K(x, x') = |<psi(x)|psi(x')>|^2 from exact statevectors."""
    return np.abs(S1.conj() @ S2.T) ** 2


class QuantumKernelSVM:
    """Fit on the 13 raw features; predict(X) returns crop names. Deterministic: `seed` is accepted for the interface."""

    def __init__(self, a=0.2, gamma=0.02, reps=1, C=100.0):
        self.a, self.gamma, self.reps, self.C = a, gamma, reps, C

    def fit(self, X, y, seed=0):
        start = perf_counter()
        self.prep = encoder().fit(np.asarray(X, float))
        self.A_train = self.prep.transform(np.asarray(X, float))
        self.S_train = statevectors(self.A_train, self.a, self.gamma, self.reps)
        self.svm = SVC(kernel='precomputed', C=self.C).fit(fidelity(self.S_train, self.S_train), np.asarray(y))
        self.circuit_evaluations = len(self.S_train)
        self.fit_seconds = perf_counter() - start
        return self

    def kernel_rows(self, X):
        S = statevectors(self.prep.transform(np.asarray(X, float)), self.a, self.gamma, self.reps)
        self.circuit_evaluations += len(S)
        return fidelity(S, self.S_train)

    def predict(self, X):
        return self.svm.predict(self.kernel_rows(X))

    def resources(self):
        qc, _ = feature_map(self.a, self.gamma, self.reps)
        ops = qc.count_ops()
        return dict(qubits=QUBITS, logical_depth=qc.depth(), rzz_gates=int(ops.get('rzz', 0)),
                    trainable_circuit_parameters=0, svm_dual_coefficients=int(self.svm.dual_coef_.size),
                    svm_intercepts=len(self.svm.intercept_), support_vectors=int(self.svm.n_support_.sum()),
                    encoder_fitted_values=encoder_size(self.prep),
                    kernel_matrix_floats=len(self.S_train) ** 2, statevector_simulations=self.circuit_evaluations)


class MatchedRBF:
    """Classical control: the same encoder and SVM family; the quantum kernel replaced by an RBF kernel."""

    def __init__(self, gamma=0.05, C=10.0):
        self.gamma, self.C = gamma, C

    def fit(self, X, y, seed=0):
        start = perf_counter()
        self.prep = encoder().fit(np.asarray(X, float))
        self.svm = SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(self.prep.transform(np.asarray(X, float)), np.asarray(y))
        self.fit_seconds = perf_counter() - start
        return self

    def predict(self, X):
        return self.svm.predict(self.prep.transform(np.asarray(X, float)))

    def resources(self):
        return dict(qubits=0, svm_dual_coefficients=int(self.svm.dual_coef_.size), svm_intercepts=len(self.svm.intercept_),
                    support_vectors=int(self.svm.n_support_.sum()), encoder_fitted_values=encoder_size(self.prep),
                    kernel_matrix_floats=0, statevector_simulations=0)


def encoder_size(prep):
    s1, pca, s2 = prep.named_steps.values()
    return int(s1.mean_.size + s1.scale_.size + pca.components_.size + pca.mean_.size + s2.mean_.size + s2.scale_.size)


def compute_uncompute(x1, x2, a, gamma, reps=1):
    """U(x2)^dagger U(x1)|0>: P(all zeros) = K(x1, x2). Used for shot-based / hardware kernel estimates."""
    qc, x = feature_map(a, gamma, reps)
    circuit = qc.assign_parameters(dict(zip(x, x1)))
    circuit.barrier()  # stops the transpiler cancelling the halves (else the x1 == x2 calibration circuit vanishes)
    circuit.compose(qc.assign_parameters(dict(zip(x, x2))).inverse(), inplace=True)
    circuit.measure_all()
    return circuit
