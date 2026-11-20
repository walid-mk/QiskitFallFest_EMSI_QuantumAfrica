"""Submitted model: 8-qubit fidelity quantum kernel + SVM on the 13 raw features (IDs never used).

Settings are read from configs/final_quantum_kernel.json, written by `python -m challenge.qk_run`.
"""
import json
from .data import ROOT
from .qkernel import QuantumKernelSVM


class Submission(QuantumKernelSVM):
    def __init__(self):
        cfg = json.loads((ROOT / 'configs/final_quantum_kernel.json').read_text())
        super().__init__(a=cfg['a'], gamma=cfg['gamma'], reps=cfg['reps'], C=cfg['C'])
