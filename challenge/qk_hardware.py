"""Optional IBM inference for the quantum-kernel SVM. No hardware training: the SVM is fitted locally, exactly.

For each chosen validation field x, the device estimates one kernel row K(x, sv) = P(00000000) of the
compute-uncompute circuit U(sv)^dagger U(x)|0> for every support vector sv, plus one self-pair circuit
(ideal K = 1) that measures the depolarizing shrink factor used for a simple global correction.
The locally fitted SVM then classifies the field from that hardware kernel row.

    python -m challenge.qk_hardware submit --backend ibm_kingston --fields 0 --shots 1000 --approve
    python -m challenge.qk_hardware analyze --job <JOB_ID>
Credentials come from the saved Qiskit account or QISKIT_IBM_TOKEN; nothing secret is written to outputs/.
"""
import argparse
import json

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler import generate_preset_pass_manager

from .data import ROOT, load_data
from .qkernel import QuantumKernelSVM, compute_uncompute, statevectors, fidelity

OUT = ROOT / 'outputs/qk/hardware'


def fitted_model():
    tables, _, manifest, _ = load_data()
    cfg = json.loads((ROOT / 'configs/final_quantum_kernel.json').read_text())
    F = manifest['features']
    model = QuantumKernelSVM(cfg['a'], cfg['gamma'], cfg['reps'], cfg['C']).fit(tables['train'][F].to_numpy(), tables['train'].target)
    return model, tables, F


def circuits_for(model, X_raw):
    """Per field: one circuit per support vector, then the self-pair calibration circuit."""
    A_sv = model.A_train[model.svm.support_]
    out = []
    for x in model.prep.transform(np.asarray(X_raw, float)):
        out += [compute_uncompute(x, sv, model.a, model.gamma, model.reps) for sv in A_sv]
        out.append(compute_uncompute(x, x, model.a, model.gamma, model.reps))
    return out


def full_row(model, K_sv):
    """Kernel values on the support vectors only; the other columns carry zero dual weight."""
    K = np.zeros((len(K_sv), len(model.S_train)))
    K[:, model.svm.support_] = K_sv
    return K


def classify_rows(model, K_sv):
    return model.svm.predict(full_row(model, K_sv))


def shots_study(model, validation, F, lam=0.69):
    """All 150 validation fields: sample each kernel entry as Binomial(shots, p)/shots, where p is the exact value
    (ideal sampler) or the depolarized value lam*K + (1-lam)/256 corrected afterwards (hardware-like, lam as measured)."""
    from sklearn.metrics import f1_score
    exact = model.kernel_rows(validation[F].to_numpy())[:, model.svm.support_]
    rng, out = np.random.default_rng(2026), []
    for shots in [1000, 4000, 10000, 100000]:
        for kind in ['ideal', 'hardware_like']:
            p = exact if kind == 'ideal' else lam * exact + (1 - lam) / 256
            f1 = []
            for _ in range(5):
                K = rng.binomial(shots, np.clip(p, 0, 1)) / shots
                if kind == 'hardware_like': K = (K - (1 - lam) / 256) / lam
                f1.append(f1_score(validation.target, classify_rows(model, K), average='macro'))
            out.append(dict(shots=shots, model=kind, macro_f1_mean=float(np.mean(f1)), macro_f1_sd=float(np.std(f1))))
            print(out[-1], flush=True)
    exact_f1 = f1_score(validation.target, classify_rows(model, exact), average='macro')
    (OUT / 'shots_study.json').write_text(json.dumps(dict(exact_macro_f1=exact_f1, lam=lam, repeats=5, rows=out), indent=2))


def summary():
    """Pool every analysed twirled job (the first, untwirled job had its calibration circuit cancelled)."""
    rs = [r for r in (json.loads(f.read_text()) for f in sorted(OUT.glob('report_*.json'))) if r.get('twirling')]
    ok = lambda key: int(sum(p == t for r in rs for p, t in zip(r[key], r['true_labels'])))
    out = dict(jobs=[dict(job_id=r['job_id'], backend=r['backend'], fields=r['fields'], shots=r['shots'], circuits=r['circuits'],
                          usage_seconds=r['usage_seconds']) for r in rs],
               fields=sum(len(r['fields']) for r in rs), device_seconds=sum(r['usage_seconds'] for r in rs),
               correct=dict(exact=ok('predicted_exact'), ideal_shots=ok('predicted_ideal_shots'),
                            hardware_raw=ok('predicted_hardware_raw'), hardware_corrected=ok('predicted_hardware_corrected')),
               agreement_hardware_vs_exact=int(sum(p == e for r in rs for p, e in zip(r['predicted_hardware_corrected'], r['predicted_exact']))))
    (OUT / 'hardware_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def zero_probability(counts, shots):
    return counts.get('0' * 8, 0) / shots


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('submit'); s.add_argument('--backend', required=True); s.add_argument('--fields', type=int, nargs='+', required=True)
    s.add_argument('--shots', type=int, default=1000); s.add_argument('--approve', action='store_true')
    s.add_argument('--twirl', action='store_true')
    a = sub.add_parser('analyze'); a.add_argument('--job', required=True)
    sub.add_parser('shots'); sub.add_parser('summary')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tables, F = fitted_model()
    validation = tables['validation']
    if args.command == 'shots':
        return shots_study(model, validation, F)
    if args.command == 'summary':
        return summary()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()

    if args.command == 'submit':
        if not args.approve: raise PermissionError('Pass --approve: this spends IBM device time from your account')
        if not 1 <= args.shots <= 4000: raise ValueError('Use 1-4000 shots')
        backend = service.backend(args.backend)
        circuits = circuits_for(model, validation[F].to_numpy()[args.fields])
        isa = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=42).run(circuits)
        sampler = SamplerV2(mode=backend)
        sampler.options.dynamical_decoupling.enable = True
        if args.twirl:  # gate + measurement twirling: turns biased errors into a closer-to-uniform shrink
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
        job = sampler.run(isa, shots=args.shots)
        meta = dict(job_id=job.job_id(), backend=args.backend, fields=args.fields, shots=args.shots,
                    sample_ids=validation.sample_id.iloc[args.fields].tolist(), twirling=args.twirl, circuits=len(isa),
                    support_vectors=int(len(model.svm.support_)),
                    isa_depth_max=max(c.depth() for c in isa),
                    isa_two_qubit_max=max(sum(len(i.qubits) == 2 for i in c.data) for c in isa))
        (OUT / f'job_{job.job_id()}.json').write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return

    meta = json.loads((OUT / f'job_{args.job}.json').read_text())
    job = service.job(args.job)
    result = job.result()
    fields, shots, n_sv = meta['fields'], meta['shots'], meta['support_vectors']
    hw = np.array([zero_probability(r.data.meas.get_counts(), shots) for r in result]).reshape(len(fields), n_sv + 1)
    # Ideal-simulator comparison: exact statevector values and a finite-shot ideal sampler with the same shots.
    circuits = circuits_for(model, validation[F].to_numpy()[fields])
    sim = StatevectorSampler(seed=2026).run(circuits, shots=shots).result()
    sim = np.array([zero_probability(r.data.meas.get_counts(), shots) for r in sim]).reshape(len(fields), n_sv + 1)
    exact = model.kernel_rows(validation[F].to_numpy()[fields])[:, model.svm.support_]
    # Global depolarizing model K_hw = lam*K + (1-lam)/256; lam from the self-pair circuit (ideal K = 1).
    lam = (hw[:, -1] - 1 / 256) / (1 - 1 / 256)
    corrected = (hw[:, :-1] - (1 - lam[:, None]) / 256) / lam[:, None]
    labels = validation.target.to_numpy()[fields]
    report = dict(**meta, true_labels=labels.tolist(),
                  predicted_exact=classify_rows(model, exact).tolist(),
                  predicted_ideal_shots=classify_rows(model, sim[:, :-1]).tolist(),
                  predicted_hardware_raw=classify_rows(model, hw[:, :-1]).tolist(),
                  predicted_hardware_corrected=classify_rows(model, corrected).tolist(),
                  self_pair_fidelity=hw[:, -1].tolist(), shrink_factor=lam.tolist(),
                  kernel_mae=dict(ideal_shots=float(np.abs(sim[:, :-1] - exact).mean()),
                                  hardware_raw=float(np.abs(hw[:, :-1] - exact).mean()),
                                  hardware_corrected=float(np.abs(corrected - exact).mean())),
                  kernel_corr=dict(hardware_raw=float(np.corrcoef(hw[:, :-1].ravel(), exact.ravel())[0, 1]),
                                   ideal_shots=float(np.corrcoef(sim[:, :-1].ravel(), exact.ravel())[0, 1])),
                  decision_exact=model.svm.decision_function(full_row(model, exact)).tolist(),
                  decision_hardware_corrected=model.svm.decision_function(full_row(model, corrected)).tolist(),
                  usage_seconds=(job.usage() if hasattr(job, 'usage') else None),
                  scope='Inference on a few fixed validation fields with a locally trained model; not an accuracy estimate.')
    np.savez(OUT / f'kernels_{args.job}.npz', exact=exact, ideal_shots=sim, hardware=hw, corrected=corrected)
    (OUT / f'report_{args.job}.json').write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith('decision')}, indent=2, default=float))


if __name__ == '__main__':
    main()
