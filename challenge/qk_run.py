"""Full quantum-kernel study: tuning, seeds 0/1/2, controls, ablations, learning curve, test predictions.

Run from the repository root:  python -m challenge.qk_run      (writes outputs/qk/ and configs/final_quantum_kernel.json)
"""
import json
import platform
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC

from .classical import reference
from .coverage import metric_vector, training_rows
from .data import ROOT, load_data
from .metrics import score
from .qkernel import LABELS, QuantumKernelSVM, MatchedRBF, encoder, statevectors, fidelity

OUT = ROOT / 'outputs/qk'
A_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]          # single-qubit angle scale (bandwidth)
G_GRID = [0.01, 0.02, 0.05, 0.1]             # entangling RZZ strength
C_GRID = [1, 10, 100, 1000]
RBF_GRID = np.geomspace(1e-3, 1, 20).round(5).tolist()  # 20 x 4 = 80 configs, same as the entangled grid


def pick(rows):
    """One selection rule for every model: best validation macro-F1, then accuracy, then the smallest C."""
    return sorted(rows, key=lambda r: (-r['f1'], -r['acc'], r['C']))[0]


def svm_scores(K, Kv, y, yv):
    out = []
    for C in C_GRID:
        p = SVC(kernel='precomputed', C=C).fit(K, y).predict(Kv)
        out.append(dict(C=C, f1=f1_score(yv, p, average='macro'), acc=accuracy_score(yv, p)))
    return out


def tune(A, B, y, yv, reps=1, gammas=G_GRID):
    rows = []
    for a, g in product(A_GRID, gammas):
        Sa, Sb = statevectors(A, a, g, reps), statevectors(B, a, g, reps)
        rows += [dict(a=a, gamma=g, reps=reps, **r) for r in svm_scores(fidelity(Sa, Sa), fidelity(Sb, Sa), y, yv)]
    return rows


def tune_rbf(A, B, y, yv):
    rows = []
    for g, C in product(RBF_GRID, C_GRID):
        p = SVC(gamma=g, C=C).fit(A, y).predict(B)
        rows.append(dict(gamma=g, C=C, f1=f1_score(yv, p, average='macro'), acc=accuracy_score(yv, p)))
    return rows


def bootstrap(y, pa, pb, groups, samples=2000):
    """Paired spatial-block bootstrap of metric(b) - metric(a): 95% two-sided interval for macro-F1 and accuracy."""
    unique, index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(2026)
    diffs = []
    while len(diffs) < samples:
        w = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[index]
        if all(w[y == k].sum() > 0 for k in range(3)):
            diffs.append(metric_vector(y, pb[None], w)[:2] - metric_vector(y, pa[None], w)[:2])
    d = np.array(diffs)
    point = metric_vector(y, pb[None], np.ones(len(y)))[:2] - metric_vector(y, pa[None], np.ones(len(y)))[:2]
    return dict(macro_f1=float(point[0]), macro_f1_ci=np.quantile(d[:, 0], [.025, .975]).tolist(),
                accuracy=float(point[1]), accuracy_ci=np.quantile(d[:, 1], [.025, .975]).tolist(),
                fields_differing=int((pa != pb).sum()))


def run_model(name, make, Xt, yt, Xv, Xs, records, preds):
    for seed in [0, 1, 2]:
        model = make()
        start = perf_counter(); model.fit(Xt, yt, seed) if name != 'reference' else model.fit(Xt, yt); fit = perf_counter() - start
        start = perf_counter(); pv = model.predict(Xv); predict = perf_counter() - start
        records.append(dict(model=name, seed=seed, metrics=score(validation.target, pv, LABELS), fit_seconds=fit,
                            predict_150_seconds=predict,
                            resources=model.resources() if hasattr(model, 'resources') else
                            dict(support_vectors=int(model[-1].n_support_.sum()), svm_dual_coefficients=int(model[-1].dual_coef_.size))))
        preds.setdefault(name, []).append(pv)
    assert all((p == preds[name][0]).all() for p in preds[name]), f'{name} is expected to be deterministic'
    return model, model.predict(Xs)


def main():
    global validation
    OUT.mkdir(parents=True, exist_ok=True)
    tables, _, manifest, _ = load_data()
    F = manifest['features']
    train, validation, test = tables['train'], tables['validation'], tables['test']
    y, yv = train.target.to_numpy(), validation.target.to_numpy()
    prep = encoder().fit(train[F].to_numpy())
    A, B = prep.transform(train[F].to_numpy()), prep.transform(validation[F].to_numpy())
    t0 = perf_counter()

    # 1. Equal tuning on validation: 80 entangled-kernel configs vs 80 RBF configs.
    start = perf_counter(); grid_q = tune(A, B, y, yv); tq = perf_counter() - start
    start = perf_counter(); grid_r = tune_rbf(A, B, y, yv); tr = perf_counter() - start
    best_q, best_r = pick(grid_q), pick(grid_r)
    print('quantum', best_q, '\nrbf', best_r, flush=True)

    # 2. Ablations, each tuned with the same rule on its own grid: entanglement removed, depth 2 and 3.
    grid_noent = tune(A, B, y, yv, gammas=[0])
    grid_deep = {reps: tune(A, B, y, yv, reps=reps) for reps in [2, 3]}
    best_noent = pick(grid_noent)
    ablation = {'entangled_reps1': dict(best=best_q, grid_mean_f1=np.mean([r['f1'] for r in grid_q]), configs=len(grid_q)),
                'no_entanglement': dict(best=best_noent, grid_mean_f1=np.mean([r['f1'] for r in grid_noent]), configs=len(grid_noent)),
                **{f'entangled_reps{k}': dict(best=pick(v), grid_mean_f1=np.mean([r['f1'] for r in v]), configs=len(v)) for k, v in grid_deep.items()},
                'matched_rbf': dict(best=best_r, grid_mean_f1=np.mean([r['f1'] for r in grid_r]), configs=len(grid_r))}
    # Paired view: for each (a, C), entangled (best gamma) vs no entanglement on the identical setting.
    print('ablation', json.dumps(ablation, default=float), flush=True)

    # 3. Final models, seeds 0/1/2, 914 training rows, plus test predictions.
    q = dict(a=best_q['a'], gamma=best_q['gamma'], reps=1, C=best_q['C'])
    Xt, Xv, Xs = train[F].to_numpy(), validation[F].to_numpy(), test[F].to_numpy()
    records, preds, test_preds, models = [], {}, {}, {}
    for name, make in [('quantum_kernel', lambda: QuantumKernelSVM(**q)),
                       ('no_entanglement', lambda: QuantumKernelSVM(best_noent['a'], 0, 1, best_noent['C'])),
                       ('matched_rbf', lambda: MatchedRBF(best_r['gamma'], best_r['C'])),
                       ('reference', reference)]:
        models[name], test_preds[name] = run_model(name, make, Xt, y, Xv, Xs, records, preds)
        print(name, records[-1]['metrics']['macro_f1'], round(records[-1]['fit_seconds'], 2), flush=True)
    supplied = json.loads((ROOT / 'results/runs.json').read_text())
    ref_supplied = [r['metrics']['macro_f1'] for r in supplied if r['model'] == 'reference']
    assert np.allclose(ref_supplied, [r['metrics']['macro_f1'] for r in records if r['model'] == 'reference']), 'reference mismatch'

    groups = json.loads((ROOT / 'data/validation_groups.json').read_text())
    g = np.array([groups[s] for s in validation.sample_id])
    enc = {c: i for i, c in enumerate(LABELS)}
    yi = np.array([enc[v] for v in yv]); P = {k: np.array([enc[v] for v in p[0]]) for k, p in preds.items()}
    comparisons = {'quantum_kernel - reference': bootstrap(yi, P['reference'], P['quantum_kernel'], g),
                   'matched_rbf - reference': bootstrap(yi, P['reference'], P['matched_rbf'], g),
                   'quantum_kernel - matched_rbf': bootstrap(yi, P['matched_rbf'], P['quantum_kernel'], g),
                   'no_entanglement - quantum_kernel': bootstrap(yi, P['quantum_kernel'], P['no_entanglement'], g)}

    # 4. Learning curve: re-fit the encoder on each subset and re-tune both models with the same rule.
    curve = []
    for size in [100, 200, 600, 914]:
        t_rows, v_rows, _ = training_rows(size)
        p2 = encoder().fit(t_rows[F].to_numpy()); A2, B2 = p2.transform(t_rows[F].to_numpy()), p2.transform(v_rows[F].to_numpy())
        bq, br = pick(tune(A2, B2, t_rows.target, yv)), pick(tune_rbf(A2, B2, t_rows.target, yv))
        curve.append(dict(size=size, quantum_kernel=bq['f1'], matched_rbf=br['f1'],
                          reference=f1_score(yv, reference().fit(t_rows[F], t_rows.target).predict(validation[F]), average='macro')))
        print('curve', curve[-1], flush=True)

    # 5. Kernel-structure check: without entanglement the kernel is the closed form prod cos^2(a dx).
    Sa = statevectors(A[:50], best_noent['a'], 0)
    closed = np.prod(np.cos(best_noent['a'] * (A[:50, None, :] - A[None, :50, :])) ** 2, axis=2)
    closed_form_error = float(np.abs(fidelity(Sa, Sa) - closed).max())

    pd.DataFrame({'sample_id': test.sample_id, 'prediction': test_preds['quantum_kernel']}).to_csv(OUT / 'test_predictions.csv', index=False)
    pd.DataFrame({'sample_id': test.sample_id, **test_preds}).to_csv(OUT / 'test_predictions_all_models.csv', index=False)
    pd.DataFrame({'sample_id': validation.sample_id, 'target': yv, **{k: v[0] for k, v in preds.items()}}).to_csv(OUT / 'validation_predictions.csv', index=False)
    pd.DataFrame(grid_q).to_csv(OUT / 'grid_quantum.csv', index=False); pd.DataFrame(grid_r).to_csv(OUT / 'grid_rbf.csv', index=False)
    summary = dict(final_config=q, matched_rbf_config=dict(gamma=best_r['gamma'], C=best_r['C']),
                   no_entanglement_config=dict(a=best_noent['a'], C=best_noent['C']),
                   tuning=dict(quantum_configs=len(grid_q), rbf_configs=len(grid_r), quantum_seconds=tq, rbf_seconds=tr,
                               rule='max validation macro-F1, then accuracy, then smallest C'),
                   runs=records, comparisons=comparisons, ablation=ablation, learning_curve=curve,
                   closed_form_error=closed_form_error, test_class_counts=pd.Series(test_preds['quantum_kernel']).value_counts().to_dict(),
                   environment=dict(python=platform.python_version(), machine=platform.processor() or platform.machine()),
                   total_seconds=perf_counter() - t0)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    (ROOT / 'configs/final_quantum_kernel.json').write_text(json.dumps(dict(model='QuantumKernelSVM', **q,
        encoder='StandardScaler -> PCA(8) -> StandardScaler, fit on training rows', qubits=8), indent=2))
    print('done in', round(perf_counter() - t0), 's')


if __name__ == '__main__':
    main()
