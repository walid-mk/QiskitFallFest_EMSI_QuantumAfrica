"""Robustness of the re-uploading QNN vs its matched MLP control (sprints.md section 5 protocol).

Models are trained once on clean training rows (seeds 0/1/2); every model then sees exactly the same degraded
validation inputs, and we report the macro-F1 drop  dF1 = F1_clean - F1_degraded.
  - Sensor noise: N(0, sigma) added to the 4 prepared inputs (standardized PCA units), sigma in SIGMAS,
    DRAWS draws per level shared by all models and seeds.
  - Band drop (cloud / lost channel): one raw satellite band replaced by its training mean before preprocessing.
The supplied reference RBF-SVM (13 raw features) gets the same band drops; for noise it gets the same draws
mapped back to raw feature space through the inverse preprocessing, so all models see one degraded field.

    python -m challenge.robustness --layers 1 2 3      -> outputs/robustness/robustness.json, fig_robustness.png
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classical import reference
from .coverage import LABELS
from .data import ROOT, load_data
from .models import QuantumClassifier, MatchedHead

OUT = ROOT / 'outputs/robustness'
SIGMAS = [0.05, 0.10, 0.20]
DRAWS = 5
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3], help='QNN re-uploading layers L; the MLP uses h = L')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--report', action='store_true', help='only merge outputs/robustness/runs/*.json into the summary and figure')
    args = parser.parse_args()
    RUNS = OUT / 'runs'; RUNS.mkdir(parents=True, exist_ok=True)
    if args.report:
        return report(json.loads((RUNS / 'reference.json').read_text())['F'])
    tables, _, manifest, _ = load_data()
    F = manifest['features']
    train, validation = tables['train'], tables['validation']
    X, y, V, yv = train[F].to_numpy(), train.target.to_numpy(), validation[F].to_numpy(), validation.target.to_numpy()
    prep = make_pipeline(StandardScaler(), PCA(4, svd_solver='full'), StandardScaler()).fit(X)
    A, B = prep.transform(X), prep.transform(V)

    rng = np.random.default_rng(2026)
    noisy = {s: [B + rng.normal(0, s, B.shape) for _ in range(DRAWS)] for s in SIGMAS}
    dropped = {}
    for j, band in enumerate(F):
        Z = V.copy(); Z[:, j] = X[:, j].mean(); dropped[band] = Z
    f1 = lambda pred: f1_score(yv, pred, labels=LABELS, average='macro')

    def evaluate(predict_prepared, predict_raw):
        clean = f1(predict_prepared(B))
        return dict(clean=clean,
                    noise={str(s): float(np.mean([clean - f1(predict_prepared(Z)) for Z in noisy[s]])) for s in SIGMAS},
                    band_drop={b: float(clean - f1(predict_raw(Z))) for b, Z in dropped.items()})

    if not (RUNS / 'reference.json').exists():
        ref = reference().fit(X, y)
        # Reference sees the same noise draws mapped to raw space: V + inverse(B + N) - inverse(B) (the map is linear).
        base = prep.inverse_transform(B)
        r = evaluate(lambda Z: ref.predict(V + prep.inverse_transform(Z) - base), ref.predict)
        (RUNS / 'reference.json').write_text(json.dumps(dict(name='reference', seed=0, F=F, **r), indent=2))
    for L in args.layers:
        for name, make in [(f'qnn_L{L}', lambda seed: QuantumClassifier(LABELS, encoding='ry', max_evals=100, layers=L).fit(A, y, seed)),
                           (f'mlp_h{L}', lambda seed: MatchedHead(LABELS, hidden_units=L, max_evals=100).fit(A, y, seed))]:
            for seed in args.seeds:
                m = make(seed)
                r = dict(name=name, seed=seed, trainable=len(m.weights), fit_seconds=m.fit_seconds,
                         **evaluate(m.predict, lambda Z, m=m: m.predict(prep.transform(Z))))
                (RUNS / f'{name}_s{seed}.json').write_text(json.dumps(r, indent=2))
                print(name, seed, round(r['clean'], 3), {k: round(v, 3) for k, v in r['noise'].items()}, flush=True)


def report(F):
    runs = [json.loads(f.read_text()) for f in sorted((OUT / 'runs').glob('*.json'))]
    order = ['reference'] + sorted({r['name'] for r in runs} - {'reference'}, key=lambda n: (n[-1], n[:3] != 'qnn'))
    results = {'protocol': __doc__.split(chr(10) * 2)[1].strip(), 'sigmas': SIGMAS, 'draws': DRAWS,
               'models': {n: sorted([r for r in runs if r['name'] == n], key=lambda r: r['seed']) for n in order}}
    summary = {}
    for name, runs in results['models'].items():
        mean = lambda key: (float(np.mean([r[key] for r in runs])), float(np.std([r[key] for r in runs])))
        summary[name] = dict(clean=mean('clean'),
                             **{f'dF1_noise_{s}': (float(np.mean([r['noise'][str(s)] for r in runs])), float(np.std([r['noise'][str(s)] for r in runs]))) for s in SIGMAS},
                             dF1_band_drop_mean=float(np.mean([np.mean(list(r['band_drop'].values())) for r in runs])),
                             worst_band=max(runs[0]['band_drop'], key=lambda b: np.mean([r['band_drop'][b] for r in runs])))
    results['summary'] = summary
    (OUT / 'robustness.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=1))

    # Figure: macro-F1 drop vs noise (mean +/- seed SD) and mean drop per band.
    colors = {'reference': '#7f8c99', 'qnn': '#6a3fb5', 'mlp': '#d9822b'}
    styles = {1: ':', 2: '--', 3: '-'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, s in summary.items():
        kind, L = (name, 2) if name == 'reference' else (name[:3], int(name[-1]))
        axes[0].errorbar([0] + SIGMAS, [0] + [s[f'dF1_noise_{x}'][0] for x in SIGMAS], yerr=[0] + [s[f'dF1_noise_{x}'][1] for x in SIGMAS],
                         color=colors[kind], ls=styles[L], marker='o', capsize=3, label=name)
    axes[0].set(xlabel='Gaussian noise sigma on the prepared inputs', ylabel='Macro-F1 drop (clean - noisy)', title='Higher = less robust · mean ± seed SD')
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=.3)
    names = list(summary); x = np.arange(len(F)); w = .8 / len(names)
    for k, name in enumerate(names):
        kind = name if name == 'reference' else name[:3]
        runs = results['models'][name]
        axes[1].bar(x + (k - len(names) / 2 + .5) * w, [np.mean([r['band_drop'][b] for r in runs]) for b in F], w, color=colors[kind],
                    alpha=.45 + .18 * (int(name[-1]) if name != 'reference' else 2), label=name)
    axes[1].set(xticks=x, xticklabels=F, ylabel='Macro-F1 drop', title='One satellite band missing (training-mean imputed)')
    axes[1].axhline(0, color='k', lw=.8); axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(OUT / 'fig_robustness.png', dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
