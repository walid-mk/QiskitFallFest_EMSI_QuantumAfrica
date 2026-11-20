"""Reliability under input perturbations (validation only; models fitted once on clean training rows).

Gaussian noise: add sigma * (training std of each feature) to every validation feature, 5 draws per level,
the same draws for every model. Feature drop: replace one feature at a time with its training mean
(a missing band imputed). NDVI is perturbed as a feature like the others and not recomputed.
    python -m challenge.qk_reliability        -> outputs/qk/reliability.json, outputs/qk/fig5_reliability.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from .classical import reference
from .data import load_data
from .qk_run import OUT
from .qkernel import QuantumKernelSVM, MatchedRBF

SIGMAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
DRAWS = 5


def main():
    tables, _, manifest, _ = load_data()
    F = manifest['features']
    train, validation = tables['train'], tables['validation']
    s = json.loads((OUT / 'summary.json').read_text())
    q, r = s['final_config'], s['matched_rbf_config']
    X, y, V, yv = train[F].to_numpy(), train.target.to_numpy(), validation[F].to_numpy(), validation.target.to_numpy()
    models = {'quantum_kernel': QuantumKernelSVM(**q).fit(X, y), 'matched_rbf': MatchedRBF(r['gamma'], r['C']).fit(X, y),
              'reference': reference().fit(X, y)}
    f1 = lambda m, Z: f1_score(yv, models[m].predict(Z), average='macro')
    clean = {m: f1(m, V) for m in models}
    assert abs(clean['quantum_kernel'] - next(x for x in s['runs'] if x['model'] == 'quantum_kernel')['metrics']['macro_f1']) < 1e-12

    rng, std = np.random.default_rng(2026), X.std(axis=0)
    noise = []
    for sigma in SIGMAS:
        draws = [V + sigma * std * rng.standard_normal(V.shape) for _ in range(DRAWS if sigma else 1)]
        scores = {m: [f1(m, Z) for Z in draws] for m in models}
        noise.append(dict(sigma=sigma, **{m: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for m, v in scores.items()}))
        print('noise', sigma, {m: round(np.mean(v), 3) for m, v in scores.items()}, flush=True)

    drop = []
    for j, name in enumerate(F):
        Z = V.copy(); Z[:, j] = X[:, j].mean()
        drop.append(dict(feature=name, **{m: float(f1(m, Z) - clean[m]) for m in models}))
    print('worst drops', sorted(drop, key=lambda d: d['quantum_kernel'])[:3], flush=True)

    report = dict(clean=clean, gaussian_noise=noise, feature_drop=drop, draws=DRAWS,
                  noise_scale='sigma x training standard deviation of each raw feature',
                  drop_rule='one feature replaced by its training mean')
    (OUT / 'reliability.json').write_text(json.dumps(report, indent=2))

    colors = {'quantum_kernel': '#6a3fb5', 'matched_rbf': '#d9822b', 'reference': '#7f8c99'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for m, c in colors.items():
        axes[0].errorbar(SIGMAS, [n[m]['mean'] for n in noise], yerr=[n[m]['sd'] for n in noise], marker='o', capsize=3, color=c, label=m)
    axes[0].set(xlabel='Gaussian noise (x training std per feature)', ylabel='Validation macro-F1', title=f'{DRAWS} noise draws per level, shared by all models')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    x = np.arange(len(F))
    for k, (m, c) in enumerate(colors.items()):
        axes[1].bar(x + (k - 1) * .27, [d[m] for d in drop], .27, color=c, label=m)
    axes[1].set(xticks=x, xticklabels=F, ylabel='Change in macro-F1', title='One feature replaced by its training mean')
    axes[1].axhline(0, color='k', lw=.8); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(OUT / 'fig5_reliability.png', dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
