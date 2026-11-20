"""Figures for the quantum-kernel study, read from outputs/qk/summary.json.   python -m challenge.qk_figures"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .data import ROOT
from .qkernel import LABELS

OUT = ROOT / 'outputs/qk'
NAMES = {'quantum_kernel': 'Quantum kernel (8 qubits)', 'no_entanglement': 'Entanglement removed',
         'matched_rbf': 'Matched RBF (same 8 inputs)', 'reference': 'Supplied RBF (13 raw)'}
COLORS = {'quantum_kernel': '#6a3fb5', 'no_entanglement': '#b39ddb', 'matched_rbf': '#d9822b', 'reference': '#7f8c99'}


def main():
    s = json.loads((OUT / 'summary.json').read_text())
    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})

    # 1. Scores and paired differences.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for i, m in enumerate(NAMES):
        f1 = s['runs'][[r['model'] for r in s['runs']].index(m)]['metrics']['macro_f1']
        axes[0].barh(i, f1, color=COLORS[m]); axes[0].text(f1 + .003, i, f'{f1:.3f}', va='center')
    axes[0].set(yticks=range(4), yticklabels=list(NAMES.values()), xlim=(.8, 1), xlabel='Validation macro-F1 (seeds 0/1/2 identical)')
    axes[0].invert_yaxis()
    for i, (k, c) in enumerate(s['comparisons'].items()):
        lo, hi = c['macro_f1_ci']
        axes[1].errorbar(c['macro_f1'], i, xerr=[[c['macro_f1'] - lo], [hi - c['macro_f1']]], fmt='o', capsize=4, color='#333')
        axes[1].text(hi + .004, i, f"{c['macro_f1']:+.3f} [{lo:+.3f}, {hi:+.3f}]", va='center', fontsize=8)
    axes[1].axvline(0, color='#c0392b', ls='--')
    axes[1].set(yticks=range(len(s['comparisons'])), yticklabels=list(s['comparisons']), xlim=(-.08, .26),
                xlabel='Paired difference in macro-F1, 95% spatial-block bootstrap')
    axes[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / 'fig1_scores.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 2. Confusion matrices.
    p = pd.read_csv(OUT / 'validation_predictions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, m in zip(axes, ['quantum_kernel', 'matched_rbf', 'reference']):
        cm = confusion_matrix(p.target, p[m], labels=LABELS)
        ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap='Purples')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > 30 else 'black')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, title=NAMES[m], xlabel='Predicted', ylabel='True')
    fig.tight_layout(); fig.savefig(OUT / 'fig2_confusion.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 3. Ablation (entanglement, depth) and learning curve.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ab = s['ablation']
    keys = ['entangled_reps1', 'entangled_reps2', 'entangled_reps3', 'no_entanglement', 'matched_rbf']
    labels = ['entangled\n1 rep', 'entangled\n2 reps', 'entangled\n3 reps', 'no entanglement\n1 rep', 'matched\nRBF']
    for i, k in enumerate(keys):
        axes[0].scatter(i, ab[k]['best']['f1'], s=70, color='#6a3fb5' if 'ent' in k and 'no' not in k else '#d9822b' if k == 'matched_rbf' else '#b39ddb', zorder=3)
        axes[0].scatter(i, ab[k]['grid_mean_f1'], s=70, marker='D', facecolor='none', edgecolor='#555')
    axes[0].set(xticks=range(5), xticklabels=labels, ylabel='Validation macro-F1', title='Filled: tuned best · open: mean over its grid')
    axes[0].grid(alpha=.3)
    c = pd.DataFrame(s['learning_curve'])
    for m in ['quantum_kernel', 'matched_rbf', 'reference']:
        axes[1].plot(c['size'], c[m], marker='o', color=COLORS[m], label=NAMES[m])
    axes[1].set(xlabel='Training fields (encoder refit and settings re-tuned per size)', ylabel='Validation macro-F1', xticks=c['size'])
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'fig3_ablation_learning.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    # 4. Optional IBM hardware run (only if its report exists): kernel entries vs exact, and the finite-shot study.
    hw = OUT / 'hardware'
    reports = sorted(hw.glob('report_*.json'), key=lambda f: json.loads(f.read_text()).get('twirling', False)) if hw.exists() else []
    if reports and (hw / 'shots_study.json').exists():
        rs = [json.loads(f.read_text()) for f in reports if json.loads(f.read_text()).get('twirling')]
        ks = [np.load(hw / f"kernels_{r['job_id']}.npz") for r in rs]
        cat = lambda key, sl=slice(None): np.concatenate([k[key][:, sl].ravel() for k in ks])
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        axes[0].scatter(cat('exact'), cat('hardware', slice(0, -1)), s=6, alpha=.4, color='#7f8c99', label='hardware, raw')
        axes[0].scatter(cat('exact'), cat('corrected'), s=6, alpha=.4, color='#6a3fb5', label='hardware, self-pair corrected')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set(xlabel='Exact kernel value', ylabel='Estimated on ' + rs[0]['backend'],
                    title=f"{sum(len(r['fields']) for r in rs)} fields · {sum(r['circuits'] for r in rs)} circuits · jobs " + ', '.join(r['job_id'][:8] for r in rs))
        axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
        st = json.loads((hw / 'shots_study.json').read_text()); d = pd.DataFrame(st['rows'])
        for m, c, lab in [('ideal', '#2e8b57', 'ideal sampler'), ('hardware_like', '#6a3fb5', f"depolarized (lam={st['lam']}) + corrected")]:
            e = d[d.model == m]; axes[1].errorbar(e.shots, e.macro_f1_mean, yerr=e.macro_f1_sd, marker='o', capsize=3, color=c, label=lab)
        axes[1].axhline(st['exact_macro_f1'], color='k', ls=':', label='exact simulation')
        axes[1].set(xscale='log', xlabel='Shots per kernel entry', ylabel='Validation macro-F1 (150 fields)', ylim=(.3, 1), title='Same fitted SVM, finite-shot kernel rows')
        axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / 'fig4_hardware.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
