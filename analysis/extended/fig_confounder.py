"""
Figure of the paper: team size as a confounder of project dormancy.
Distribution of log(1 + number of developers) in active and inactive projects under the
maintenance-based (v2) and temporal (v1) definitions. Reads rq2_project_labels.csv
(written by rq2_extras.py) and writes outputs/fig_confounder_teamsize.pdf.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panel import OUT_DIR

BLUE, RED, DARK = '#4F6D8A', '#973327', '#3C4148'
P = pd.read_csv(OUT_DIR / 'rq2_project_labels.csv')
y = np.log1p(P['n_dev'])

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': DARK,
                     'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK})
fig, ax = plt.subplots(figsize=(6.2, 3.0))
groups, labels, colors = [], [], []
for d, c in (('v2', BLUE), ('v1', RED)):
    for lab, v in (('Active', 0), ('Inactive', 1)):
        g = y[P[d] == v]
        groups.append(g)
        labels.append(f'{lab}\n(n={len(g)})')
        colors.append(c)
pos = [1, 2, 3.6, 4.6]
bp = ax.boxplot(groups, positions=pos, widths=0.55, patch_artist=True, showfliers=True,
                medianprops={'color': 'white', 'linewidth': 1.6},
                whiskerprops={'color': DARK, 'linewidth': 0.8}, capprops={'color': DARK, 'linewidth': 0.8},
                flierprops={'marker': 'o', 'markersize': 2.5, 'markerfacecolor': DARK, 'markeredgewidth': 0, 'alpha': 0.5})
for patch, c in zip(bp['boxes'], colors):
    patch.set(facecolor=c, edgecolor=c, alpha=0.9)
ax.set_xticks(pos)
ax.set_xticklabels(labels)
ax.set_ylabel('log(1 + number of developers)')
ax.axvline(2.8, color='#B8BDC3', linewidth=0.8, linestyle='--')
top = y.max() * 1.08
ax.text(1.5, top, 'v2 (maintenance-based)', ha='center', color=BLUE, fontweight='bold')
ax.text(4.1, top, 'v1 (temporal)', ha='center', color=RED, fontweight='bold')
ax.set_ylim(0, top * 1.08)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color='#E3E5E8', linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_confounder_teamsize.pdf')
print('Written', OUT_DIR / 'fig_confounder_teamsize.pdf')
