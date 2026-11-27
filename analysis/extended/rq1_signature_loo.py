"""
RQ1 — pre-departure signature (Figure 2) with the lagged leave-one-out turnover.

The signature of the main pipeline (STEP 16 of pipeline_v2_maintenance_based.py) compares
the first and the last window of activity of the developers who leave, using the team
turnover reported by csDetector for the same window. That metric may include the departure
of the developer under analysis, which happens right after their last window. This script
reproduces the original signature on the same developers, and adds the lagged leave-one-out
turnover and ratio of smelly quitters (panel.add_loo_metrics), which only count the
departures of the OTHER developers active in the previous window.
Outputs: extended/outputs/rq1_signature_loo.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from panel import OUT_DIR, load_panel, add_core_flags, add_loo_metrics

print('Loading panel...')
df, _, _ = load_panel(extra=False)
df = add_core_flags(df)
df = add_loo_metrics(df)
df['log_commits_count'] = np.log1p(pd.to_numeric(df['commits_count'], errors='coerce').fillna(0))

# Same developers as STEP 16: leavers observed in at least three windows.
ev = df.groupby(['project_url', 'developer_id']).agg(
    event=('has_failed', 'max'), n_win=('time_window_rank', 'nunique')).reset_index()
aid = ev[(ev.n_win >= 3) & (ev.event == 1)][['project_url', 'developer_id']]
traj = df.merge(aid, on=['project_url', 'developer_id']).sort_values(
    ['project_url', 'developer_id', 'time_window_rank'])
seq = traj.groupby(['project_url', 'developer_id']).cumcount()
tot = traj.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('size')
pos = seq - (tot - 1)

SIG = ['community_smell_count', 'traditional_smell_count', 'stqf_global_turnover',
       'loo_turnover', 'loo_smelly_quitters', 'log_commits_count']
first = traj[seq == 0].set_index(['project_url', 'developer_id'])[SIG]
last = traj[pos == 0].set_index(['project_url', 'developer_id'])[SIG]
pair = first.add_suffix('_f').join(last.add_suffix('_l')).dropna()
print(f'Leavers with a trajectory: {len(pair):,}')

rows = []
for c in SIG:
    a, b = pair[c + '_f'], pair[c + '_l']
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        p = np.nan
    dpc = (b.mean() - a.mean()) / (abs(a.mean()) + 1e-9) * 100
    rows.append({'metric': c, 'first_mean': a.mean(), 'last_mean': b.mean(), 'delta_pct': dpc, 'p': p,
                 'n': len(pair)})
    print(f'  {c:26s} first={a.mean():6.3f} last={b.mean():6.3f} ({dpc:+.0f}%) p={p:.1e}')

# Sanity check against the original signature (prefailure_signature.csv of the v2 pipeline).
orig = {'stqf_global_turnover': 24.7, 'traditional_smell_count': -57.4, 'log_commits_count': -28.5}
R = pd.DataFrame(rows).set_index('metric')
for k, v in orig.items():
    assert abs(R.loc[k, 'delta_pct'] - v) < 0.5, f'{k}: {R.loc[k, "delta_pct"]:.1f} != {v} (Figure 2)'
R.reset_index().to_csv(OUT_DIR / 'rq1_signature_loo.csv', index=False)
print('\nDone. Outputs in', OUT_DIR)
