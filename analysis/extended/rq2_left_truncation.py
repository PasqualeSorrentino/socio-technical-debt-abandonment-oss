"""
RQ2 — left truncation of the project histories.

NICHE includes only projects with at least one commit after May 2020. A project whose history
started earlier is therefore in the sample only because it survived until then, so the project-level
Cox models, whose time axis is the length of the history, are left-truncated. This script refits them
with delayed entry: each project enters the risk set at the quarter of May 2020, measured from its
first window (or at its first window, if it started later), and is never at risk before.

Models: the scale-adjusted Cox model of RQ2 (four socio-technical factors and three structural
controls) and the six-factor Cox model with team size and lines of code, under v1, v2 and v3.
Outputs: extended/outputs/rq2_left_truncation.csv
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from panel import OUT_DIR, INACTIVITY_THRESHOLD, load_panel, quarter_index, fmt_p

print('Loading panel...')
df, _, _ = load_panel(extra=False)

P = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'), n_win=('time_window_rank', 'nunique'),
    loc=('project_loc', 'max'),
    SD_truck=('stqf_global_truck', 'mean'), SD_community=('community_smell_count', 'mean'),
    SD_stcongr=('stqf_st_congruence', 'mean'), TD_trad=('traditional_smell_count', 'mean'),
    TD_ml=('ml_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    first=('time_window_rank', 'min'), last_obs=('project_last_observed_rank', 'max'),
    last_commit=('project_last_commit_rank', 'max'), global_last=('global_last_rank', 'max'),
    v2=('proj_failed', 'max')).reset_index()
P['v1'] = ((P['global_last'] - P['last_obs']) >= INACTIVITY_THRESHOLD).astype(int)
P['v3'] = 1 - ((P['v1'] == 0) & (P['v2'] == 0)).astype(int)
P['log_n_dev'] = np.log1p(P['n_dev'])
P['log_n_windows'] = np.log1p(P['n_win'])
P['log_loc'] = np.log1p(pd.to_numeric(P['loc'], errors='coerce').fillna(0))
P['duration'] = P['last_commit'] - P['first'] + 1

SELECTION_QUARTER = quarter_index('20200501')  # quarter that contains May 2020
P['entry'] = (SELECTION_QUARTER - P['first']).clip(lower=0)
bad = P['entry'] >= P['duration']
print(f'Projects started before the selection quarter: {int((P.entry > 0).sum())} of {len(P)}; '
      f'median delayed entry {P.loc[P.entry > 0, "entry"].median():.0f} windows; '
      f'{int(bad.sum())} with entry not before the last commit (entry set just before it)')
P.loc[bad, 'entry'] = P.loc[bad, 'duration'] - 0.5

SOCTEC = ['SD_truck', 'SD_community', 'TD_trad', 'TD_vuln']
SPECS = {'scale_adjusted': SOCTEC + ['log_n_dev', 'log_n_windows', 'log_loc'],
         'six_factors': ['SD_truck', 'SD_community', 'SD_stcongr', 'TD_trad', 'TD_ml', 'TD_vuln',
                         'log_n_dev', 'log_loc']}


def z(frame, cols):
    out = frame[cols].copy()
    for c in cols:
        s = out[c].std()
        out[c] = (out[c] - out[c].mean()) / s if s > 0 else 0.0
    return out


rows = []
for v in ('v2', 'v1', 'v3'):
    for spec, cols in SPECS.items():
        for entry in (False, True):
            C = pd.concat([P[['duration', 'entry']], P[[v]].rename(columns={v: 'event'}), z(P, cols)], axis=1)
            kw = {'entry_col': 'entry'} if entry else {}
            if not entry:
                C = C.drop(columns='entry')
            cph = CoxPHFitter(penalizer=0.1).fit(C, 'duration', 'event', **kw)
            s = cph.summary
            for c in cols:
                rows.append({'definition': v, 'spec': spec, 'delayed_entry': entry, 'predictor': c,
                             'HR': s.loc[c, 'exp(coef)'], 'ci_low': s.loc[c, 'exp(coef) lower 95%'],
                             'ci_high': s.loc[c, 'exp(coef) upper 95%'], 'p': s.loc[c, 'p'],
                             'events': int(P[v].sum())})
            print(f"{v} {spec:15s} entry={'yes' if entry else 'no '} " + ', '.join(
                f"{c}={s.loc[c, 'exp(coef)']:.2f}({fmt_p(s.loc[c, 'p'])})" for c in cols))

pd.DataFrame(rows).to_csv(OUT_DIR / 'rq2_left_truncation.csv', index=False)
print(f'Done. Outputs in {OUT_DIR.name}')
