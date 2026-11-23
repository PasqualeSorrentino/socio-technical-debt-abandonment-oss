"""
RQ2 — definitions of dormancy, recency and age, minimum detectable effect.

  - project labels under v1 (temporal), v2 (maintenance-based) and v3 (active under both);
  - cross-tabulation of v1 and v2;
  - project-level models with all six socio-technical factors under v1, v2 and v3:
    logit with team size, lines of code and the age of the project at its last commit,
    Cox with team size and lines of code (the length of the history is the time axis);
  - v2 recomputed without the commits of bot accounts;
  - team size under v1 and v2;
  - minimum detectable hazard ratio of the scale-adjusted project-level Cox model
    (Schoenfeld's formula, with the Hsieh-Lavori variance inflation).
Outputs: extended/outputs/rq2_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from scipy import stats

from panel import (OUT_DIR, INACTIVITY_THRESHOLD, PROJECT_ACTIVITY_WINDOWS, PROJECT_COMMIT_THRESHOLD,
                   load_panel, add_bot_flags, fmt_p)

print('Loading panel...')
df, _, _ = load_panel(extra=False)
df = add_bot_flags(df)

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
P['log_age_last_commit'] = np.log1p(P['duration'])  # age at the last commit, in windows
P['quarters_since_last'] = P['global_last'] - P['last_obs']

for v in ('v1', 'v2', 'v3'):
    print(f'  {v}: active={int((P[v] == 0).sum())} dormant={int((P[v] == 1).sum())}')

# ── Cross-tabulation v1 x v2 ───────────────────────────────────────────────
ct = pd.crosstab(P['v2'].map({0: 'v2 active', 1: 'v2 dormant'}), P['v1'].map({0: 'v1 active', 1: 'v1 dormant'}))
ct.to_csv(OUT_DIR / 'rq2_crosstab_v1_v2.csv')
gap = P[(P.v2 == 0) & (P.v1 == 1)]['quarters_since_last']
print(ct.to_string())
print(f'  v2-active but v1-dormant: {len(gap)} of {int((P.v2 == 0).sum())}, '
      f'median {gap.median():.0f} quarters since their last activity')

# ── Team size by definition ────────────────────────────────────────────────
ts = []
for v in ('v1', 'v2', 'v3'):
    a, d = P[P[v] == 0]['n_dev'], P[P[v] == 1]['n_dev']
    ts.append({'definition': v, 'median_active': a.median(), 'median_dormant': d.median(),
               'mannwhitney_p': stats.mannwhitneyu(a, d).pvalue})
pd.DataFrame(ts).to_csv(OUT_DIR / 'rq2_team_size_by_definition.csv', index=False)
print(pd.DataFrame(ts).round(3).to_string(index=False))

# ── Six-factor models under v1, v2, v3 ─────────────────────────────────────
SIX = ['SD_truck', 'SD_community', 'SD_stcongr', 'TD_trad', 'TD_ml', 'TD_vuln']
# In the logit, the age at the last commit replaces the number of observed windows;
# in the Cox model the length of the history is the time axis and is not a covariate.
LOGIT_CTRL = ['log_n_dev', 'log_loc', 'log_age_last_commit']
COX_CTRL = ['log_n_dev', 'log_loc']


def z(frame, cols):
    out = frame[cols].copy()
    for c in cols:
        sd = out[c].std()
        out[c] = (out[c] - out[c].mean()) / sd if sd > 0 else 0.0
    return out


rows = []
for v in ('v1', 'v2', 'v3'):
    X = z(P, SIX + LOGIT_CTRL)
    m = sm.Logit(P[v], sm.add_constant(X)).fit(disp=0, maxiter=200)
    for c in SIX + LOGIT_CTRL:
        rows.append({'definition': v, 'model': 'logit_six_factors_age', 'predictor': c,
                     'estimate': np.exp(m.params[c]), 'p': m.pvalues[c]})
    C = pd.concat([P[['duration']], P[[v]].rename(columns={v: 'event'}), z(P, SIX + COX_CTRL)], axis=1)
    cph = CoxPHFitter(penalizer=0.1).fit(C, 'duration', 'event')
    for c in SIX + COX_CTRL:
        rows.append({'definition': v, 'model': 'cox_six_factors', 'predictor': c,
                     'estimate': cph.summary.loc[c, 'exp(coef)'], 'p': cph.summary.loc[c, 'p']})
R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / 'rq2_six_factor_models.csv', index=False)
print('\nSix-factor models (OR / HR per SD):')
print(R.pivot_table(index='predictor', columns=['model', 'definition'], values='estimate').round(2).to_string())
print('p-values:')
print(R.pivot_table(index='predictor', columns=['model', 'definition'], values='p').round(3).to_string())

# ── v2 without bot commits ─────────────────────────────────────────────────
nb = df[df['is_bot'] == 0].copy()
nb['_c'] = pd.to_numeric(nb['commits_raw'], errors='coerce').fillna(0)
pw = nb.groupby(['project_url', 'time_window_rank'], as_index=False)['_c'].sum()
lab = {}
for proj, g in pw.groupby('project_url'):
    pos = g[g['_c'] > 0]
    if pos.empty:
        lab[proj] = 1
        continue
    last = pos['time_window_rank'].max()
    c12 = g.loc[g['time_window_rank'].between(last - PROJECT_ACTIVITY_WINDOWS + 1, last), '_c'].sum()
    lab[proj] = int(c12 < PROJECT_COMMIT_THRESHOLD)
P['v2_no_bots'] = P['project_url'].map(lab).fillna(1).astype(int)
changed = int((P['v2_no_bots'] != P['v2']).sum())
print(f'\nv2 without bot commits: {int(P.v2_no_bots.sum())} dormant (vs {int(P.v2.sum())}), {changed} labels change')

# ── Minimum detectable effect (scale-adjusted Cox under v2) ────────────────
SOCTEC = ['SD_truck', 'SD_community', 'TD_trad', 'TD_vuln']
SCALE = ['log_n_dev', 'log_n_windows', 'log_loc']
D = int(P['v2'].sum())
zsum = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
Z = z(P, SOCTEC + SCALE)
mde = [{'factor': 'any (no collinearity)', 'R2_with_other_predictors': 0.0,
        'min_HR_risk': np.exp(zsum / np.sqrt(D)), 'min_HR_protective': np.exp(-zsum / np.sqrt(D))}]
for f in SOCTEC:
    others = [c for c in SOCTEC + SCALE if c != f]
    r2 = sm.OLS(Z[f], sm.add_constant(Z[others])).fit().rsquared
    mde.append({'factor': f, 'R2_with_other_predictors': r2,
                'min_HR_risk': np.exp(zsum / np.sqrt(D * (1 - r2))),
                'min_HR_protective': np.exp(-zsum / np.sqrt(D * (1 - r2)))})
M = pd.DataFrame(mde)
M.to_csv(OUT_DIR / 'rq2_minimum_detectable_effect.csv', index=False)
print(f'\nMinimum detectable HR ({D} events, alpha=0.05 two-sided, power=0.80):')
print(M.round(3).to_string(index=False))

P.drop(columns=['loc']).to_csv(OUT_DIR / 'rq2_project_labels.csv', index=False)
print('\nDone. Outputs in', OUT_DIR)
