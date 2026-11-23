"""
RQ1 — further robustness checks of the developer-level models.

Addresses three threats of the main specification:
  - nesting of developers within projects: a Cox model with a gamma frailty term per
    project, estimated with R's `survival` package (frailty_cox.R), compared with the same
    model without frailty;
  - look-ahead in the averaged covariates: (a) a Cox model whose covariates are measured
    only in the first later-stage window of each developer, i.e. before the time at risk
    that follows, and (b) a time-varying Cox model whose covariates are lagged by one
    window, so that the risk in window t only depends on the metrics of window t-1;
  - multiplicity: Benjamini-Hochberg correction pooled over all the coefficients of all
    the developer-level specifications (robustness Cox models, time-varying models,
    early-stage logits, and the models estimated here).
Outputs: extended/outputs/rq1_extra_*.csv
Requires rq1_robustness.py to have been run first (it reads its outputs).
"""
import shutil
import subprocess

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from statsmodels.stats.multitest import multipletests

from panel import (OUT_DIR, BASE_DIR, EARLY_STAGE_WINDOWS, load_panel, add_core_flags,
                   later_stage_panel, check_main_model, fmt_p)

KEY = ['stqf_global_truck', 'stqf_ratio_smelly_quitters', 'stqf_global_turnover',
       'stqf_core_code_turnover', 'stqf_ratio_sponsored', 'log_traditional_smell_count',
       'stqf_st_congruence', 'stqf_communicability', 'stqf_ratio_smelly_devs']

print('Loading panel...')
df, FEAT_MODEL, _ = load_panel(extra=True)
df = add_core_flags(df)
FEATS = check_main_model(df, FEAT_MODEL)
P = later_stage_panel(df, FEATS)
rows = []


def record(summary, spec, n, events, hr='exp(coef)', lo='exp(coef) lower 95%', hi='exp(coef) upper 95%'):
    for cov, r in summary.iterrows():
        rows.append({'spec': spec, 'covariate': cov, 'HR': r[hr], 'ci_low': r.get(lo, np.nan),
                     'ci_high': r.get(hi, np.nan), 'p': r['p'], 'n_pairs': n, 'n_events': events})
    print(f'  {spec:34s} ' + ' '.join(f"{k.replace('stqf_', '')[:14]}={summary.loc[k, hr]:.2f}({fmt_p(summary.loc[k, 'p'])})"
                                     for k in KEY if k in summary.index))


# ── (1) Gamma frailty per project (R survival) ──────────────────────────────
print('\nFrailty Cox model (R survival, gamma frailty per project):')
rscript = shutil.which('Rscript')
if rscript is None:
    print('  Rscript not found: skipping the frailty model.')
else:
    inp = OUT_DIR / '_frailty_input.csv'
    P[FEATS + ['duration', 'event', 'project_url']].to_csv(inp, index=False)
    out = OUT_DIR / '_frailty_output.csv'
    subprocess.run([rscript, str(BASE_DIR / 'frailty_cox.R'), str(inp), str(out)], check=True)
    F = pd.read_csv(out)
    for spec, g in F.groupby('spec'):
        s = g.set_index('covariate').rename(columns={'HR': 'exp(coef)'})
        record(s, spec, len(P), int(P.event.sum()), lo='ci_low', hi='ci_high')
    print('  frailty variance: ' + ', '.join(f"{v:.3f}" for v in F.loc[F.spec == 'gamma_frailty', 'frailty_variance'].dropna().unique()))
    inp.unlink()
    out.unlink()

# ── (2a) Covariates measured in the first later-stage window only ───────────
print('\nCox model with covariates of the first later-stage window:')
late = df[df['developer_tenure'] > EARLY_STAGE_WINDOWS].sort_values('developer_tenure')
first = late.groupby(['project_url', 'developer_id'])[FEATS].first().reset_index()
B = P[['project_url', 'developer_id', 'duration', 'event']].merge(first, on=['project_url', 'developer_id'])
B = B.replace([np.inf, -np.inf], np.nan).fillna(0)
feats_b = [f for f in FEATS if B[f].std() > 0 and f != 'developer_tenure']
cph = CoxPHFitter(penalizer=0.1).fit(B[feats_b + ['duration', 'event']], 'duration', 'event')
record(cph.summary, 'baseline_covariates', len(B), int(B.event.sum()))
cph = CoxPHFitter(penalizer=0.1).fit(B[feats_b + ['duration', 'event', 'project_url']], 'duration', 'event',
                                     cluster_col='project_url')
record(cph.summary, 'baseline_covariates_clustered', len(B), int(B.event.sum()))

# ── (2b) Time-varying Cox with covariates lagged by one window ─────────────
print('\nTime-varying Cox with covariates lagged by one window:')
ADV = ['community_smell_count', 'stqf_global_turnover', 'stqf_ratio_smelly_quitters', 'stqf_st_congruence',
       'traditional_smell_count', 'vulnerability_count', 'log_commits_count', 'se_score']
ev = df.groupby(['project_url', 'developer_id']).agg(event=('has_failed', 'max'),
                                                      n_win=('time_window_rank', 'nunique')).reset_index()
keep = ev[ev.n_win >= 3]
sub = df.merge(keep[['project_url', 'developer_id', 'event']].rename(columns={'event': 'event_pair'}),
               on=['project_url', 'developer_id'])
long = []
for k, (_, g) in enumerate(sub.groupby(['project_url', 'developer_id'])):
    g = g.sort_values('developer_tenure')
    ten = g['developer_tenure'].values
    vals = g[ADV].fillna(0).values
    evp = int(g['event_pair'].iloc[0])
    for j in range(1, len(g)):  # interval j uses the metrics of the previous observed window
        start = ten[j]
        stop = ten[j + 1] if j < len(g) - 1 else ten[j] + 1
        if stop <= start:
            stop = start + 1
        long.append([k, start, stop, evp if j == len(g) - 1 else 0, *vals[j - 1]])
L = pd.DataFrame(long, columns=['id', 'start', 'stop', 'event', *ADV])
for c in ADV:
    sd = L[c].std()
    if sd > 0:
        L[c] = (L[c] - L[c].mean()) / sd
ctv = CoxTimeVaryingFitter(penalizer=0.1).fit(L, id_col='id', event_col='event', start_col='start', stop_col='stop')
record(ctv.summary, 'time_varying_lagged', L['id'].nunique(), int(L.event.sum()))

R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / 'rq1_extra_models.csv', index=False)

# ── (3) Benjamini-Hochberg pooled over all developer-level specifications ──
print('\nBenjamini-Hochberg pooled over all developer-level specifications:')
fam = [pd.read_csv(OUT_DIR / 'rq1_cox_robustness_long.csv')[['spec', 'covariate', 'p']].assign(model='cox'),
       pd.read_csv(OUT_DIR / 'rq1_timevarying_cox.csv')[['spec', 'covariate', 'p']].assign(model='time_varying'),
       pd.read_csv(OUT_DIR / 'rq1_early_stage_robustness.csv').query('covariate != "const"')[['spec', 'covariate', 'p']].assign(model='early_logit'),
       R[['spec', 'covariate', 'p']].assign(model='extra')]
A = pd.concat(fam, ignore_index=True).dropna(subset=['p'])
A['q_bh_pooled'] = multipletests(A['p'], method='fdr_bh')[1]
A['significant_p'] = A['p'] < 0.05
A['significant_q'] = A['q_bh_pooled'] < 0.05
A.to_csv(OUT_DIR / 'rq1_extra_bh_pooled.csv', index=False)
print(f"  {len(A)} coefficients; significant at p<0.05: {int(A.significant_p.sum())}, "
      f"after pooled BH: {int(A.significant_q.sum())}")
main = A[(A.model == 'cox') & (A.spec == 'main')]
print('  main model, significant after pooled BH: '
      + ', '.join(main.loc[main.significant_q, 'covariate']))
lost = main[main.significant_p & ~main.significant_q]['covariate'].tolist()
print('  main model, lost after pooled BH: ' + (', '.join(lost) if lost else 'none'))
print('\nDone. Outputs in', OUT_DIR)
