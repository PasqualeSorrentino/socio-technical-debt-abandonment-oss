"""
RQ1 — robustness checks of the developer-level models (Table R1 of the paper).

Starts from the same panel as the main pipeline (panel.py) and, after verifying that the
later-stage Cox model reproduces Table 2, estimates:
  - the model without tenure, with standard errors clustered by project, with a
    Benjamini-Hochberg correction, stratified by tenure terciles, without bots, with the
    individual role and team size, on the projects active under v2, stratified by project;
  - the proportional hazards test (scaled Schoenfeld residuals);
  - lagged leave-one-out (LOO) versions of team turnover, core turnover and ratio of
    smelly quitters, in the later-stage and in the time-varying Cox models;
  - the early-stage logit with clustered standard errors and with role and team size;
  - the validation of the twelve-month departure threshold.
All results are written to extended/outputs/rq1_*.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests

from panel import (OUT_DIR, EARLY_STAGE_WINDOWS, INACTIVITY_THRESHOLD, load_panel, add_core_flags,
                   add_bot_flags, later_stage_panel, early_stage_panel, check_main_model, fmt_p)

KEY = ['stqf_global_truck', 'stqf_ratio_smelly_quitters', 'stqf_global_turnover',
       'stqf_core_code_turnover', 'stqf_ratio_sponsored', 'log_traditional_smell_count',
       'stqf_st_congruence', 'stqf_communicability', 'stqf_ratio_smelly_devs']
LOO = {'stqf_global_turnover': 'loo_turnover', 'stqf_core_code_turnover': 'loo_core_turnover',
       'stqf_ratio_smelly_quitters': 'loo_smelly_quitters'}

print('Loading panel...')
df, FEAT_MODEL, project_status = load_panel(extra=True)
df = add_core_flags(df)
df = add_bot_flags(df)
team = df.groupby('project_url')['developer_id'].nunique()
df['log_team_size'] = np.log1p(df['project_url'].map(team))
FEATS = check_main_model(df, FEAT_MODEL)
print(f'Table 2 reproduced ({len(FEATS)} covariates).')

rows = []  # long-format record of every estimate: spec, covariate, HR, CI, p


def fit_cox(panel, feats, spec, **kw):
    cols = list(dict.fromkeys(feats + ['duration', 'event'] + [c for c in ('project_url', 'stratum') if c in panel.columns and (kw.get('cluster_col') == c or c in (kw.get('strata') or []))]))
    cph = CoxPHFitter(penalizer=0.1).fit(panel[cols], 'duration', 'event', **kw)
    s = cph.summary
    for cov, r in s.iterrows():
        rows.append({'spec': spec, 'covariate': cov, 'HR': r['exp(coef)'],
                     'ci_low': r['exp(coef) lower 95%'], 'ci_high': r['exp(coef) upper 95%'],
                     'p': r['p'], 'n_pairs': len(panel), 'n_events': int(panel['event'].sum()),
                     'concordance': cph.concordance_index_})
    print(f"  {spec:42s} n={len(panel):5d} C={cph.concordance_index_:.3f} | "
          + ' '.join(f"{k.replace('stqf_', '')[:14]}={s.loc[k, 'exp(coef)']:.2f}({fmt_p(s.loc[k, 'p'])})"
                     for k in KEY[:5] + list(LOO.values()) if k in s.index))
    return cph


# ── Lagged leave-one-out metrics ────────────────────────────────────────────
print('Building lagged leave-one-out metrics...')
pair_last = df.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('max')
df['_leaves_after'] = ((df['time_window_rank'] == pair_last) & (df['has_failed'] == 1)).astype(int)
df['_smelly_leaver'] = df['_leaves_after'] * (df['community_smell_raw'] > 0).astype(int)
w = df.groupby(['project_url', 'time_window_rank']).agg(
    n=('developer_id', 'size'), lv=('_leaves_after', 'sum'), slv=('_smelly_leaver', 'sum'),
    n_core=('is_core', 'sum'),
    lv_core=('_leaves_after', lambda s: int((s * df.loc[s.index, 'is_core']).sum()))).reset_index()
w['time_window_rank'] += 1  # window t-1 seen from window t
df = df.merge(w, on=['project_url', 'time_window_rank'], how='left')
for c in ['n', 'lv', 'slv', 'n_core', 'lv_core']:
    df[c] = df[c].fillna(0)
# Leave-one-out: remove developer i from the t-1 counts if i was active in t-1
# (i is active in t, so i is never among the leavers of t-1).
prev = df[['project_url', 'developer_id', 'time_window_rank']].copy()
prev['time_window_rank'] += 1
prev['_i_in_prev'] = 1
df = df.merge(prev, on=['project_url', 'developer_id', 'time_window_rank'], how='left')
df['_i_in_prev'] = df['_i_in_prev'].fillna(0)
core_prev = df[['project_url', 'developer_id', 'time_window_rank', 'is_core']]
den = df['n'] - df['_i_in_prev']
den_core = df['n_core'] - df['_i_in_prev'] * df['is_core']
df['loo_turnover'] = np.where(den > 0, df['lv'] / den.where(den > 0, 1), 0.0)
df['loo_core_turnover'] = np.where(den_core > 0, df['lv_core'] / den_core.where(den_core > 0, 1), 0.0)
df['loo_smelly_quitters'] = np.where(df['lv'] > 0, df['slv'] / df['lv'].where(df['lv'] > 0, 1), 0.0)
corr = {orig: df[orig].corr(df[new]) for orig, new in LOO.items()}
print('  correlation with the original metrics: ' + ', '.join(f'{k}={v:.2f}' for k, v in corr.items()))
pd.DataFrame([{'metric': k, 'loo_metric': LOO[k], 'pearson_r_row_level': v} for k, v in corr.items()]
             ).to_csv(OUT_DIR / 'rq1_loo_correlations.csv', index=False)

# ── Later-stage panel with every extra column needed by the checks ──────────
extra = ['is_core', 'is_bot', 'log_team_size', 'proj_failed']
P = later_stage_panel(df, list(dict.fromkeys(FEATS + list(LOO.values()))), extra_cols=extra)
P['log_team_size'] = P['log_team_size']
print(f'Later-stage pairs: {len(P)} (events {int(P.event.sum())}), projects {P.project_url.nunique()}')

print('\nLater-stage Cox, robustness specifications:')
main = fit_cox(P, FEATS, 'main')
fit_cox(P, [f for f in FEATS if f != 'developer_tenure'], 'without_tenure')
fit_cox(P, FEATS, 'clustered_by_project', cluster_col='project_url')

# Benjamini-Hochberg on all the coefficients of the main model
s = main.summary
rej, q, _, _ = multipletests(s['p'], method='fdr_bh')
bh = pd.DataFrame({'covariate': s.index, 'HR': s['exp(coef)'], 'p': s['p'], 'q_bh': q, 'significant_bh': rej})
bh.to_csv(OUT_DIR / 'rq1_benjamini_hochberg.csv', index=False)
print(f"  Benjamini-Hochberg on {len(s)} coefficients: significant -> "
      + ', '.join(bh.loc[bh.significant_bh, 'covariate']))

# Proportional hazards: scaled Schoenfeld residuals
ph = proportional_hazard_test(main, P[FEATS + ['duration', 'event']], time_transform='rank')
ph.summary.reset_index().rename(columns={'index': 'covariate'}).to_csv(OUT_DIR / 'rq1_schoenfeld_test.csv', index=False)
viol = ph.summary[ph.summary['p'] < 0.05]
print('  PH assumption violated (p<0.05): ' + ', '.join(f'{c}(p={fmt_p(p)})' for c, p in viol['p'].items()))

# Terciles of the (averaged) tenure; tenure enters the model only through the strata.
P['stratum'] = pd.qcut(P['developer_tenure'], 3, labels=False, duplicates='drop')
fit_cox(P, [f for f in FEATS if f != 'developer_tenure'], 'stratified_tenure_terciles', strata=['stratum'])
P = P.drop(columns='stratum')

nob = P[P['is_bot'] == 0]
print(f'  bot accounts: {int(df.loc[df.is_bot == 1, "developer_id"].nunique())} developers, '
      f'{int(df[df.is_bot == 1].groupby(["project_url", "developer_id"]).ngroups)} project-developer pairs')
fit_cox(nob, FEATS, 'without_bots')
RT = FEATS + ['is_core', 'log_team_size']
fit_cox(P, RT, 'role_and_team_size')
fit_cox(nob, RT, 'role_team_size_no_bots_clustered', cluster_col='project_url')
fit_cox(P[P['proj_failed'] == 0], FEATS, 'active_projects_only_v2')
fit_cox(P, FEATS, 'stratified_by_project', strata=['project_url'])

# Lagged leave-one-out
FL = [LOO.get(f, f) for f in FEATS]
fit_cox(P, FL, 'loo_metrics')
fit_cox(P, FL, 'loo_metrics_clustered', cluster_col='project_url')

# Extended model with SD x TD interactions is already in the main pipeline (STEP 9).
res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / 'rq1_cox_robustness_long.csv', index=False)

# ── Time-varying Cox, original and LOO ─────────────────────────────────────
print('\nTime-varying Cox:')
ev = df.groupby(['project_url', 'developer_id']).agg(event=('has_failed', 'max'),
                                                      n_win=('time_window_rank', 'nunique')).reset_index()
keep = ev[ev.n_win >= 3]
sub = df.merge(keep[['project_url', 'developer_id', 'event']].rename(columns={'event': 'event_pair'}),
               on=['project_url', 'developer_id'])


def time_varying(cols, label):
    long = []
    for k, (key, g) in enumerate(sub.groupby(['project_url', 'developer_id'])):
        g = g.sort_values('developer_tenure')
        ten = g['developer_tenure'].values
        evp = int(g['event_pair'].iloc[0])
        vals = g[cols].values
        for j in range(len(g)):
            start = ten[j]
            stop = ten[j + 1] if j < len(g) - 1 else ten[j] + 1
            if stop <= start:
                stop = start + 1
            long.append([k, start, stop, evp if j == len(g) - 1 else 0, *vals[j]])
    L = pd.DataFrame(long, columns=['id', 'start', 'stop', 'event', *cols])
    for c in cols:
        sd = L[c].std()
        if sd > 0:
            L[c] = (L[c] - L[c].mean()) / sd
    ctv = CoxTimeVaryingFitter(penalizer=0.1).fit(L, id_col='id', event_col='event',
                                                  start_col='start', stop_col='stop')
    out = ctv.summary[['exp(coef)', 'p']].copy()
    out['spec'] = label
    print(f'  {label}: ' + ', '.join(f"{i}={r['exp(coef)']:.2f}({fmt_p(r['p'])})" for i, r in out.iterrows()))
    return out


ADV = ['community_smell_count', 'stqf_global_turnover', 'stqf_st_congruence',
       'traditional_smell_count', 'vulnerability_count', 'log_commits_count', 'se_score', 'sentiment_score']
tv = [time_varying(ADV, 'original'),
      time_varying([LOO.get(c, c) for c in ADV] + ['loo_smelly_quitters'], 'loo')]
pd.concat(tv).reset_index().rename(columns={'index': 'covariate'}).to_csv(
    OUT_DIR / 'rq1_timevarying_cox.csv', index=False)

# ── Early stage: clustered SE, role and team size ──────────────────────────
print('\nEarly-stage logit:')
E = early_stage_panel(df, FEAT_MODEL, extra_cols=['is_core', 'log_team_size'])
erows = []


def early(cols, label, cluster=False):
    X = E[cols].fillna(0)
    X = X.loc[:, X.std() > 0]
    X = (X - X.mean()) / X.std(ddof=0)
    kw = {'cov_type': 'cluster', 'cov_kwds': {'groups': pd.factorize(E['project_url'])[0]}} if cluster else {}
    m = sm.GLM(E['has_failed'], sm.add_constant(X), family=sm.families.Binomial()).fit(maxiter=100, **kw)
    for c in m.params.index:
        erows.append({'spec': label, 'covariate': c, 'OR': np.exp(m.params[c]), 'p': m.pvalues[c]})
    print(f'  {label:28s} ' + ', '.join(f"{c.replace('stqf_', '')[:14]}={np.exp(m.params[c]):.2f}({fmt_p(m.pvalues[c])})"
                                         for c in ['stqf_global_truck', 'stqf_ratio_smelly_quitters', 'stqf_ratio_smelly_devs',
                                                   'stqf_communicability', 'log_bug_introduced_count_rszz', 'log_team_size']
                                         if c in m.params.index))


early(FEAT_MODEL, 'main')
early(FEAT_MODEL, 'clustered_by_project', cluster=True)
early(FEAT_MODEL + ['is_core', 'log_team_size'], 'role_and_team_size')
pd.DataFrame(erows).to_csv(OUT_DIR / 'rq1_early_stage_robustness.csv', index=False)

# ── Validation of the twelve-month departure threshold ─────────────────────
print('\nDeparture threshold:')
act = df[['project_url', 'developer_id', 'time_window_rank', 'is_core', 'project_last_observed_rank']]
gaps = []
for (proj, dev), g in act.groupby(['project_url', 'developer_id']):
    r = np.sort(g['time_window_rank'].unique())
    horizon = int(g['project_last_observed_rank'].iloc[0])
    core = int(g['is_core'].iloc[0])
    nxt = np.append(r[1:], np.inf)
    for a, b in zip(r, nxt):
        gaps.append((core, a, b, horizon))
G = pd.DataFrame(gaps, columns=['is_core', 'from', 'next', 'horizon'])
thr = []
# An absence that reaches k windows is eligible if the project is observed for at least
# eight further windows; it counts as a return if the developer is active again within
# those eight windows (i.e. within two years).
for k in range(1, 13):
    reach = G[(G['next'] > G['from'] + k) & (G['horizon'] >= G['from'] + k + 8)]
    ret = (reach['next'] <= reach['from'] + k + 8)
    thr.append({'k': k, 'absences': len(reach), 'share_returned': ret.mean(),
                'share_returned_core': ret[reach.is_core == 1].mean(),
                'share_returned_peripheral': ret[reach.is_core == 0].mean()})
T = pd.DataFrame(thr)
T.to_csv(OUT_DIR / 'rq1_threshold_returns.csv', index=False)
print(T[T.k.isin([4, 8])].round(3).to_string(index=False))
dep = df.groupby(['project_url', 'developer_id']).agg(ev=('has_failed', 'max'), lag=('inactivity_lag', 'max'))
lag = dep[dep.ev == 1]['lag']
print(f'  departed developers: median inactivity {lag.median():.0f} windows, '
      f'{(lag >= 8).mean():.1%} at least eight')
pairs = df.groupby(['project_url', 'developer_id']).agg(ev=('has_failed', 'max'), core=('is_core', 'max'))
print(f"  core events: {int(pairs[(pairs.ev == 1) & (pairs.core == 1)].shape[0])} of {int(pairs.ev.sum())}")
print('\nDone. Outputs in', OUT_DIR)
