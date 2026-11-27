"""
RQ1 — what the ratio of smelly quitters measures.

The ratio of smelly quitters is the share of the developers who left that were involved in
community smells. It combines how many people left (turnover) with who they were (smelly or not).
To separate the two parts, this script refits the later-stage Cox model and the early-stage logit
in two ways, keeping every other covariate unchanged:

  - decomposition: team turnover and the ratio of smelly quitters are replaced by
      smelly turnover     = turnover x ratio of smelly quitters      (smelly leavers / team)
      non-smelly turnover = turnover x (1 - ratio of smelly quitters) (other leavers / team)
    If the association comes from the smells of those who left, smelly turnover should carry more
    risk than non-smelly turnover; if it comes from turnover, the two should carry similar risk.
  - interaction: team turnover, the ratio of smelly quitters, and their product.

Each specification is estimated with the original metrics and with the lagged leave-one-out ones,
plain, with standard errors clustered by project, and stratified by project. The new terms are
standardized, so their hazard ratios are per one standard deviation.
Outputs: extended/outputs/rq1_smelly_quitters_decomposition.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from lifelines import CoxPHFitter

from panel import (OUT_DIR, load_panel, add_core_flags, add_loo_metrics, later_stage_panel, early_stage_panel,
                   check_main_model, fmt_p)

print('Loading panel...')
df, FEAT_MODEL, _ = load_panel(extra=True)
df = add_core_flags(df)
FEATS = check_main_model(df, FEAT_MODEL)
df = add_loo_metrics(df)

VARIANTS = {'original': ('stqf_global_turnover', 'stqf_ratio_smelly_quitters', 'stqf_core_code_turnover'),
            'loo': ('loo_turnover', 'loo_smelly_quitters', 'loo_core_turnover')}
for name, (turn, ratio, _) in VARIANTS.items():
    df[f'{name}_smelly_turnover'] = df[turn] * df[ratio]
    df[f'{name}_nonsmelly_turnover'] = df[turn] * (1 - df[ratio])

rows = []


def z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def record(model, variant, spec, se, est, ci_lo, ci_hi, p, n, events, terms):
    for t in terms:
        rows.append({'model': model, 'variant': variant, 'spec': spec, 'se': se, 'term': t,
                     'estimate': est[t], 'ci_low': ci_lo[t], 'ci_high': ci_hi[t], 'p': p[t],
                     'n': n, 'events': events})


for variant, (turn, ratio, core) in VARIANTS.items():
    base = [f for f in FEATS if f not in ('stqf_global_turnover', 'stqf_ratio_smelly_quitters',
                                          'stqf_core_code_turnover')] + [core]
    st, ns = f'{variant}_smelly_turnover', f'{variant}_nonsmelly_turnover'
    P = later_stage_panel(df, list(dict.fromkeys(base + [turn, ratio, st, ns])))
    P['z_smelly_turnover'], P['z_nonsmelly_turnover'] = z(P[st]), z(P[ns])
    P['z_turnover'], P['z_ratio'] = z(P[turn]), z(P[ratio])
    P['z_turnover_x_ratio'] = P['z_turnover'] * P['z_ratio']
    specs = {'decomposition': ['z_smelly_turnover', 'z_nonsmelly_turnover'],
             'interaction': ['z_turnover', 'z_ratio', 'z_turnover_x_ratio']}
    for spec, terms in specs.items():
        feats = [f for f in base if P[f].std() > 0] + terms
        for se, kw, extra in (('plain', {}, []), ('clustered', {'cluster_col': 'project_url'}, ['project_url']),
                              ('stratified_by_project', {'strata': ['project_url']}, ['project_url'])):
            cph = CoxPHFitter(penalizer=0.1).fit(P[feats + ['duration', 'event'] + extra], 'duration', 'event', **kw)
            s = cph.summary
            record('cox_later_stage', variant, spec, se, s['exp(coef)'], s['exp(coef) lower 95%'],
                   s['exp(coef) upper 95%'], s['p'], len(P), int(P['event'].sum()), terms)
            if spec == 'decomposition':
                # Wald test of equal coefficients for smelly and non-smelly turnover
                b, V = cph.params_, cph.variance_matrix_
                d = b['z_smelly_turnover'] - b['z_nonsmelly_turnover']
                var = (V.loc['z_smelly_turnover', 'z_smelly_turnover'] + V.loc['z_nonsmelly_turnover', 'z_nonsmelly_turnover']
                       - 2 * V.loc['z_smelly_turnover', 'z_nonsmelly_turnover'])
                pd_ = 2 * norm.sf(abs(d) / np.sqrt(var))
                rows.append({'model': 'cox_later_stage', 'variant': variant, 'spec': spec, 'se': se,
                             'term': 'equality_test_smelly_vs_nonsmelly', 'estimate': np.exp(d),
                             'ci_low': np.nan, 'ci_high': np.nan, 'p': pd_, 'n': len(P),
                             'events': int(P['event'].sum())})

    # Early-stage logit (original metrics only make sense here: LOO metrics are built for the later stage too)
    E = early_stage_panel(df, list(dict.fromkeys(base + [turn, ratio, st, ns])))
    E['z_smelly_turnover'], E['z_nonsmelly_turnover'] = z(E[st]), z(E[ns])
    E['z_turnover'], E['z_ratio'] = z(E[turn]), z(E[ratio])
    E['z_turnover_x_ratio'] = E['z_turnover'] * E['z_ratio']
    for spec, terms in specs.items():
        X = E[[f for f in base]].fillna(0)
        X = X.loc[:, X.std() > 0]
        X = (X - X.mean()) / X.std(ddof=0)
        X = pd.concat([X, E[terms]], axis=1)
        for se, kw in (('plain', {}), ('clustered', {'cov_type': 'cluster',
                                                     'cov_kwds': {'groups': pd.factorize(E['project_url'])[0]}})):
            m = sm.GLM(E['has_failed'], sm.add_constant(X), family=sm.families.Binomial()).fit(maxiter=100, **kw)
            ci = m.conf_int()
            record('logit_early_stage', variant, spec, se, np.exp(m.params), np.exp(ci[0]), np.exp(ci[1]),
                   m.pvalues, len(E), int(E['has_failed'].sum()), terms)

R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / 'rq1_smelly_quitters_decomposition.csv', index=False)
for (model, variant, spec, se), g in R.groupby(['model', 'variant', 'spec', 'se'], sort=False):
    print(f'{model:18s} {variant:9s} {spec:14s} {se:22s} ' + ', '.join(
        f"{r.term.replace('z_', '').replace('equality_test_smelly_vs_nonsmelly', 'equal')}={r.estimate:.2f}({fmt_p(r.p)})"
        for r in g.itertuples()))
print(f'Done. Outputs in {OUT_DIR.name}')
