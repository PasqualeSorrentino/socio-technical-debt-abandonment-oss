"""
RQ1 — robustness of the technical debt measures.

In the main models, technical debt is measured as the number of distinct types of traditional
smells, ML-specific smells, and vulnerabilities found in the files touched by a developer in a
window. Since these counts grow with how much a developer works, and since Bandit's most frequent
check (B101, use of assert) is a coding habit rather than a security weakness, this script refits
the later-stage Cox model (also with standard errors clustered by project) and the early-stage
logit with three alternative operationalizations, keeping every other covariate unchanged:
  - instances:          number of instances instead of distinct types, log(1+x);
  - density:            instances per file touched, log(1+x);
  - vulnerabilities without B101: distinct vulnerability types excluding B101.
Outputs: extended/outputs/rq1_td_robustness.csv, rq1_b101_share.csv
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter

from panel import (OUT_DIR, DATA_PATH, OUTLIER_QUANTILE, load_panel, later_stage_panel,
                   early_stage_panel, check_main_model, fmt_p)

KEYS = ['project_url', 'developer_id', 'time_window_id']
TD_MAIN = ['log_traditional_smell_count', 'ml_smell_count', 'vulnerability_count']

print('Loading panel...')
df, FEAT_MODEL, _ = load_panel()
check_main_model(df, FEAT_MODEL)

raw = pd.read_csv(DATA_PATH, usecols=KEYS + ['traditional_smell_instances', 'ml_smell_instances',
                                             'vulnerability_instances', 'vulnerabilities',
                                             'files_touched_count'])
assert not raw.duplicated(KEYS).any()
for c in ['traditional_smell_instances', 'ml_smell_instances', 'vulnerability_instances', 'files_touched_count']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(0)
raw = raw.rename(columns={'files_touched_count': 'files_raw'})
raw['has_b101'] = raw['vulnerabilities'].fillna('').str.lower().str.contains(r'\bb101\b').astype(int)
df = df.merge(raw.drop(columns='vulnerabilities'), on=KEYS, how='left', validate='one_to_one')

# Share of B101 among all vulnerability instances and among observations with at least one vulnerability
_dev = pd.read_csv(DATA_PATH, usecols=['project_url', 'developer_raw_json'])
_dev = _dev[_dev['project_url'].isin(set(df['project_url']))]
_ids = [x.get('vuln_id', '').lower() for v in _dev['developer_raw_json'].dropna()
        for x in (json.loads(v).get('vulnerability_details') or [])]
_types = raw.loc[raw['project_url'].isin(set(df['project_url'])), 'vulnerabilities'].fillna('').str.lower().str.strip()
_has = _types.str.len() > 0
b101 = pd.DataFrame([{'b101_share_of_instances': np.mean([i == 'b101' for i in _ids]),
                      'b101_only_share_of_observations_with_vulnerabilities': _types[_has].eq('b101').mean(),
                      'instances': len(_ids), 'observations_with_vulnerabilities': int(_has.sum())}])
b101.to_csv(OUT_DIR / 'rq1_b101_share.csv', index=False)
print(b101.round(3).to_string(index=False))

files = df['files_raw'].where(df['files_raw'] > 0)
vuln_types = pd.to_numeric(df['vulnerability_count'], errors='coerce').fillna(0)  # already capped
new = {
    'inst_trad': df['traditional_smell_instances'], 'inst_ml': df['ml_smell_instances'],
    'inst_vuln': df['vulnerability_instances'],
    'dens_trad': df['traditional_smell_instances'] / files, 'dens_ml': df['ml_smell_instances'] / files,
    'dens_vuln': df['vulnerability_instances'] / files,
    'vuln_no_b101': (vuln_types - df['has_b101']).clip(lower=0),
}
for name, s in new.items():
    s = s.fillna(0)
    cap = s.quantile(OUTLIER_QUANTILE)
    if cap > 0:
        s = s.clip(upper=cap)
    df[name] = np.log1p(s) if name.startswith(('inst_', 'dens_')) else s

SPECS = {
    'main': TD_MAIN,
    'instances': ['inst_trad', 'inst_ml', 'inst_vuln'],
    'density_per_file': ['dens_trad', 'dens_ml', 'dens_vuln'],
    'vulnerabilities_without_b101': ['log_traditional_smell_count', 'ml_smell_count', 'vuln_no_b101'],
}
LABEL = {'log_traditional_smell_count': 'traditional', 'inst_trad': 'traditional', 'dens_trad': 'traditional',
         'ml_smell_count': 'ml', 'inst_ml': 'ml', 'dens_ml': 'ml',
         'vulnerability_count': 'vulnerabilities', 'inst_vuln': 'vulnerabilities',
         'dens_vuln': 'vulnerabilities', 'vuln_no_b101': 'vulnerabilities'}
SOCIAL = ['stqf_global_truck', 'stqf_ratio_smelly_quitters', 'stqf_core_code_turnover',
          'stqf_st_congruence', 'stqf_ratio_smelly_devs', 'stqf_ratio_sponsored']

base = [f for f in FEAT_MODEL if f not in TD_MAIN]
rows = []
for spec, td in SPECS.items():
    feats = base + td
    p = later_stage_panel(df, feats)
    feats = [f for f in feats if p[f].std() > 0]
    for clustered in (False, True):
        kw = {}
        data = p[feats + ['duration', 'event']]
        if clustered:
            data = p[feats + ['duration', 'event', 'project_url']]
            kw = {'cluster_col': 'project_url'}
        cph = CoxPHFitter(penalizer=0.1).fit(data, 'duration', 'event', **kw)
        s = cph.summary
        for c in td + SOCIAL:
            rows.append({'model': 'cox_later_stage' + ('_clustered' if clustered else ''), 'spec': spec,
                         'covariate': LABEL.get(c, c), 'variable': c, 'estimate': s.loc[c, 'exp(coef)'],
                         'ci_low': s.loc[c, 'exp(coef) lower 95%'], 'ci_high': s.loc[c, 'exp(coef) upper 95%'],
                         'p': s.loc[c, 'p']})
    E = early_stage_panel(df, base + td)
    X = E[base + td].fillna(0)
    X = X.loc[:, X.std() > 0]
    X = (X - X.mean()) / X.std(ddof=0)
    for clustered in (False, True):
        kw = {'cov_type': 'cluster', 'cov_kwds': {'groups': pd.factorize(E['project_url'])[0]}} if clustered else {}
        m = sm.GLM(E['has_failed'], sm.add_constant(X), family=sm.families.Binomial()).fit(maxiter=100, **kw)
        ci = m.conf_int()
        for c in td + SOCIAL:
            rows.append({'model': 'logit_early_stage' + ('_clustered' if clustered else ''), 'spec': spec,
                         'covariate': LABEL.get(c, c), 'variable': c, 'estimate': np.exp(m.params[c]),
                         'ci_low': np.exp(ci.loc[c, 0]), 'ci_high': np.exp(ci.loc[c, 1]), 'p': m.pvalues[c]})

R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / 'rq1_td_robustness.csv', index=False)
for (model, spec), g in R.groupby(['model', 'spec'], sort=False):
    print(f'{model:28s} {spec:30s} ' + ', '.join(
        f"{r.covariate.replace('stqf_', '')[:14]}={r.estimate:.2f}({fmt_p(r.p)})" for r in g.itertuples()))
