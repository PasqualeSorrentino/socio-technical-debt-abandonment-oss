"""
Overlap between community smells and the descriptive structure of the community.

Community smells (csDetector) are detected on the same collaboration and communication
network from which the structural metrics are computed. This script quantifies the overlap
at the level of project windows: for each window, the number of instances of each of the five
retained smells (raw and per developer) is correlated (Spearman) with the structural metrics
returned by csDetector. It also reports, for each smell, the detection rule as stated in the
evidence messages of the tool.

Outputs: extended/outputs/smell_structure_correlations.csv, smell_detection_rules.csv
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from panel import OUT_DIR, DATA_PATH, included_projects

SMELLS = ['organisational_silo', 'lone_wolf', 'bottleneck', 'radio_silence', 'black_cloud']
STRUCTURE = ['st.congruence', 'communicability', 'global.truck', 'density', 'closeness.centr',
             'betweenness.centr', 'degree.centr', 'global.mod', 'num.tz', 'devs']


def parse(v):
    try:
        d = json.loads(v)
    except (TypeError, ValueError):
        return None
    return d


cols = ['project_url', 'time_window_id', 'project_community_smells_count_json',
        'project_community_smell_instances_json', 'Socio-Technical Quality Factors']
df = pd.read_csv(DATA_PATH, usecols=cols)
df = df[df['project_url'].isin(included_projects())].drop_duplicates(['project_url', 'time_window_id'])
print(f'Project windows: {len(df):,}')

# Detection rule of each smell, as stated by the tool (first message, frequency of evidence modes)
rules, modes = {}, {s: Counter() for s in SMELLS}
for v in df['project_community_smell_instances_json'].dropna():
    for inst in parse(v) or []:
        sid = inst.get('smell_id')
        if sid not in modes:
            continue
        modes[sid][(inst.get('evidence') or {}).get('mode')] += 1
        rules.setdefault(sid, re.sub(r'\S+@\S+', '<developer>', inst.get('message') or ''))
pd.DataFrame([{'smell': s, 'instances': sum(modes[s].values()), 'example_message': rules.get(s),
               'evidence_modes': dict(modes[s])} for s in SMELLS]) \
    .to_csv(OUT_DIR / 'smell_detection_rules.csv', index=False)

counts = pd.DataFrame([parse(v) or {} for v in df['project_community_smells_count_json']], index=df.index)
counts = counts.reindex(columns=SMELLS).fillna(0)
stqf = pd.DataFrame([parse(v) or {} for v in df['Socio-Technical Quality Factors']], index=df.index)
stqf = stqf.reindex(columns=STRUCTURE).apply(pd.to_numeric, errors='coerce')
devs = stqf['devs'].replace(0, np.nan)

rows = []
for s in SMELLS + ['all_smells']:
    raw = counts.sum(axis=1) if s == 'all_smells' else counts[s]
    for norm, x in (('raw', raw), ('per_developer', raw / devs)):
        for m in STRUCTURE:
            if norm == 'per_developer' and m == 'devs':
                continue
            pair = pd.concat([x, stqf[m]], axis=1).dropna()
            rows.append({'smell': s, 'normalization': norm, 'metric': m,
                         'spearman_rho': pair.iloc[:, 0].corr(pair.iloc[:, 1], method='spearman'),
                         'n_windows': len(pair)})
R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / 'smell_structure_correlations.csv', index=False)
print(R.pivot_table(index=['smell', 'normalization'], columns='metric', values='spearman_rho').round(2).to_string())
