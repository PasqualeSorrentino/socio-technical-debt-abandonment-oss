"""
Context selection — descriptive statistics of the 308 projects in the sample.

Combines the metadata that NICHE reports for each project (GitHub stars, commits and lines of code
at the time the dataset was built) with measures computed on the mined panel (contributors, mined
commits, length of the observed history, lines of code in the last observed window), to show how
heterogeneous the sample is.
Inputs:  dataset/mining.csv, dataset/NICHE.csv
Outputs: extended/outputs/context_descriptives.csv, extended/outputs/context_descriptives_by_project.csv
"""
import pandas as pd

from panel import DATA_PATH, OUT_DIR, apply_inclusion_criteria, quarter_index


def repo_key(url):
    return url.replace('https://github.com/', '').removesuffix('.git').strip('/').lower()


cols = ['project_url', 'developer_id', 'commits_count', 'time_window_id', 'project_loc']
df = apply_inclusion_criteria(pd.read_csv(DATA_PATH, usecols=cols, low_memory=False))
df['rank'] = df['time_window_id'].astype(str).map(quarter_index)
df['commits'] = pd.to_numeric(df['commits_count'], errors='coerce').fillna(0)
df['loc'] = pd.to_numeric(df['project_loc'], errors='coerce')

last_loc = (df.sort_values('rank').dropna(subset=['loc'])
            .groupby('project_url')['loc'].last().rename('mined_loc'))
P = df.groupby('project_url').agg(contributors=('developer_id', 'nunique'),
                                   mined_commits=('commits', 'sum'),
                                   first=('rank', 'min'), last=('rank', 'max')).join(last_loc)
P['history_years'] = (P['last'] - P['first'] + 1) / 4
P = P.drop(columns=['first', 'last']).reset_index()
P['repo'] = P['project_url'].map(repo_key)

niche = pd.read_csv(DATA_PATH.parent / 'NICHE.csv', encoding='utf-8-sig')
niche['repo'] = niche['GitHubRepo'].str.strip().str.lower()
niche = niche.rename(columns={'Stars': 'stars', 'Commits': 'niche_commits', 'Lines of Code': 'niche_loc'})
P = P.merge(niche[['repo', 'stars', 'niche_commits', 'niche_loc']], on='repo', how='left', validate='one_to_one')
missing = P['stars'].isna().sum()
print(f'{len(P)} projects; {len(P) - missing} matched to NICHE metadata, {missing} unmatched')

MEASURES = {'stars': 'GitHub stars (NICHE)', 'niche_commits': 'Commits (NICHE)',
            'niche_loc': 'Lines of code (NICHE)', 'contributors': 'Contributors (mined)',
            'mined_commits': 'Commits (mined)', 'history_years': 'Observed history in years (mined)'}
rows = []
for c, label in MEASURES.items():
    s = P[c].dropna()
    rows.append({'measure': label, 'n': len(s), 'min': s.min(), 'q1': s.quantile(.25), 'median': s.median(),
                 'mean': s.mean(), 'q3': s.quantile(.75), 'max': s.max(), 'sd': s.std()})
R = pd.DataFrame(rows)
print(R.round(1).to_string(index=False))
R.to_csv(OUT_DIR / 'context_descriptives.csv', index=False)
P.drop(columns='repo').to_csv(OUT_DIR / 'context_descriptives_by_project.csv', index=False)
print(f'Done. Outputs in {OUT_DIR.name}')
