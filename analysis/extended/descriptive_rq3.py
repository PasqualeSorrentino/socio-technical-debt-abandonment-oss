"""
Descriptive analyses (how developers leave, how projects fade) and RQ3 (debt left behind).

  (i)   last contribution of developers who leave vs stay, core vs peripheral (codebook);
  (ii)  community smells that involve each developer in the last observed window;
  (iii) final active window of each project and explicit end-of-life notices;
  (iv)  activity trajectories (share of commits per tenth of life, k-means);
  (v)   contrasting cases (debt index);
  (vi)  RQ3: fate of the traditional-smell files of the developers who leave, and share
        of orphaned smelly files per project.
The commit codebook is a closed set of deterministic keyword rules evaluated in a fixed
order (CODEBOOK below); a stratified sample of coded messages is exported for manual
validation, together with a simple random hold-out sample. Outputs: extended/outputs/desc_*.csv and rq3_*.csv
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from panel import OUT_DIR, EARLY_STAGE_WINDOWS, load_panel, add_core_flags

EOL = r'deprecation notice|\bdeprecat\w* (this|the) (project|repo|repository|library|package)|(project|repo|repository) (is|has been) (deprecated|archived|discontinued|moved)|no longer (maintained|supported|under active development)|\bunmaintained\b|\bend[- ]of[- ]life\b'
# ── Codebook: first matching rule wins ─────────────────────────────────────
# Leading tags such as '[docs]' or 'org/repo!123 - ' are stripped before matching.
# Two categories have two rules each (an early unambiguous rule and a later general one).
CODEBOOK = [
    ('end-of-life notice', EOL),
    ('merge', r'^merged?\b(?!\s+\S+\s+(with|into)\s+(?!.*\b(branch|master|main|dev)\b))'),
    ('revert', r'^revert\b|\brevert(s|ed)?\b|\bback(ed)? out\b'),
    # unambiguous dependency and packaging changes, checked before release and documentation
    ('build and dependencies', r'^bump \S+ from\b|\bbump(s|ed)? \S+ (from|to) v?\d|\b(un)?pin(s|ned)?\b|\bpipfile\b|\bdependabot\b|\brequirements?(\.txt)?\b|\bdependenc\w*'),
    ('feature', r'^feat(\(.*?\))?!?:'),
    ('release', r'^v?\d+(\.\d+)+\b|^(version|release)\b|\b(code|model|models|data|github) release\b|(?<![-\w])releas(e|es|ed|ing) (of )?(the )?(v?\d|pre-?trained|code|models?|data)|\b(increment|update|bump(ed)?) (the )?version|version bump|\bversion numbers?\b|\bchangelog\b|^code (and data )?(for|of)\b|^open[- ]?sourc\w*'),
    ('documentation', r'\b(readme|docs?\b|documentation|docstring|tutorial|typo\w*|spelling|grammar|clarif\w*|comments?\b(?! out)|license|citation|cite\b|bibtex|markdown|\.md\b|\.rst\b|example(s)?\b|notebook|link(s)?\b|badge|paper)'),
    ('build and dependencies', r'\b(upgrade|downgrade|setup\.py|setup\.cfg|pyproject|dockerfile|docker|ci\b|workflow|github actions|travis|circleci|appveyor|tox\b|conda|environment\.ya?ml|build|install(ation)?|packaging|pypi|wheel|pip\b|makefile|pre-commit|lint\w*|flake8|black\b|isort|ruff)'),
    ('test', r'\b(test(s|ing|ed)?|unittest|pytest|coverage)\b'),
    ('bug fix', r'bug ?fix\w*|\b(fix(es|ed|ing)?|bugs?|hotfix|patch|issue|error|crash|correct(ed|ion)?|resolve(s|d)?|wrong|broken|repair|fail(s|ed|ing|ure)?)\b|\bput back\b|\bhandle \w*error'),
    ('refactoring', r'\b(refactor\w*|clean ?up|cleanup|renam\w*|restructur\w*|reorgani[sz]\w*|simplif\w*|mov(e|es|ed|ing)\b|remove (unused|unnecessary|obsolete|dead)|unused\b|tidy|reformat\w*|formatting|format with|pyfmt|code ?style|replac\w*|in favou?r of|unif(y|ies|ied)|dedup\w*|consolidat\w*)'),
    ('feature', r'\b(add(s|ed|ing)?|implement\w*|introduc\w*|support(s|ed)?|new\b|feature\w*|enable\w*|allow\w*|extend\w*|improve\w*|initial|create(s|d)?)\b'),
]
CATEGORIES = list(dict.fromkeys(c for c, _ in CODEBOOK)) + ['other']
MAINTENANCE = {'documentation', 'build and dependencies', 'bug fix'}
_RX = [(c, re.compile(p, re.I)) for c, p in CODEBOOK]
_PREFIX = re.compile(r'^(\s*\[[^\]]*\])+\s*|^[\w./-]+!\d+\s*-\s*')


def code(message):
    if not isinstance(message, str) or not message.strip():
        return 'other'
    first = _PREFIX.sub('', message.strip().splitlines()[0]).strip()
    for cat, rx in _RX:
        if rx.search(first):
            return cat
    return 'other'


def pct(series):
    return (series.value_counts(normalize=True).reindex(CATEGORIES).fillna(0) * 100).round(1)


print('Loading panel...')
df, _, project_status = load_panel(extra=True)
df = add_core_flags(df)
df['category'] = df['last_commit_message'].map(code)

# ── (i) last contribution of developers ────────────────────────────────────
last_row = df.sort_values('time_window_rank').groupby(['project_url', 'developer_id']).tail(1)
pairs = df.groupby(['project_url', 'developer_id']).agg(
    leave=('has_failed', 'max'), core=('is_core', 'max'), commits=('commits_raw', 'sum'),
    n_win=('time_window_rank', 'nunique')).reset_index()
pairs = pairs.merge(last_row[['project_url', 'developer_id', 'category', 'last_commit_message']],
                    on=['project_url', 'developer_id'])
grp = {('Core', 'Leave'): (pairs.core == 1) & (pairs.leave == 1), ('Core', 'Stay'): (pairs.core == 1) & (pairs.leave == 0),
       ('Peripheral', 'Leave'): (pairs.core == 0) & (pairs.leave == 1),
       ('Peripheral', 'Stay'): (pairs.core == 0) & (pairs.leave == 0)}
tab = pd.DataFrame({f'{a} {b}': pct(pairs[m]['category']) for (a, b), m in grp.items()})
units = {f'{a} {b}': int(m.sum()) for (a, b), m in grp.items()}
med = {f'{a} {b}': pairs[m]['commits'].median() for (a, b), m in grp.items()}

per = pairs[pairs.core == 0]
pl = per[per.leave == 1]
casual = {
    'peripheral_departures': len(pl), 'all_departures': int(pairs.leave.sum()),
    'peripheral_leaver_median_commits': pl['commits'].median(),
    'peripheral_leavers_one_commit_pct': (pl['commits'] == 1).mean() * 100,
    'peripheral_leavers_one_window_pct': (pl['n_win'] == 1).mean() * 100,
    'departure_rate_single_commit_pct': per[per.commits == 1]['leave'].mean() * 100,
    'departure_rate_repeated_pct': pairs[pairs.commits > 1]['leave'].mean() * 100,  # all developers with >1 commit
}
# core departure rate by non-core-ratio terciles among later-stage developers
late = df[df['developer_tenure'] > EARLY_STAGE_WINDOWS].groupby(['project_url', 'developer_id']).agg(
    leave=('has_failed', 'max'), core=('is_core', 'max'), noncore=('stqf_global_truck', 'mean')).reset_index()
late['tercile'] = pd.qcut(late['noncore'], 3, labels=['low', 'mid', 'high'])
lc = late[late.core == 1]
rates = lc.groupby('tercile', observed=True)['leave'].mean() * 100
chi = stats.chi2_contingency(pd.crosstab(lc['tercile'], lc['leave']))
casual.update({f'core_departure_rate_{k}_noncore_pct': v for k, v in rates.items()})
casual['core_departure_by_tercile_chi2_p'] = chi[1]
single = pl[pl.commits == 1].groupby('project_url').size().sort_values(ascending=False)
casual['top_single_commit_leaver_projects'] = '; '.join(f'{u.split("/")[-1].removesuffix(".git")}={n}'
                                                        for u, n in single.head(5).items())
pd.Series(casual).to_csv(OUT_DIR / 'desc_casual_contributors.csv', header=['value'])
print('\nCasual contributors:')
print(pd.Series(casual).to_string())

# ── (ii) community smells in the last observed window ─────────────────────
SMELLS = ['radio_silence', 'organisational_silo', 'bottleneck', 'lone_wolf', 'black_cloud']
lr = last_row.merge(pairs[['project_url', 'developer_id', 'leave']], on=['project_url', 'developer_id'])
sm_rows = []
for smell in SMELLS:
    has = lr['community_smells'].fillna('').str.contains(smell)
    sm_rows.append({'smell': smell,
                    'freq_leavers_pct': has[lr.leave == 1].mean() * 100,
                    'freq_stayers_pct': has[lr.leave == 0].mean() * 100,
                    'departure_rate_with_pct': lr[has]['leave'].mean() * 100,
                    'departure_rate_without_pct': lr[~has]['leave'].mean() * 100})
SM = pd.DataFrame(sm_rows)
SM.to_csv(OUT_DIR / 'desc_smells_last_window.csv', index=False)
print('\nCommunity smells in the last window:')
print(SM.round(1).to_string(index=False))

# ── (iii) final active window of projects ─────────────────────────────────
ps = project_status.set_index('project_url')
hist = df.groupby('project_url').agg(first=('time_window_rank', 'min'), last=('project_last_observed_rank', 'max'))
hist['history'] = hist['last'] - hist['first'] + 1
fw = df[df['time_window_rank'] == df['project_last_commit_rank']]
fwp = fw.groupby('project_url').agg(n_dev=('developer_id', 'nunique'), commits=('commits_raw', 'sum'),
                                    cats=('category', lambda s: set(s))).join(ps['proj_failed'])
fwp['maintenance_only'] = fwp['cats'].map(lambda s: s <= MAINTENANCE)
fwp['maintenance_only_incl_merge'] = fwp['cats'].map(lambda s: s <= MAINTENANCE | {'merge'})
fw = fw.merge(ps['proj_failed'].reset_index(), on='project_url', suffixes=('', '_p'))
tab['Projects Active'] = pct(fw[fw.proj_failed == 0]['category'])
tab['Projects Inactive'] = pct(fw[fw.proj_failed == 1]['category'])
units.update({'Projects Active': int((fw.proj_failed == 0).sum()), 'Projects Inactive': int((fw.proj_failed == 1).sum())})
med.update({'Projects Active': fwp[fwp.proj_failed == 0]['commits'].median(),
            'Projects Inactive': fwp[fwp.proj_failed == 1]['commits'].median()})
tab.loc['Units'] = pd.Series(units)
tab.loc['Median commits'] = pd.Series(med)
tab.to_csv(OUT_DIR / 'desc_last_contribution_categories.csv')
print('\nTable of last-contribution categories (column %):')
print(tab.to_string())

eol = df[(df['category'] == 'end-of-life notice')].merge(ps['proj_failed'].reset_index(), on='project_url', suffixes=('', '_p'))
eol_proj = sorted(eol.loc[eol.proj_failed == 1, 'project_url'].unique())
fade = {
    'history_median_dormant': hist.join(ps)['history'][ps['proj_failed'] == 1].median(),
    'history_median_active': hist.join(ps)['history'][ps['proj_failed'] == 0].median(),
    'final_window_median_devs_dormant': fwp[fwp.proj_failed == 1]['n_dev'].median(),
    'final_window_median_devs_active': fwp[fwp.proj_failed == 0]['n_dev'].median(),
    'final_window_median_commits_dormant': fwp[fwp.proj_failed == 1]['commits'].median(),
    'final_window_median_commits_active': fwp[fwp.proj_failed == 0]['commits'].median(),
    'maintenance_only_final_window_dormant_pct': fwp[fwp.proj_failed == 1]['maintenance_only'].mean() * 100,
    'maintenance_only_final_window_active_pct': fwp[fwp.proj_failed == 0]['maintenance_only'].mean() * 100,
    'maintenance_or_merge_only_dormant_pct': fwp[fwp.proj_failed == 1]['maintenance_only_incl_merge'].mean() * 100,
    'maintenance_or_merge_only_active_pct': fwp[fwp.proj_failed == 0]['maintenance_only_incl_merge'].mean() * 100,
    'dormant_projects_with_end_of_life_message': len(eol_proj),
    'end_of_life_projects': '; '.join(u.split('/')[-1] for u in eol_proj),
}
pd.Series(fade).to_csv(OUT_DIR / 'desc_project_final_window.csv', header=['value'])
print('\nHow projects fade:')
print(pd.Series(fade).to_string())

# Stratified sample of coded messages for manual validation (30 per category).
msgs = pairs[['project_url', 'developer_id', 'last_commit_message', 'category']].dropna()
sample = msgs.groupby('category', group_keys=False).apply(lambda g: g.sample(min(len(g), 30), random_state=0))
sample.to_csv(OUT_DIR / 'desc_codebook_validation_sample.csv', index=False)

# Simple random sample (not stratified), disjoint from the stratified one, to support the
# validation of the coding on the actual distribution of messages.
holdout = msgs.drop(sample.index).sample(150, random_state=1)
holdout.to_csv(OUT_DIR / 'desc_codebook_holdout_sample.csv', index=False)


# ── (iv) activity trajectories ────────────────────────────────────────────
pw = df.groupby(['project_url', 'time_window_rank'])['commits_raw'].sum().reset_index()
traj = {}
for proj, g in pw.groupby('project_url'):
    first, last = g['time_window_rank'].min(), g['time_window_rank'].max()
    L = last - first + 1
    if L < 6:
        continue
    series = np.zeros(L)
    series[(g['time_window_rank'] - first).values] = g['commits_raw'].values
    tenth = np.floor(10 * np.arange(L) / L).astype(int)
    shares = np.bincount(tenth, weights=series, minlength=10)
    traj[proj] = shares / shares.sum() if shares.sum() > 0 else shares
TR = pd.DataFrame(traj).T
X = TR.values
sil = {k: silhouette_score(X, KMeans(k, n_init=50, random_state=0).fit_predict(X)) for k in (2, 3, 4)}
km = KMeans(3, n_init=50, random_state=0).fit(X)
TR['cluster'] = km.labels_
first_fifth = TR[[0, 1]].sum(axis=1)
order = TR.assign(ff=first_fifth).groupby('cluster')['ff'].median().sort_values(ascending=False).index
names = dict(zip(order, ['initial burst', 'early peak', 'sustained']))
TR['trajectory'] = TR['cluster'].map(names)
TR['first_fifth_share'] = first_fifth

# debt index (project means, standardized)
pm = df.groupby('project_url').agg(cs=('community_smell_count', 'mean'), ts=('traditional_smell_count', 'mean'),
                                   vu=('vulnerability_count', 'mean'), sd=('stqf_ratio_smelly_devs', 'mean'),
                                   turnover=('stqf_global_turnover', 'mean'), n_dev=('developer_id', 'nunique'),
                                   n_win=('time_window_rank', 'nunique'))
zc = (pm[['cs', 'ts', 'vu', 'sd']] - pm[['cs', 'ts', 'vu', 'sd']].mean()) / pm[['cs', 'ts', 'vu', 'sd']].std()
pm['debt_index'] = zc.mean(axis=1)
labels = pd.read_csv(OUT_DIR / 'rq2_project_labels.csv').set_index('project_url')[['v1', 'v2']] \
    if (OUT_DIR / 'rq2_project_labels.csv').exists() else None
TR = TR.join(pm['debt_index']).join(ps['proj_failed'])
if labels is not None:
    TR = TR.join(labels['v1'])
tr_rows = []
for name, g in TR.groupby('trajectory'):
    tr_rows.append({'trajectory': name, 'projects': len(g), 'median_first_fifth_share_pct': g['first_fifth_share'].median() * 100,
                    'dormant_v2_pct': g['proj_failed'].mean() * 100,
                    'dormant_v1_pct': g['v1'].mean() * 100 if 'v1' in g else np.nan,
                    'median_debt_index': g['debt_index'].median()})
T3 = pd.DataFrame(tr_rows)
chi_v2 = stats.chi2_contingency(pd.crosstab(TR['trajectory'], TR['proj_failed']))[1]
kw3 = stats.kruskal(*[g['debt_index'] for _, g in TR.groupby('trajectory')]).pvalue
TR['cluster2'] = KMeans(2, n_init=50, random_state=0).fit_predict(X)
kw2 = stats.kruskal(*[g['debt_index'] for _, g in TR.groupby('cluster2')]).pvalue
T3.to_csv(OUT_DIR / 'desc_trajectories.csv', index=False)
pd.Series({'projects_with_6plus_windows': len(TR), 'silhouette_k2': sil[2], 'silhouette_k3': sil[3],
           'silhouette_k4': sil[4], 'chi2_p_trajectory_vs_v2': chi_v2, 'kruskal_p_debt_k3': kw3,
           'kruskal_p_debt_k2': kw2}).to_csv(OUT_DIR / 'desc_trajectories_tests.csv', header=['value'])
TR.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_csv(OUT_DIR / 'desc_trajectory_assignment.csv')
print('\nTrajectories:')
print(T3.round(2).to_string(index=False))
print(f'  silhouette k=2 {sil[2]:.2f}, k=3 {sil[3]:.2f}; chi2 p (v2) {chi_v2:.4f}; '
      f'Kruskal-Wallis debt index p k=3 {kw3:.2f}, k=2 {kw2:.2f}')

# ── (v) contrasting cases ─────────────────────────────────────────────────
pm = pm.join(ps['proj_failed']).join(fwp['maintenance_only'])
hi = pm[pm.proj_failed == 0].nlargest(10, 'debt_index')
lo = pm[pm.proj_failed == 1].nsmallest(10, 'debt_index')
cases = pd.concat([hi.assign(group='active, highest debt'), lo.assign(group='dormant, lowest debt')])
cases.to_csv(OUT_DIR / 'desc_contrasting_cases.csv')
cc = cases.groupby('group').agg(median_devs=('n_dev', 'median'), median_windows=('n_win', 'median'),
                                median_turnover=('turnover', 'median'), maintenance_only=('maintenance_only', 'sum'),
                                at_most_4_devs=('n_dev', lambda s: int((s <= 4).sum())))
rho = stats.spearmanr(pm['debt_index'], pm['n_dev'])
print('\nContrasting cases:')
print(cc.to_string())
print(f'  Spearman debt index vs team size: rho={rho.statistic:.2f} (p={rho.pvalue:.3f})')
cc.to_csv(OUT_DIR / 'desc_contrasting_cases_summary.csv')

# ── (vi) RQ3: debt left behind ────────────────────────────────────────────
print('\nRQ3: parsing smell locations...')


def smelly_files(raw):
    try:
        d = json.loads(raw)
    except Exception:
        return set()
    return {str(x.get('file', '')) for x in d.get('traditional_smell_details') or [] if x.get('file')}


df['files'] = df['developer_raw_json'].map(smelly_files)
occ = df[['project_url', 'developer_id', 'time_window_rank', 'files']].explode('files').dropna(subset=['files'])
occ = occ.rename(columns={'files': 'file'})
pl_last = df.groupby(['project_url', 'developer_id']).agg(last=('time_window_rank', 'max'), leave=('has_failed', 'max'),
                                                         core=('is_core', 'max')).reset_index()
leavers = pl_last[pl_last.leave == 1]
lf = occ.merge(leavers, on=['project_url', 'developer_id'])[['project_url', 'developer_id', 'file', 'last', 'core']].drop_duplicates(
    ['project_url', 'developer_id', 'file'])
others = occ[['project_url', 'developer_id', 'time_window_rank', 'file']].rename(
    columns={'developer_id': 'other', 'time_window_rank': 'w'})
m = lf.merge(others, on=['project_url', 'file'])
m = m[(m['other'] != m['developer_id']) & (m['w'] > m['last'])]
taken = m[['project_url', 'developer_id', 'file']].drop_duplicates().assign(taken=1)
lf = lf.merge(taken, on=['project_url', 'developer_id', 'file'], how='left').fillna({'taken': 0})
per_leaver = lf.groupby(['project_url', 'developer_id', 'core']).agg(files=('file', 'size'), taken=('taken', 'sum')).reset_index()
core_l = per_leaver[per_leaver.core == 1]
rq3 = {
    'smelly_files_of_leavers': len(lf),
    'taken_over_pct': lf['taken'].mean() * 100,
    'taken_over_peripheral_pct': lf[lf.core == 0]['taken'].mean() * 100,
    'taken_over_core_pct': lf[lf.core == 1]['taken'].mean() * 100,
    'leavers_with_smelly_files': len(per_leaver),
    'leavers_with_nothing_taken_over_pct': (per_leaver['taken'] == 0).mean() * 100,
    'core_leaver_median_smelly_files': core_l['files'].median(),
}
# Orphaned smelly files: share of a project's smelly files whose last contributors all left
# (all the developers who worked on the file in the last window in which it was smelly).
last_w = occ.groupby(['project_url', 'file'])['time_window_rank'].transform('max')
lastc = occ[occ['time_window_rank'] == last_w].drop_duplicates(['project_url', 'file', 'developer_id'])
lastc = lastc.merge(pl_last[['project_url', 'developer_id', 'leave']], on=['project_url', 'developer_id'])
file_orph = lastc.groupby(['project_url', 'file'])['leave'].min()
orph = file_orph.groupby('project_url').mean().rename('orphaned_share').to_frame().join(ps['proj_failed'])
a, d = orph[orph.proj_failed == 0]['orphaned_share'], orph[orph.proj_failed == 1]['orphaned_share']
rq3.update({'orphaned_share_median_dormant_pct': d.median() * 100, 'orphaned_share_median_active_pct': a.median() * 100,
            'orphaned_share_mannwhitney_p': stats.mannwhitneyu(a, d).pvalue})

# Permutation baseline: the orphaned share expected if departures were unrelated to who
# worked last on the smelly files. Within each project, the leave labels are permuted
# among its developers (the number of leavers is kept) and the orphaned share recomputed.
rng = np.random.default_rng(0)
N_PERM = 500
smelly_devs = occ.groupby('project_url')['developer_id'].apply(set).to_dict()
expected = {}
for proj, g in lastc.groupby('project_url'):
    files = g['file'].astype('category').cat.codes.to_numpy()
    out = []
    # (a) among all the developers of the project; (b) only among those with smelly files
    for pool in (pl_last[pl_last.project_url == proj],
                 pl_last[(pl_last.project_url == proj) & pl_last.developer_id.isin(smelly_devs.get(proj, ()))]):
        idx = {d: i for i, d in enumerate(pool['developer_id'])}
        labels = pool['leave'].to_numpy()
        A = np.zeros((files.max() + 1, len(labels)))
        A[files, g['developer_id'].map(idx).to_numpy()] = 1
        out.append(np.mean([((A @ (1 - rng.permutation(labels))) == 0).mean() for _ in range(N_PERM)]))
    expected[proj] = out
orph['expected_share'] = pd.Series({k: v[0] for k, v in expected.items()})
orph['expected_share_smelly_devs'] = pd.Series({k: v[1] for k, v in expected.items()})
orph['excess_smelly_devs'] = orph['orphaned_share'] - orph['expected_share_smelly_devs']
orph['excess'] = orph['orphaned_share'] - orph['expected_share']
ea, ed = orph[orph.proj_failed == 0]['excess'], orph[orph.proj_failed == 1]['excess']
rq3.update({
    'expected_share_median_dormant_pct': orph[orph.proj_failed == 1]['expected_share'].median() * 100,
    'expected_share_median_active_pct': orph[orph.proj_failed == 0]['expected_share'].median() * 100,
    'excess_median_dormant_pp': ed.median() * 100, 'excess_median_active_pp': ea.median() * 100,
    'excess_dormant_wilcoxon_p': stats.wilcoxon(ed).pvalue, 'excess_active_wilcoxon_p': stats.wilcoxon(ea).pvalue,
    'excess_dormant_vs_active_mannwhitney_p': stats.mannwhitneyu(ea, ed).pvalue,
    'excess_positive_dormant_pct': (ed > 0).mean() * 100, 'excess_positive_active_pct': (ea > 0).mean() * 100,
    'projects_dormant': len(ed), 'projects_active': len(ea),
    'expected_smelly_devs_median_dormant_pct': orph[orph.proj_failed == 1]['expected_share_smelly_devs'].median() * 100,
    'expected_smelly_devs_median_active_pct': orph[orph.proj_failed == 0]['expected_share_smelly_devs'].median() * 100,
    'excess_smelly_devs_median_dormant_pp': orph[orph.proj_failed == 1]['excess_smelly_devs'].median() * 100,
    'excess_smelly_devs_median_active_pp': orph[orph.proj_failed == 0]['excess_smelly_devs'].median() * 100,
    'excess_smelly_devs_dormant_wilcoxon_p': stats.wilcoxon(orph[orph.proj_failed == 1]['excess_smelly_devs']).pvalue,
    'excess_smelly_devs_active_wilcoxon_p': stats.wilcoxon(orph[orph.proj_failed == 0]['excess_smelly_devs']).pvalue,
})
pd.Series(rq3).to_csv(OUT_DIR / 'rq3_debt_left_behind.csv', header=['value'])
orph.to_csv(OUT_DIR / 'rq3_orphaned_share_by_project.csv')
print(pd.Series(rq3).round(3).to_string())
print('\nDone. Outputs in', OUT_DIR)
