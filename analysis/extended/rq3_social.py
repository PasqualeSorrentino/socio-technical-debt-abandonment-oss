"""
RQ3 (social side) — what the departure of a core developer leaves behind in the community.

Event study on the project-window panel. The event is the first departure of a core
developer in a project, dated at the leaver's last active window (event time k = 0).
For each project-window we compute social debt outcomes and estimate

    y_pt = sum_k beta_k * 1[t - t0_p = k] + project FE + calendar-quarter FE + e_pt

with k binned in [-4, +4] (endpoints pooled), k = -1 as reference, never-treated projects
as controls, and standard errors clustered by project. The pre-event coefficients test
for pre-trends; the average of k = +1..+3 is the post-departure change.

The same design is re-estimated controlling for the number of active developers (to separate
the shrinking of the team from other changes), and applied, as a comparison, to the first
departure of a repeated peripheral contributor (at least two active windows) in projects
with no core departure. Benjamini-Hochberg q-values are computed over the social outcomes.

Outputs: extended/outputs/rq3_social_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from panel import OUT_DIR, load_panel, add_core_flags

K_MIN, K_MAX, REF = -4, 4, -1
SMELLS = ['organisational_silo', 'lone_wolf', 'radio_silence', 'bottleneck']

print('Loading panel...')
df, _, _ = load_panel(extra=True)
df = add_core_flags(df)
df['community_smells'] = df['community_smells'].fillna('').astype(str)
for s in SMELLS:
    df['has_' + s] = df['community_smells'].str.contains(s).astype(int)
df['has_any_smell'] = (df['community_smell_raw'] > 0).astype(int)

# ── Project-window outcomes ────────────────────────────────────────────────
OUTCOMES = {
    'share_devs_any_smell': ('has_any_smell', 'mean'),
    **{f'share_devs_{s}': (f'has_{s}', 'mean') for s in SMELLS},
    'community_smells_per_dev': ('community_smell_raw', 'mean'),
    'ratio_smelly_devs': ('stqf_ratio_smelly_devs', 'first'),
    'st_congruence': ('stqf_st_congruence', 'first'),
    'communicability': ('stqf_communicability', 'first'),
    'non_core_ratio': ('stqf_global_truck', 'first'),
    'network_density': ('stqf_density', 'first'),
    'network_modularity': ('stqf_global_mod', 'first'),
    'log_active_devs': ('developer_id', 'nunique'),
}
pw = df.groupby(['project_url', 'time_window_rank']).agg(**OUTCOMES).reset_index()
pw['log_active_devs'] = np.log1p(pw['log_active_devs'])
OUT_NAMES = [c for c in OUTCOMES]

dev = df.groupby(['project_url', 'developer_id']).agg(
    core=('is_core', 'max'), leave=('has_failed', 'max'), last=('time_window_rank', 'max'),
    n_win=('time_window_rank', 'nunique')).reset_index()


def event_dates(mask):
    return dev[mask].groupby('project_url')['last'].min()


def bin_k(k):
    return int(min(max(k, K_MIN), K_MAX))


def event_study(t0, controls, label, control_size=False):
    data = pw[pw.project_url.isin(t0.index) | pw.project_url.isin(controls)].copy()
    data['k'] = data['time_window_rank'] - data['project_url'].map(t0)
    data['treated'] = data['project_url'].isin(t0.index)
    terms = []
    for k in range(K_MIN, K_MAX + 1):
        if k == REF:
            continue
        name = f'k_m{-k}' if k < 0 else f'k_p{k}'
        data[name] = (data['treated'] & (data['k'].fillna(0).map(bin_k) == k)).astype(int)
        terms.append((k, name))
    rhs = ' + '.join(n for _, n in terms) + ' + C(project_url) + C(time_window_rank)'
    rows = []
    for y in OUT_NAMES:
        if control_size and y == 'log_active_devs':
            continue
        rhs_y = rhs + (' + log_active_devs' if control_size else '')
        d = data.dropna(subset=[y])
        fit = smf.ols(f'{y} ~ {rhs_y}', data=d).fit(cov_type='cluster', cov_kwds={'groups': d['project_url']})
        post = [n for k, n in terms if 1 <= k <= 3]
        pre = [n for k, n in terms if k in (-3, -2)]
        avg = fit.t_test('(' + ' + '.join(post) + f')/{len(post)} = 0')
        pre_test = fit.f_test(', '.join(f'{n} = 0' for n in pre))
        sd = d.loc[~d['treated'] | (d['k'] < 0), y].std()
        rows.append({
            'design': label, 'outcome': y, 'n_project_windows': int(fit.nobs),
            'treated_projects': int(t0.size), 'control_projects': int(len(controls)),
            'post_avg': float(np.ravel(avg.effect)[0]),
            'post_ci_low': float(avg.conf_int()[0][0]), 'post_ci_high': float(avg.conf_int()[0][1]),
            'post_p': float(avg.pvalue), 'post_avg_in_sd': float(np.ravel(avg.effect)[0]) / sd if sd > 0 else np.nan,
            'pretrend_p': float(pre_test.pvalue),
            **{f'beta_{n}': float(fit.params[n]) for _, n in terms},
        })
    R = pd.DataFrame(rows)
    social = R['outcome'] != 'log_active_devs'
    R.loc[social, 'post_q_bh'] = multipletests(R.loc[social, 'post_p'], method='fdr_bh')[1]
    return R


core_t0 = event_dates((dev.core == 1) & (dev.leave == 1))
never_core = sorted(set(pw.project_url) - set(core_t0.index))
per_t0 = event_dates((dev.core == 0) & (dev.leave == 1) & (dev.n_win >= 2))
per_t0 = per_t0[per_t0.index.isin(never_core)]
never_per = sorted(set(never_core) - set(per_t0.index))

# Projects that are still observed after the core departure (survivorship of the post period)
obs_after = pw.merge(core_t0.rename('t0'), left_on='project_url', right_index=True)
still = obs_after[obs_after.time_window_rank > obs_after.t0].project_url.nunique()
print(f'Core departures: {core_t0.size} projects (first event), {still} still observed afterwards; '
      f'{len(never_core)} never-treated projects')
print(f'Comparison (repeated peripheral): {per_t0.size} projects, {len(never_per)} controls')

R = pd.concat([event_study(core_t0, never_core, 'core_departure'),
               event_study(core_t0, never_core, 'core_departure_size_adjusted', control_size=True),
               event_study(per_t0, never_per, 'repeated_peripheral_departure')])
R.to_csv(OUT_DIR / 'rq3_social_event_study.csv', index=False)
pd.Series({'core_event_projects': core_t0.size, 'core_event_projects_observed_after': still,
           'never_treated_projects': len(never_core), 'placebo_projects': per_t0.size}).to_csv(
    OUT_DIR / 'rq3_social_counts.csv', header=['value'])
cols = ['design', 'outcome', 'post_avg', 'post_ci_low', 'post_ci_high', 'post_p', 'post_q_bh', 'post_avg_in_sd', 'pretrend_p']
print(R[cols].round(3).to_string(index=False))
print('\nDone. Outputs in', OUT_DIR)
