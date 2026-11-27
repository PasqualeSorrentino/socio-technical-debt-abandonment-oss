"""
Shared data preparation for the extended analyses.

Reproduces, step by step, STEP 1-6 of ``v2-maintenance-based/pipeline_v2_maintenance_based.py``
(loading, STQF extraction, calendar-quarter windows, developer and project outcomes,
preprocessing, VIF selection), so that every extended analysis starts from exactly the
same panel as the main models of the paper. ``check_main_model`` re-estimates the
later-stage Cox model and verifies that it matches Table 2 before anything else runs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
# Share of a project's commits covered by its core developers (main analysis: 0.50).
# Set CORE_THRESHOLD=0.8 to run the sensitivity analysis; its outputs go to outputs_core80/.
CORE_THRESHOLD = float(os.environ.get('CORE_THRESHOLD', '0.5'))
# Inactivity threshold (in windows) used to label the departure of CORE developers only
# (main analysis: 4, as for everybody else). Set CORE_DEPARTURE_WINDOWS=8 to run the
# sensitivity analysis on the departure threshold of core developers, whose four-window
# absences end with a return far more often; its outputs go to outputs_coredep8/.
CORE_DEPARTURE_WINDOWS = int(os.environ.get('CORE_DEPARTURE_WINDOWS', '4'))
_suffix = ('' if CORE_THRESHOLD == 0.5 else f'_core{int(round(CORE_THRESHOLD * 100))}') + \
          ('' if CORE_DEPARTURE_WINDOWS == 4 else f'_coredep{CORE_DEPARTURE_WINDOWS}')
OUT_DIR = BASE_DIR / ('outputs' + _suffix)
SENSITIVITY_RUN = bool(_suffix)
OUT_DIR.mkdir(exist_ok=True, parents=True)


def locate_dataset():
    for parent in [BASE_DIR, *BASE_DIR.parents]:
        candidate = parent / 'dataset' / 'mining.csv'
        if candidate.exists():
            return candidate
    raise FileNotFoundError('dataset/mining.csv not found')


DATA_PATH = locate_dataset()

# Inclusion criteria (Section 3.1 of the paper):
#  (i)  the history must be reconstructed, i.e. the mined history must contain at least the
#       100 commits that NICHE requires of every project;
#  (ii) the project must involve more than one contributor, as required by the network metrics.
MIN_MINED_COMMITS = 100
MIN_CONTRIBUTORS = 2


def apply_inclusion_criteria(df):
    """Keep only the projects that meet the two inclusion criteria."""
    commits = pd.to_numeric(df['commits_count'], errors='coerce').fillna(0)
    g = df.assign(_c=commits).groupby('project_url').agg(_commits=('_c', 'sum'),
                                                         _devs=('developer_id', 'nunique'))
    keep = g.index[(g['_commits'] >= MIN_MINED_COMMITS) & (g['_devs'] >= MIN_CONTRIBUTORS)]
    print(f"Inclusion criteria: {len(keep)} of {len(g)} projects retained "
          f"({(g['_commits'] < MIN_MINED_COMMITS).sum()} with fewer than {MIN_MINED_COMMITS} mined commits, "
          f"{(g['_devs'] < MIN_CONTRIBUTORS).sum()} with a single contributor)")
    return df[df['project_url'].isin(keep)].reset_index(drop=True)


def included_projects():
    """Set of the projects that meet the inclusion criteria."""
    d = pd.read_csv(DATA_PATH, usecols=['project_url', 'developer_id', 'commits_count'])
    return set(apply_inclusion_criteria(d)['project_url'])

INACTIVITY_THRESHOLD = 4
EARLY_STAGE_WINDOWS = 3
PROJECT_ACTIVITY_WINDOWS = 4
PROJECT_COMMIT_THRESHOLD = 12
VIF_THRESHOLD = 5.0
OUTLIER_QUANTILE = 0.99

REQUIRED_COLUMNS = [
    'project_url', 'time_window_id', 'developer_id',
    'last_interaction_window_id', 'commits_count', 'code_churn',
    'files_touched_count',
    'community_smell_count', 'traditional_smell_count', 'ml_smell_count',
    'vulnerability_count', 'vulnerability_high', 'project_loc',
    'Socio-Technical Quality Factors', 'developer_raw_json'
]
# Extra raw columns used only by the descriptive analyses; never used as predictors.
EXTRA_COLUMNS = ['aliases', 'emails', 'community_smells', 'last_commit_message',
                 'last_commit_date', 'last_commit_hash']

STQF_KEYS = [
    'devs', 'ml.only.devs', 'code.only.devs',
    'ratio.smelly.devs', 'global.turnover', 'core.code.turnover',
    'global.truck', 'density', 'st.congruence', 'communicability',
    'num.tz', 'ratio.smelly.quitters', 'ratio.sponsored',
    'closeness.centr', 'betweenness.centr', 'degree.centr',
    'global.mod', 'code.mod', 'core.global.devs', 'core.code.devs'
]

FEATURES_BASE = [
    'community_smell_count', 'stqf_ratio_smelly_devs', 'stqf_global_turnover',
    'stqf_core_code_turnover', 'stqf_global_truck', 'stqf_density',
    'stqf_st_congruence', 'stqf_communicability', 'stqf_num_tz',
    'stqf_ratio_smelly_quitters', 'stqf_ratio_sponsored', 'stqf_devs',
    'stqf_closeness_centr', 'stqf_degree_centr', 'stqf_core_global_devs',
    'stqf_core_code_devs',
    'traditional_smell_count', 'ml_smell_count', 'vulnerability_count',
    'vulnerability_high',
    'commits_count', 'code_churn', 'files_touched_count',
    'project_age', 'developer_tenure',
]
LOG_COLS = ['commits_count', 'code_churn', 'traditional_smell_count',
            'community_smell_count', 'files_touched_count']

# Automation accounts, matched on the developer identity (e-mail) only: GitHub app
# accounts ([bot]), identities with a standalone 'bot' token, and known automation
# services whose name embeds 'bot' (dpebot, CodemodService Bot, Pyre Bot Jr).
# Matching on display names would also catch humans (e.g. 'Travis ...', 'I-am-Bot').
# Result: 28 accounts, 147 project-developer pairs.
BOT_PATTERN = (r'\[bot\]|(?:^|[^a-z])bot(?:$|[^a-z])|dependabot|renovate|pre-commit-ci|imgbot'
               r'|dpebot|codemodservicebot|pyrebotjr')


def safe_json(s):
    try:
        if pd.isna(s) or str(s).strip() in ('', '{}'):
            return {}
        return json.loads(s)
    except Exception:
        return {}


def quarter_index(window_id):
    start = str(window_id)[:8]
    return (int(start[:4]) * 12 + int(start[4:6]) - 1) // 3


def load_panel(extra=False):
    """Return (df, FEAT_MODEL, project_status) exactly as in the v2 pipeline."""
    cols = REQUIRED_COLUMNS + (EXTRA_COLUMNS if extra else [])
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    df = apply_inclusion_criteria(df)

    stqf = df['Socio-Technical Quality Factors'].apply(safe_json)
    stqf_df = pd.DataFrame([{k: d.get(k, np.nan) for k in STQF_KEYS} for d in stqf])
    stqf_df.columns = ['stqf_' + c.replace('.', '_') for c in stqf_df.columns]
    df = pd.concat([df.reset_index(drop=True), stqf_df.reset_index(drop=True)], axis=1)

    ids = pd.concat([df['time_window_id'], df['last_interaction_window_id']]).dropna().astype(str).unique()
    rank = {w: quarter_index(w) for w in ids}
    global_last = max(rank[w] for w in df['time_window_id'].astype(str).unique())
    df['time_window_rank'] = df['time_window_id'].astype(str).map(rank)
    df['last_interaction_rank'] = df['last_interaction_window_id'].astype(str).map(rank)

    proj_last_obs = df.groupby('project_url')['time_window_rank'].transform('max')
    dev_first = df.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('min')
    dev_last = df.groupby(['project_url', 'developer_id'])['last_interaction_rank'].transform('max')
    df['inactivity_lag'] = proj_last_obs - dev_last
    df['has_failed'] = (df['inactivity_lag'] >= INACTIVITY_THRESHOLD).astype(int)
    df['obs_duration'] = np.where(df['has_failed'] == 1, dev_last - dev_first,
                                  proj_last_obs - dev_first) + 1
    df['project_last_observed_rank'] = proj_last_obs
    df['global_last_rank'] = global_last

    df['_commits_raw'] = pd.to_numeric(df['commits_count'], errors='coerce').fillna(0)
    pwc = df.groupby(['project_url', 'time_window_rank'], as_index=False).agg(
        project_commits=('_commits_raw', 'sum'))
    rows = []
    for proj, g in pwc.groupby('project_url'):
        pos = g[g['project_commits'] > 0]
        last = int(pos['time_window_rank'].max()) if len(pos) else int(g['time_window_rank'].max())
        c12 = float(g.loc[g['time_window_rank'].between(last - PROJECT_ACTIVITY_WINDOWS + 1, last),
                          'project_commits'].sum()) if len(pos) else 0.0
        rows.append({'project_url': proj, 'project_last_commit_rank': last,
                     'project_commits_last_12m': c12,
                     'proj_failed': int(c12 < PROJECT_COMMIT_THRESHOLD)})
    project_status = pd.DataFrame(rows)
    df = df.merge(project_status, on='project_url', how='left', validate='many_to_one')
    df = df.drop(columns='_commits_raw')

    df = df.sort_values(['project_url', 'developer_id', 'time_window_rank']).reset_index(drop=True)
    pf = df.groupby('project_url')['time_window_rank'].min().rename('proj_first_rank')
    dfr = df.groupby(['project_url', 'developer_id'])['time_window_rank'].min().rename('dev_first_rank')
    df = df.merge(pf, on='project_url', how='left').merge(dfr, on=['project_url', 'developer_id'], how='left')
    df['project_age'] = df['time_window_rank'] - df['proj_first_rank']
    df['developer_tenure'] = df['time_window_rank'] - df['dev_first_rank']

    # Raw (uncapped) copies used by descriptive analyses.
    df['commits_raw'] = pd.to_numeric(df['commits_count'], errors='coerce').fillna(0)
    df['community_smell_raw'] = pd.to_numeric(df['community_smell_count'], errors='coerce').fillna(0)

    for f in FEATURES_BASE:
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    for col in FEATURES_BASE:
        cap = df[col].quantile(OUTLIER_QUANTILE)
        if cap > 0:
            df[col] = df[col].clip(upper=cap)
    for c in LOG_COLS:
        df[f'log_{c}'] = np.log1p(df[c])
    feat_final = [f'log_{c}' for c in LOG_COLS] + [f for f in FEATURES_BASE if f not in LOG_COLS]

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    X = df[feat_final].fillna(0)
    X = X.loc[:, (X.std() > 0) & ~X.isin([np.inf, -np.inf]).any()]
    Xc = add_constant(X)
    vif = pd.DataFrame({'feature': X.columns,
                        'VIF': [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]})
    high = vif[vif['VIF'] > VIF_THRESHOLD]['feature'].tolist()
    feat_model = [f for f in feat_final if f not in high]
    return df, feat_model, project_status


def add_core_flags(df):
    """Commit-based core: smallest set of developers covering >= CORE_THRESHOLD of the project's commits."""
    dp = df.groupby(['project_url', 'developer_id']).agg(commits=('commits_raw', 'sum')).reset_index()
    flags = []
    for proj, g in dp.groupby('project_url'):
        g = g.sort_values('commits', ascending=False)
        tot = g['commits'].sum()
        g['is_core'] = 0
        if tot > 0:
            cut = (g['commits'].cumsum() / tot >= CORE_THRESHOLD).values.argmax()
            g.iloc[:cut + 1, g.columns.get_loc('is_core')] = 1
        flags.append(g[['project_url', 'developer_id', 'is_core']])
    df = df.merge(pd.concat(flags), on=['project_url', 'developer_id'], how='left')
    if CORE_DEPARTURE_WINDOWS != INACTIVITY_THRESHOLD:
        # Relabel the departure of core developers with their own threshold; the
        # duration of a core developer who is no longer labeled as departed runs to
        # the horizon of the project (right-censoring), exactly as for everybody else.
        core = df['is_core'] == 1
        dev_first = df.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('min')
        dev_last = df.groupby(['project_url', 'developer_id'])['last_interaction_rank'].transform('max')
        failed = (df['inactivity_lag'] >= CORE_DEPARTURE_WINDOWS).astype(int)
        dur = np.where(failed == 1, dev_last - dev_first, df['project_last_observed_rank'] - dev_first) + 1
        df.loc[core, 'has_failed'] = failed[core]
        df.loc[core, 'obs_duration'] = dur[core]
    return df


def add_loo_metrics(df):
    """Lagged leave-one-out turnover, core turnover and ratio of smelly quitters (needs add_core_flags).

    For a developer i active in window t: share of the other developers active in t-1 who
    left the project for good after t-1 (restricted to core developers for core turnover),
    and share of those leavers affected by at least one community smell in their last window.
    """
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
    den = df['n'] - df['_i_in_prev']
    den_core = df['n_core'] - df['_i_in_prev'] * df['is_core']
    df['loo_turnover'] = np.where(den > 0, df['lv'] / den.where(den > 0, 1), 0.0)
    df['loo_core_turnover'] = np.where(den_core > 0, df['lv_core'] / den_core.where(den_core > 0, 1), 0.0)
    df['loo_smelly_quitters'] = np.where(df['lv'] > 0, df['slv'] / df['lv'].where(df['lv'] > 0, 1), 0.0)
    return df


def add_bot_flags(df):
    df['is_bot'] = df['developer_id'].astype(str).str.lower().str.contains(BOT_PATTERN, regex=True).astype(int)
    return df


def later_stage_panel(df, features, extra_cols=()):
    """One row per project-developer pair with tenure > 3 windows (STEP 8 of the pipeline)."""
    late = df[df['developer_tenure'] > EARLY_STAGE_WINDOWS]
    agg = {f: 'mean' for f in features}
    agg.update({'obs_duration': 'max', 'has_failed': 'max'})
    for c in extra_cols:
        agg[c] = 'max'
    p = late.groupby(['project_url', 'developer_id']).agg(agg).reset_index()
    p = p.rename(columns={'obs_duration': 'duration', 'has_failed': 'event'})
    p = p[p['duration'] > 0].replace([np.inf, -np.inf], np.nan).fillna(0)
    return p


def early_stage_panel(df, features, extra_cols=()):
    early = df[df['developer_tenure'] <= EARLY_STAGE_WINDOWS]
    agg = {f: 'mean' for f in features}
    agg['has_failed'] = 'max'
    for c in extra_cols:
        agg[c] = 'max'
    return early.groupby(['project_url', 'developer_id'], as_index=False).agg(agg)


def check_main_model(df, feat_model):
    """Re-estimate Table 2 and stop if it does not match the paper."""
    from lifelines import CoxPHFitter
    p = later_stage_panel(df, feat_model)
    feats = [f for f in feat_model if p[f].std() > 0]
    if SENSITIVITY_RUN and CORE_DEPARTURE_WINDOWS != INACTIVITY_THRESHOLD:
        return feats  # the core relabeling changes the outcome, so Table 2 is not expected
    cph = CoxPHFitter(penalizer=0.1).fit(p[feats + ['duration', 'event']], 'duration', 'event')
    hr = cph.summary['exp(coef)']
    expected = {'stqf_global_truck': 2.42, 'stqf_ratio_smelly_quitters': 1.87,
                'stqf_ratio_sponsored': 0.67, 'developer_tenure': 0.74}
    for k, v in expected.items():
        assert abs(hr[k] - v) < 0.006, f'{k}: {hr[k]:.3f} != {v} (Table 2)'
    assert len(p) == 1905 and int(p['event'].sum()) == 1103
    return feats


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'
