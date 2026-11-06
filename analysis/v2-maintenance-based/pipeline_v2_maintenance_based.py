"""
Socio-Technical Debt e inattività dei progetti OSS
===================================================

La pipeline distingue due outcome differenti.

RQ1 — Abbandono del developer
    has_failed = 1: il developer interrompe l'attività almeno quattro
    finestre trimestrali prima dell'orizzonte osservabile del proprio progetto.
    has_failed = 0: osservazione censurata a destra.

RQ2 — Inattività del progetto, definizione commit-based
    proj_failed = 1: nelle quattro finestre trimestrali terminanti con la
    finestra del commit più recente il progetto ha prodotto meno di 12 commit.
    proj_failed = 0: nello stesso periodo ha prodotto almeno 12 commit.

La definizione di progetto è applicata letteralmente, senza richiedere un gap
successivo all'ultimo commit, una storia minima o ulteriori condizioni.

Output:
    - tabelle CSV;
    - grafici PNG;
    - risultati statistici nel terminale.

Esecuzione di riferimento inserita nei commenti:
    dataset di 24.916 righe, 332 progetti, 10.416 developer e 154 finestre.
"""

# Limitare i thread BLAS evita rallentamenti marcati su alcune installazioni
# di NumPy/statsmodels. Le variabili sono impostate prima degli import numerici.
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy import stats as _ss
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from lifelines import CoxTimeVaryingFitter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def _locate_dataset():
    """Risale l'albero finché non trova dataset/mining.csv (robusto ai rinomini)."""
    for parent in [BASE_DIR, *BASE_DIR.parents]:
        candidate = parent / 'dataset' / 'mining.csv'
        if candidate.exists():
            return candidate
    for fallback in (BASE_DIR / 'mining.csv', BASE_DIR / 'mining (1).csv'):
        if fallback.exists():
            return fallback
    return BASE_DIR / 'dataset' / 'mining.csv'

DATA_PATH = _locate_dataset()
OUT_DIR = BASE_DIR / 'outputs' / 'file'
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Outcome developer
INACTIVITY_THRESHOLD = 4       # quattro trimestri di inattività
EARLY_STAGE_WINDOWS = 3        # prime tre finestre del developer

# Outcome progetto: definizione richiesta
PROJECT_ACTIVITY_WINDOWS = 4   # quattro trimestri = dodici mesi
PROJECT_COMMIT_THRESHOLD = 12  # meno di 12 commit = meno di un commit al mese

# Preprocessing
VIF_THRESHOLD = 5.0
OUTLIER_QUANTILE = 0.99


def cliffs_delta(a, b):
    """Effect size non parametrico compreso tra -1 e +1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a) * len(b)
    if n == 0:
        return np.nan
    return (
        sum(np.sum(value > b) for value in a)
        - sum(np.sum(value < b) for value in a)
    ) / n


def eff_label(delta):
    """Classificazione convenzionale dell'ampiezza di Cliff's delta."""
    absolute = abs(delta)
    if absolute < 0.147:
        return 'trascurabile'
    if absolute < 0.33:
        return 'piccolo'
    if absolute < 0.474:
        return 'medio'
    return 'grande'


def gini(values):
    """Indice di Gini per la concentrazione dei contributi."""
    values = np.sort(np.asarray(values, float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return np.nan
    return (
        2 * np.sum(np.arange(1, n + 1) * values) / (n * values.sum())
        - (n + 1) / n
    )


def youden(y, x):
    """Soglia che massimizza l'indice di Youden e relativa AUC."""
    fpr, tpr, thresholds = roc_curve(y, x)
    index = (tpr - fpr).argmax()
    return thresholds[index], auc(fpr, tpr), tpr[index], 1 - fpr[index]


EMO = ['Anger', 'Fear', 'Sadness', 'Love', 'Joy', 'Surprise']


# ─────────────────────────────────────────────
# STEP 1 — CARICAMENTO
# ─────────────────────────────────────────────

print("STEP 1 — Caricamento dati")


REQUIRED_COLUMNS = [
    'project_url', 'time_window_id', 'developer_id',
    'last_interaction_window_id', 'commits_count', 'code_churn',
    'bug_fix_commits_count', 'files_touched_count',
    'community_smell_count', 'traditional_smell_count', 'ml_smell_count',
    'vulnerability_count', 'vulnerability_high', 'bug_introduced_count_rszz',
    'se_score', 'sentiment_score', 'project_loc',
    'Socio-Technical Quality Factors', 'developer_raw_json'
]
_source_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
_missing_required = [c for c in REQUIRED_COLUMNS if c not in _source_columns]
if _missing_required:
    raise KeyError(f"Colonne obbligatorie mancanti: {_missing_required}")
df = pd.read_csv(DATA_PATH, usecols=REQUIRED_COLUMNS, low_memory=False)
print(f"Shape analitica: {df.shape[0]:,} righe × {df.shape[1]} colonne caricate "
      f"(dataset sorgente: {len(_source_columns)} colonne)")
print(f"Progetti: {df['project_url'].nunique():,}")
print(f"Developer: {df['developer_id'].nunique():,}")
print(f"Finestre: {df['time_window_id'].nunique()} "
      f"({df['time_window_id'].min()} -> {df['time_window_id'].max()})")


# Risultato dell'esecuzione di riferimento:
# - 24.916 righe e 76 colonne nel CSV sorgente;
# - 19 colonne caricate perché sono le sole necessarie alla pipeline;
# - 332 progetti, 10.416 developer e 154 finestre trimestrali;
# - circa 2,4 righe per developer: il panel è molto sparso e contiene molti
#   contributori occasionali.


# ─────────────────────────────────────────────
# STEP 2 — ESTRAZIONE STQF DA JSON
# ─────────────────────────────────────────────
print("\n")
print("STEP 2 — Estrazione Socio-Technical Quality Factors")

"Evita errori di parsing JSON e valori NaN"
def safe_json(s):
    try:
        if pd.isna(s) or str(s).strip() in ('', '{}'): return {}
        return json.loads(s)
    except: return {}


# La colonna grezza Socio-Technical Quality Factors contiene, in ogni cella, un oggetto JSON con 34 chiavi.
# Ne sono state selezionate 20 per l'analisi, che vengono estratte in colonne separate.
# Le 14 chiavi rimanenti non sono state considerate per il focus dell'analisi o per la ridondanza con altre feature già presenti nel dataset.
# E sono: ml.code.devs, perc.ml.only.devs, perc.code.only.devs, perc.ml.code.devs, sponsored.devs, core.mail.devs,
# sponsored.core.devs, ratio.sponsored.core, mail.truck, code.truck, code.turnover, core.global.turnover, core.mail.turnover, mail.mod

STQF_KEYS = [
    'devs', 'ml.only.devs', 'code.only.devs',
    'ratio.smelly.devs', 'global.turnover', 'core.code.turnover',
    'global.truck', 'density', 'st.congruence', 'communicability',
    'num.tz', 'ratio.smelly.quitters', 'ratio.sponsored',
    'closeness.centr', 'betweenness.centr', 'degree.centr',
    'global.mod', 'code.mod', 'core.global.devs', 'core.code.devs'
]

stqf_parsed = df['Socio-Technical Quality Factors'].apply(safe_json)
stqf_df = pd.DataFrame(
    [{k: d.get(k, np.nan) for k in STQF_KEYS} for d in stqf_parsed]
)
stqf_df.columns = ['stqf_' + c.replace('.', '_') for c in stqf_df.columns]    #viene sostituito il punto con underscore per evitare problemi di analisi
df = pd.concat([df.reset_index(drop=True), stqf_df.reset_index(drop=True)], axis=1)
print(f"Feature STQF estratte: {len(stqf_df.columns)}")

# ─────────────────────────────────────────────
# STEP 3 — OUTCOME DEVELOPER E PROGETTO
# ─────────────────────────────────────────────
print("\n")
print("STEP 3 — Outcome developer e progetto")

# Ranking cronologico delle finestre trimestrali.
all_windows = sorted(df['time_window_id'].astype(str).unique())
window_rank_map = {window: rank for rank, window in enumerate(all_windows)}
GLOBAL_LAST = window_rank_map[all_windows[-1]]

df['time_window_rank'] = df['time_window_id'].astype(str).map(window_rank_map)
df['last_interaction_rank'] = (
    df['last_interaction_window_id'].astype(str).map(window_rank_map)
)

# ---------------------------------------------------------------------------
# RQ1 — Outcome developer con censura a destra
# ---------------------------------------------------------------------------
# L'orizzonte del developer è la fine osservata del proprio progetto, non il
# cutoff globale. Un developer è evento se la sua ultima interazione precede
# di almeno quattro finestre la fine osservata del progetto.
_project_last_observed = df.groupby('project_url')['time_window_rank'].transform('max')
_developer_first = (
    df.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('min')
)
_developer_last = (
    df.groupby(['project_url', 'developer_id'])['last_interaction_rank'].transform('max')
)

df['inactivity_lag'] = _project_last_observed - _developer_last
df['has_failed'] = (df['inactivity_lag'] >= INACTIVITY_THRESHOLD).astype(int)
df['obs_duration'] = np.where(
    df['has_failed'] == 1,
    _developer_last - _developer_first,
    _project_last_observed - _developer_first,
) + 1

_project_alive_at_global_cutoff = (
    GLOBAL_LAST - _project_last_observed
) < INACTIVITY_THRESHOLD

df['censor_type'] = np.where(
    df['has_failed'] == 1,
    'evento',
    np.where(_project_alive_at_global_cutoff, 'admin', 'fine_progetto'),
)

# ---------------------------------------------------------------------------
# RQ2 — Outcome progetto commit-based
# ---------------------------------------------------------------------------
# 1. Somma dei commit dei developer per progetto e finestra.
# 2. Individuazione della finestra che contiene il commit più recente.
# 3. Somma dei commit nelle quattro finestre terminanti con tale finestra.
# 4. INACTIVE se la somma è inferiore a 12.
#
# Le finestre mancanti nel range non compaiono nel panel e contribuiscono zero
# alla somma. Non viene applicato alcun altro vincolo.
df['_commits_raw'] = pd.to_numeric(df['commits_count'], errors='coerce').fillna(0)
project_window_commits = (
    df.groupby(['project_url', 'time_window_rank'], as_index=False)
      .agg(project_commits=('_commits_raw', 'sum'))
)

project_status_rows = []
for project_url, group in project_window_commits.groupby('project_url'):
    group = group.sort_values('time_window_rank')
    positive_windows = group[group['project_commits'] > 0]

    if positive_windows.empty:
        # Caso difensivo: nel dataset analizzato non si verifica.
        last_commit_rank = int(group['time_window_rank'].max())
        commits_last_12m = 0.0
    else:
        last_commit_rank = int(positive_windows['time_window_rank'].max())
        lookback_start = last_commit_rank - PROJECT_ACTIVITY_WINDOWS + 1
        commits_last_12m = float(
            group.loc[
                group['time_window_rank'].between(lookback_start, last_commit_rank),
                'project_commits',
            ].sum()
        )

    project_status_rows.append({
        'project_url': project_url,
        'project_last_commit_rank': last_commit_rank,
        'project_commits_last_12m': commits_last_12m,
        'project_avg_commits_month_last_12m': commits_last_12m / 12.0,
        'proj_failed': int(commits_last_12m < PROJECT_COMMIT_THRESHOLD),
    })

project_status = pd.DataFrame(project_status_rows)
df = df.merge(project_status, on='project_url', how='left', validate='many_to_one')
df = df.drop(columns='_commits_raw')
project_status.to_csv(OUT_DIR / 'project_commit_based_outcome.csv', index=False)

# Riepilogo developer: una unità è una coppia progetto-developer.
developer_status = (
    df.groupby(['project_url', 'developer_id'])['has_failed'].max()
)
print(
    f"Developer censurati: {(developer_status == 0).sum():,} "
    f"({(developer_status == 0).mean():.1%}) | "
    f"eventi di abbandono: {(developer_status == 1).sum():,} "
    f"({(developer_status == 1).mean():.1%})"
)

project_counts = project_status['proj_failed'].value_counts()
print(
    f"Progetti ACTIVE (>=12 commit): {project_counts.get(0, 0)} | "
    f"INACTIVE (<12 commit): {project_counts.get(1, 0)}"
)
print(
    "Commit mediani nelle quattro finestre finali: "
    f"{project_status['project_commits_last_12m'].median():.1f}"
)
print(f"Durata a rischio developer, mediana: {df['obs_duration'].median():.0f} finestre")

# Risultato dell'esecuzione di riferimento:
# - developer: 85,1% eventi e 14,9% censurati a livello di coppia
#   progetto-developer;
# - progetti: 170 INACTIVE e 162 ACTIVE;
# - la mediana è 11 commit nelle quattro finestre finali, quindi la soglia di
#   12 produce due classi quasi perfettamente bilanciate.
# La nuova etichetta non dipende dalla distanza temporale dal cutoff globale.


# ─────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING E PREPROCESSING
# ─────────────────────────────────────────────
print("\n")
print("STEP 4 — Feature engineering e preprocessing")

# Estrazione nuove variabili: project age e developer tenure
df = df.sort_values(['project_url', 'developer_id', 'time_window_rank']).reset_index(drop=True)

# ordinamento del panel e calcola due variabili temporali ossia l'età del progetto e la tenure del developer — come distanza (in finestre) dalla prima apparizione.
proj_first = (df.groupby('project_url')['time_window_rank']
                .min().rename('proj_first_rank'))
dev_first  = (df.groupby(['project_url','developer_id'])['time_window_rank']
                .min().rename('dev_first_rank'))
df = df.merge(proj_first, on='project_url', how='left')
df = df.merge(dev_first,  on=['project_url','developer_id'], how='left')

df['project_age']     = df['time_window_rank'] - df['proj_first_rank']
df['developer_tenure']= df['time_window_rank'] - df['dev_first_rank']

# Lista completa delle feature
FEATURES_BASE = [
    # Social Debt 16 feature
    'community_smell_count',
    'stqf_ratio_smelly_devs',
    'stqf_global_turnover',
    'stqf_core_code_turnover',
    'stqf_global_truck',
    'stqf_density',
    'stqf_st_congruence',
    'stqf_communicability',
    'stqf_num_tz',
    'stqf_ratio_smelly_quitters',
    'stqf_ratio_sponsored',
    'stqf_devs',
    'stqf_closeness_centr',
    'stqf_degree_centr',
    'stqf_core_global_devs',
    'stqf_core_code_devs',
    # Technical Debt 5 feature
    'traditional_smell_count',
    'ml_smell_count',
    'vulnerability_count',
    'vulnerability_high',
    'bug_introduced_count_rszz',
    # Attività (controllo) 4 feature
    'commits_count',
    'code_churn',
    'bug_fix_commits_count',
    'files_touched_count',
    # Developer (controllo) 2 feature
    'se_score',
    'sentiment_score',
    # Temporali (controllo) 2 feature
    'project_age',
    'developer_tenure',
]
FEATURES_BASE = [f for f in FEATURES_BASE if f in df.columns] # tiene solo quelle esistenti

# Motivazione selezione feature:
# queste 29 sono le uniche colonne che sono contemporaneamente numeriche e utilizzabili, collegate alle ipotesi dello studio:
# che il debito sociale e il debito tecnico influenzino il fallimento;
# e prive di leakage rispetto all'esito.
# Le 4 variabili di attività, le 2 developer e le 2mtemporali non sono "predittori d'interesse" ma controlli:
# servono a isolare l'effetto del debito da quello di quanto uno lavora e da quanto tempo è nel progetto.


# Le feature escluse sono:

# Variabili di leakage, ossia che costruiscono l'esito (11):
# is_abandoned, abandonment_status, last_interaction_window_id, last_interaction_window_label, abandoned_since_window_id,
# abandoned_since_window_label, abandoned_since_date, project_abandoned_developers_count, project_abandoned_developers_ids,
# più le derivate inactivity_lag e last_interaction_rank create precedentemente.

# Identificatori e metadati temporali (6):
# project_url, developer_id, time_window_id, time_window_label, time_window_start, time_window_end.

# Campi testuali (13):
# aliases, emails, classification, gender_source, pronouns_detected, sentiment_label, last_commit_hash, last_commit_date, last_commit_message,
# last_message_before_abandonment_hash, last_message_before_abandonment_date, last_message_before_abandonment.

#JSON grezzi (10):
# Socio-Technical Quality Factors, developer_raw_json, project_metrics_json, project_community_smells_count_json, project_ml_smells_count_json,
# project_traditional_smells_count_json, project_vulnerabilities_count_json, project_vulnerabilities_severity_count_json,
# project_community_smell_instances_json,project_collaboration_edges_json.

#Ridondanti:
# project_loc, project_nom, i vari project_*_total e project_vulnerabilities_high/medium/low, project_collaboration_edges_count,
# lines_added, lines_deleted, avg_files_per_commit, community_smells/ml_smells/traditional_smells e le loro *_instances, vulnerability_medium, vulnerability_low,
# gender, gender_confidence, ai_score, ml_score, sentiment_messages_count.


# feature engineering: conversione a numerico (NaN -> 0) e rimozione outlier top 1%
for f in FEATURES_BASE:
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

# Rimozione outlier top 1%
for col in FEATURES_BASE:
    cap = df[col].quantile(OUTLIER_QUANTILE) #0.99
    if cap > 0:
        df[col] = df[col].clip(upper=cap) #schiacciamo gli outlier sopra il 99° percentile senza rimuovere le righe (per evitare di perdere i developer con più contributi)

# Log-trasformazione variabili skewed
LOG_COLS = [
    'commits_count', 'code_churn', 'traditional_smell_count',
    'community_smell_count', 'bug_introduced_count_rszz', 'files_touched_count'
]
LOG_COLS = [c for c in LOG_COLS if c in df.columns]
for c in LOG_COLS:
    df[f'log_{c}'] = np.log1p(df[c])    # log(1+x): per comprimere le code lunghe, riducendo l'effetto degli outlier e rendendo la distribuzione più simmetrica
# log1p gestisce anche i valori zero senza generare errori.


# Feature finali (sostituisce le log-trasformate)
FEAT_FINAL = (
    [f'log_{c}' for c in LOG_COLS] +
    [f for f in FEATURES_BASE if f not in LOG_COLS]
)
FEAT_FINAL = [f for f in FEAT_FINAL if f in df.columns]
print(f"Feature totali: {len(FEAT_FINAL)}")

# ─────────────────────────────────────────────
# STEP 5 — EDA
# ─────────────────────────────────────────────
print("\n")
print("STEP 5 — EDA e statistiche descrittive")

key_cols = [
    'community_smell_count', 'traditional_smell_count', 'ml_smell_count',
    'vulnerability_count', 'stqf_global_turnover', 'stqf_ratio_smelly_devs',
    'commits_count', 'code_churn', 'stqf_devs', 'stqf_st_congruence'
]
key_cols = [c for c in key_cols if c in df.columns]

desc = (df.groupby('has_failed')[key_cols]
          .agg(['mean', 'median', 'std'])
          .round(3))
print(desc.to_string())
desc.to_csv(OUT_DIR / 'descriptive_stats.csv')

# Risultato dell'esecuzione di riferimento, livello developer:
# - censurati: 19,99 commit medi e churn medio 13.190;
# - eventi: 6,55 commit medi e churn medio 4.217;
# - community smells quasi identici nei due gruppi (0,712);
# - smell tradizionali e vulnerabilità sono più elevati nei censurati.
# Le metriche tecniche grezze riflettono quindi soprattutto esposizione e volume
# di attività: chi continua a contribuire produce anche più codice e più smell.


# Kaplan-Meier per developer
print("\nCalcolo Kaplan-Meier...")
from lifelines import KaplanMeierFitter
# durata = tenure massima +1, evento = has_failed; stratificazione per dimensione team (mediana)

rows_km = []
for (proj, dev), grp in df.groupby(['project_url', 'developer_id']):
    rows_km.append({
        'duration' : grp['obs_duration'].max(),
        'event'    : int(grp['has_failed'].max()),
        'proj_size': grp['stqf_devs'].mean()
    })
km_df = pd.DataFrame(rows_km)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curva globale
kmf = KaplanMeierFitter()
kmf.fit(km_df['duration'], event_observed=km_df['event'], label='Tutti i developer')
kmf.plot_survival_function(ax=axes[0], color='steelblue')
axes[0].set_title("Kaplan-Meier: Sopravvivenza globale\n(ACTIVE vs INACTIVE)")
axes[0].set_xlabel("Finestre trimestrali dalla prima contribuzione")
axes[0].set_ylabel("P(non fallimento)")
axes[0].grid(alpha=0.3)

# costruisce una curva di sopravvivenza per developer, dove l'"evento" è il fallimento e la "durata" è la tenure

# Stratificato per dimensione progetto (piccolo vs grande)
median_size = km_df['proj_size'].median() # mediana della dimensione del team (numero medio di developer per progetto)
for label, mask, color in [
    (f'Team piccolo (≤{median_size:.0f} dev)', km_df['proj_size'] <= median_size, 'tomato'),
    (f'Team grande (>{median_size:.0f} dev)',  km_df['proj_size'] >  median_size, 'seagreen'),
]:
    sub = km_df[mask]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(sub['duration'], event_observed=sub['event'], label=f"{label} (n={len(sub):,})")
    kmf2.plot_survival_function(ax=axes[1], color=color)
axes[1].set_title("Kaplan-Meier stratificato per dimensione team")
axes[1].set_xlabel("Finestre trimestrali")
axes[1].set_ylabel("P(non fallimento)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'kaplan_meier.png', dpi=150)
plt.close()
print("Kaplan-Meier salvato.")

# Risultato Kaplan-Meier:
# la sopravvivenza individuale diminuisce rapidamente nelle prime finestre,
# confermando la presenza di molti contributori occasionali. Le curve per team
# piccoli e grandi rimangono relativamente vicine: la dimensione del progetto
# non separa nettamente la persistenza del singolo developer.

# Distribuzione delle feature per ACTIVE vs INACTIVE
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_pairs = [
    ('community_smell_count',    'Community Smells'),
    ('traditional_smell_count',  'Traditional Code Smells'),
    ('ml_smell_count',           'ML Smells'),
    ('stqf_global_turnover',     'Global Turnover'),
    ('stqf_ratio_smelly_devs',   'Ratio Smelly Devs'),
    ('vulnerability_count',      'Vulnerabilità'),
]
for ax, (col, title) in zip(axes.flat, plot_pairs):
    if col not in df.columns: continue
    for val, color, label in [(0,'steelblue','ACTIVE'), (1,'tomato','FAILED')]:
        sub = df[df['has_failed'] == val][col].dropna()
        sub = sub[sub <= sub.quantile(0.95)]
        ax.hist(sub, bins=30, alpha=0.55, density=True, color=color, label=label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.suptitle("Distribuzione metriche SD/TD per outcome (ACTIVE vs INACTIVE)", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'eda_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Distribuzioni EDA salvate.")

# ─────────────────────────────────────────────
# STEP 6 — VIF
# ─────────────────────────────────────────────
print("\n")
print("STEP 6 — Test VIF (multicollinearità)")


#VIF per misurare la multicollinearità tra le feature. Rimuove quelle con VIF > 5

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X_vif = df[FEAT_FINAL].fillna(0)
    X_vif = X_vif.loc[:, (X_vif.std() > 0) & ~X_vif.isin([np.inf, -np.inf]).any()]
    X_c   = add_constant(X_vif)
    vif_data = pd.DataFrame({
        'feature': X_vif.columns,
        'VIF'    : [variance_inflation_factor(X_c.values, i+1)
                    for i in range(X_vif.shape[1])]
    }).sort_values('VIF', ascending=False)

    print(vif_data.to_string(index=False))
    vif_data.to_csv(OUT_DIR / 'vif_results.csv', index=False)

    high_vif   = vif_data[vif_data['VIF'] > VIF_THRESHOLD]['feature'].tolist()
    FEAT_MODEL = [f for f in FEAT_FINAL if f not in high_vif]
    print(f"\nRimosse per VIF > {VIF_THRESHOLD}: {high_vif}")
    print(f"Feature nel modello finale: {len(FEAT_MODEL)}")
except Exception as e:
    print(f"VIF error: {e}")
    FEAT_MODEL = FEAT_FINAL

# ─────────────────────────────────────────────
# STEP 7 — EARLY-STAGE: Logistic Regression
# ─────────────────────────────────────────────
print("\n")
print("STEP 7 — Early-Stage Model (Logistic Regression)")

print(f"Definizione early-stage: developer_tenure <= {EARLY_STAGE_WINDOWS} finestre")


# sui soli developer "giovani" (tenure ≤ 3) stima una regressione logistica: probabilità di fallire in funzione delle metriche SD/TD.
# L'idea è: il debito socio-tecnico precoce predice l'abbandono?
# Il McFadden R² misura la bontà di adattamento

# Una sola osservazione per coppia progetto-developer. Le feature delle prime
# finestre sono mediate, mentre l'outcome e' quello canonico del developer.
# Questa aggregazione evita di trattare come indipendenti piu' righe dello
# stesso developer e rende il modello piu' rapido e statisticamente coerente.
_early_source = df[df['developer_tenure'] <= EARLY_STAGE_WINDOWS].copy()
_early_agg = {f: 'mean' for f in FEAT_MODEL}
_early_agg['has_failed'] = 'max'
df_early = (
    _early_source.groupby(['project_url', 'developer_id'], as_index=False)
                 .agg(_early_agg)
)
print(f"Developer early-stage: {len(df_early):,}  |  Event rate: {df_early['has_failed'].mean():.2%}")

try:
    import statsmodels.api as sm

    X_e = df_early[FEAT_MODEL].fillna(0)
    X_e = X_e.loc[:, X_e.std() > 0]
    # Standardizzazione: coefficienti e OR sono riferiti a +1 deviazione standard.
    X_e = (X_e - X_e.mean()) / X_e.std(ddof=0).replace(0, 1)
    y_e = df_early['has_failed']

    logit_model = sm.GLM(
        y_e,
        sm.add_constant(X_e),
        family=sm.families.Binomial()
    ).fit(maxiter=100, method='irls')

    mcfadden = 1 - logit_model.llf / logit_model.llnull
    print(f"\nMcFadden R²: {mcfadden:.4f}")
    print(logit_model.summary2())

    res_logit = pd.DataFrame({
        'feature'    : logit_model.params.index,
        'coef'       : logit_model.params.values,
        'odds_ratio' : np.exp(np.clip(logit_model.params.values, -15, 15)),
        'p_value'    : logit_model.pvalues.values,
        'sig'        : pd.cut(logit_model.pvalues.values,
                              bins=[-1, 0.001, 0.01, 0.05, 1],
                              labels=['***','**','*','ns'])
    }).sort_values('p_value')
    res_logit.to_csv(OUT_DIR / 'early_stage_logit_results.csv', index=False)

    # Plot odds ratio variabili significative
    sig_logit = res_logit[
        (res_logit['sig'].isin(['***','**','*'])) &
        (res_logit['feature'] != 'const')
    ].sort_values('odds_ratio')

    if not sig_logit.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, len(sig_logit) * 0.6)))
        colors = ['tomato' if o > 1 else 'steelblue' for o in sig_logit['odds_ratio']]
        bars = ax.barh(sig_logit['feature'], sig_logit['odds_ratio'],
                       color=colors, edgecolor='white', alpha=0.85)
        ax.axvline(1, color='black', linewidth=1, linestyle='--')
        for bar, (_, row) in zip(bars, sig_logit.iterrows()):
            ax.text(bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height()/2,
                    f"OR={row['odds_ratio']:.2f} {row['sig']}",
                    va='center', fontsize=8.5)
        ax.set_xlabel("Odds Ratio\nRosso > 1 = aumenta rischio fallimento  |  Blu < 1 = protettivo")
        ax.set_title(f"Logistic Regression — Early Stage (p < 0.05)\n"
                     f"McFadden R² = {mcfadden:.3f}  |  n = {len(df_early):,} osservazioni")
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'logit_odds_ratios.png', dpi=150)
        plt.close()
    print("Logit salvato.")
except Exception as e:
    print(f"Logit error: {e}")
    import traceback; traceback.print_exc()



# Risultato dell'esecuzione di riferimento:
# - 11.360 developer, event rate 85,15%, McFadden R² = 0,109;
# - rischio maggiore: ratio_smelly_devs OR=1,29, ratio_smelly_quitters
#   OR=1,27, global_truck OR=1,20, communicability OR=1,11 e bug introdotti
#   OR=1,18;
# - protettivi: commit OR=0,67, tenure OR=0,72, project_age OR=0,88,
#   bug-fix commit OR=0,87 e numero di fusi orari OR=0,91;
# - smell tradizionali, community smell aggregato, ML smell e vulnerabilità
#   non risultano significativi.
# Gli OR sono riferiti a un incremento di una deviazione standard.


# ─────────────────────────────────────────────
# STEP 8 — LATER-STAGE: Cox PH
# ─────────────────────────────────────────────
print("\n")
print("STEP 8 — Later-Stage Model (Cox Proportional-Hazards)")


# Sui developer "maturi" (tenure > 3) usa un modello di Cox (survival analysis): stima l'hazard, cioè il rischio istantaneo di fallire, in funzione delle covariate.
# Il penalizer=0.1 (regolarizzazione ridge) stabilizza le stime in presenza di collinearità residua.
# L'exp(coef) è l'Hazard Ratio: >1 aumenta il rischio, <1 lo riduce.
# La concordance (tipo AUC per survival) misura la capacità discriminante.

df_late = df[df['developer_tenure'] > EARLY_STAGE_WINDOWS].copy()
# collassa il panel a 1 riga per developer: durata = tenure max, evento = has_failed max, feature = media
print(f"Osservazioni later-stage: {len(df_late):,}")

# Un record per developer (durata = finestre totali, event = has_failed)
print("Costruzione panel (1 riga per developer)...")
rows_cox = []
for (proj, dev), grp in df_late.groupby(['project_url', 'developer_id']):
    row = {
        'duration': grp['obs_duration'].max(),
        'event'   : int(grp['has_failed'].max()),
    }
    for f in FEAT_MODEL:
        row[f] = grp[f].mean() if f in grp.columns else 0.0
    rows_cox.append(row)

cox_panel = pd.DataFrame(rows_cox)
cox_panel = (cox_panel[cox_panel['duration'] > 0]
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0))

feat_valid = [f for f in FEAT_MODEL if cox_panel[f].std() > 0]
print(f"Panel: {cox_panel.shape[0]:,} developer  |  Event rate: {cox_panel['event'].mean():.2%}")
print(f"Feature valide: {len(feat_valid)}")

try:
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=0.1) # penalizzazione L2 per stabilità
    cph.fit(cox_panel[feat_valid + ['duration', 'event']],
            duration_col='duration', event_col='event')
    cph.print_summary()

    cox_results = cph.summary.reset_index()
    cox_results.to_csv(OUT_DIR / 'later_stage_cox_results.csv', index=False)

    # Plot coefficienti significativi
    sig_cox = cox_results[cox_results['p'] < 0.05].sort_values('coef')
    if not sig_cox.empty:
        fig, ax = plt.subplots(figsize=(10, max(5, len(sig_cox) * 0.6)))
        colors = ['tomato' if c > 0 else 'steelblue' for c in sig_cox['coef']]
        bars = ax.barh(sig_cox['covariate'], sig_cox['coef'],
                       color=colors, edgecolor='white', alpha=0.85,
                       xerr=sig_cox['se(coef)'], capsize=3)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        for _, row in sig_cox.iterrows():
            pstar = ('***' if row['p'] < 0.001 else
                     '**'  if row['p'] < 0.01  else '*')
            offset = row['coef'] + (0.02 if row['coef'] >= 0 else -0.02)
            ax.text(offset + row['se(coef)'],
                    list(sig_cox['covariate']).index(row['covariate']),
                    f"HR={row['exp(coef)']:.2f} {pstar}",
                    va='center', fontsize=8.5,
                    ha='left' if row['coef'] >= 0 else 'right')
        ax.set_xlabel("Log Hazard Ratio (± SE)\nRosso > 0 = aumenta rischio  |  Blu < 0 = protettivo")
        ax.set_title(f"Cox PH — Later Stage (p < 0.05)\n"
                     f"Concordance = {cph.concordance_index_:.3f}  |  "
                     f"n = {len(cox_panel):,} developer")
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'cox_hazard_ratios.png', dpi=150)
        plt.close()
    print(f"Cox salvato. Concordance index: {cph.concordance_index_:.4f}")
except Exception as e:
    print(f"Cox error: {e}")
    import traceback; traceback.print_exc()




# Risultato dell'esecuzione di riferimento:
# - 2.791 developer later-stage, 2.001 eventi e 790 censurati;
# - concordance = 0,888;
# - rischio: global_truck HR=2,13, turnover HR=1,71, communicability
#   HR=1,45 e smelly_quitters HR=1,35;
# - protezione: commit HR=0,65 e tenure HR=0,89;
# - il debito tecnico non mantiene un'associazione positiva significativa.

# ─────────────────────────────────────────────
# STEP 9 — INTERAZIONI SD × TD
# ─────────────────────────────────────────────
print("\n")
print("STEP 9 — Modello esteso con interazioni SD × TD")

# Si testa l'ipotesi di sinergia tra debito sociale (SD) e tecnico (TD)
# I termini di interazione sono semplici prodotti tra una metrica SD e una TD, poi si rifitta il Cox

interaction_pairs = [
    ('community_smell_count',    'traditional_smell_count', 'SD_comm_x_TD_trad'),
    ('stqf_ratio_smelly_devs',   'vulnerability_count',     'SD_smelly_x_TD_vuln'),
    ('stqf_global_turnover',     'code_churn',              'SD_turnover_x_TD_churn'),
    ('stqf_global_turnover',     'traditional_smell_count', 'SD_turnover_x_TD_trad'),
    ('stqf_ratio_smelly_quitters','bug_introduced_count_rszz','SD_quitters_x_TD_bugs'),
]

for sd, td, name in interaction_pairs:
    if sd in df_late.columns and td in df_late.columns:
        df_late[name] = df_late[sd] * df_late[td]  # termine di interazione dato dal prodotto
        cap = df_late[name].quantile(0.99)
        if cap > 0:
            df_late[name] = df_late[name].clip(upper=cap)
        print(f"  Creato: {name}")

interact_cols = [n for _, _, n in interaction_pairs if n in df_late.columns]

try:
    from lifelines import CoxPHFitter

    rows_int = []
    for (proj, dev), grp in df_late.groupby(['project_url', 'developer_id']):
        row = {
            'duration': grp['obs_duration'].max(),
            'event'   : int(grp['has_failed'].max()),
        }
        for f in feat_valid + interact_cols:
            row[f] = grp[f].mean() if f in grp.columns else 0.0
        rows_int.append(row)

    cox_int = (pd.DataFrame(rows_int)
                 .pipe(lambda x: x[x['duration'] > 0])
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0))

    feat_int_ok = [f for f in feat_valid + interact_cols if cox_int[f].std() > 0]
    cph_int = CoxPHFitter(penalizer=0.1)
    cph_int.fit(cox_int[feat_int_ok + ['duration', 'event']],
                duration_col='duration', event_col='event')

    print("\nModello Cox con interazioni SD × TD:")
    cph_int.print_summary()

    int_results = cph_int.summary.reset_index()
    int_results.to_csv(OUT_DIR / 'cox_interactions_results.csv', index=False)

    # Plot solo i termini di interazione
    int_only = int_results[int_results['covariate'].str.startswith('SD_')]
    if not int_only.empty:
        fig, ax = plt.subplots(figsize=(10, max(3, len(int_only) * 0.9)))
        colors = ['tomato' if c > 0 else 'steelblue' for c in int_only['coef']]
        ax.barh(int_only['covariate'], int_only['coef'],
                color=colors, edgecolor='white', alpha=0.85,
                xerr=int_only['se(coef)'], capsize=4)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        for _, row in int_only.iterrows():
            pstar = ('***' if row['p'] < 0.001 else
                     '**'  if row['p'] < 0.01  else
                     '*'   if row['p'] < 0.05  else 'ns')
            ax.text(row['coef'] + row['se(coef)'] + 0.01,
                    list(int_only['covariate']).index(row['covariate']),
                    f"p={row['p']:.3f} {pstar}", va='center', fontsize=9)
        ax.set_xlabel("Log Hazard Ratio (± SE)")
        ax.set_title(f"Termini di interazione SD × TD\n"
                     f"Concordance modello base={cph.concordance_index_:.3f}  "
                     f"con interazioni={cph_int.concordance_index_:.3f}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'cox_interaction_terms.png', dpi=150)
        plt.close()
    print(f"Interazioni salvate. Concordance: {cph_int.concordance_index_:.4f}")
except Exception as e:
    print(f"Interazioni error: {e}")
    import traceback; traceback.print_exc()


# Risultato dell'esecuzione di riferimento:
# nessuno dei cinque termini SD × TD è significativo (tutti p > 0,37) e la
# concordance passa soltanto da 0,8879 a 0,8882. Non emerge quindi una sinergia
# moltiplicativa robusta tra debito sociale e tecnico nel rischio individuale.


# ─────────────────────────────────────────────
# STEP 10 — ANALISI PROGETTO: ACTIVE vs INACTIVE
# ─────────────────────────────────────────────
print("\n")
print("STEP 10 — Analisi a livello progetto: ACTIVE vs INACTIVE")

# SI sale dal livello developer al livello progetto, calcolando metriche medie per progetto e confrontando i 38 ACTIVE contro i 294 FAILED con il test di Mann-Whitney U

proj_rows = []
for proj, grp in df.groupby('project_url'):
    # Un progetto è FAILED se tutti i suoi developer sono failed
    proj_failed = int(grp['proj_failed'].max())
    proj_rows.append({
        'project_url'         : proj,
        'proj_failed'         : proj_failed,
        'n_developers'        : grp['developer_id'].nunique(),
        'n_windows'           : grp['time_window_id'].nunique(),
        'avg_community_smells': grp['community_smell_count'].mean(),
        'avg_trad_smells'     : grp['traditional_smell_count'].mean(),
        'avg_ml_smells'       : grp['ml_smell_count'].mean(),
        'avg_vuln'            : grp['vulnerability_count'].mean(),
        'avg_turnover'        : grp['stqf_global_turnover'].mean(),
        'avg_truck'           : grp['stqf_global_truck'].mean(),
        'avg_churn'           : grp['code_churn'].mean(),
        'avg_commits'         : grp['commits_count'].mean(),
        'avg_smelly_devs'     : grp['stqf_ratio_smelly_devs'].mean(),
        'avg_st_congruence'   : grp['stqf_st_congruence'].mean(),
        'avg_density'         : grp['stqf_density'].mean(),
        'avg_smelly_quitters' : grp['stqf_ratio_smelly_quitters'].mean(),
    })
proj_level = pd.DataFrame(proj_rows)

print(f"Progetti ACTIVE: {(proj_level['proj_failed']==0).sum()}")
print(f"Progetti INACTIVE: {(proj_level['proj_failed']==1).sum()}")
proj_level.to_csv(OUT_DIR / 'project_level_outcomes.csv', index=False)

# Confronto statistico ACTIVE vs INACTIVE
active_p = proj_level[proj_level['proj_failed'] == 0]
failed_p = proj_level[proj_level['proj_failed'] == 1]
compare_cols = [
    'avg_community_smells', 'avg_trad_smells', 'avg_ml_smells',
    'avg_vuln', 'avg_turnover', 'avg_churn', 'n_developers',
    'avg_smelly_devs', 'avg_st_congruence', 'avg_density',
    'avg_smelly_quitters'
]

print("\nConfronto medie ACTIVE vs INACTIVE (livello progetto):")
compare_df = pd.DataFrame({
    'ACTIVE': active_p[compare_cols].mean(),
    'FAILED': failed_p[compare_cols].mean(),
    'diff%' : ((failed_p[compare_cols].mean() - active_p[compare_cols].mean())
               / active_p[compare_cols].mean().replace(0, np.nan) * 100)
}).round(3)
print(compare_df.to_string())
compare_df.to_csv(OUT_DIR / 'active_vs_failed_comparison.csv')

print("\nMann-Whitney U (ACTIVE vs INACTIVE a livello progetto):")
mw_rows = []
for col in compare_cols:
    stat, p = scipy_stats.mannwhitneyu(
        active_p[col].dropna(), failed_p[col].dropna(), alternative='two-sided')
    pstar = ('***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns')
    print(f"  {col:35s}: p={p:.4f} {pstar}")
    mw_rows.append({'feature': col, 'U': stat, 'p_value': p, 'sig': pstar,
                    'mean_active': active_p[col].mean(),
                    'mean_failed': failed_p[col].mean()})
mw_df = pd.DataFrame(mw_rows)
mw_df.to_csv(OUT_DIR / 'active_vs_failed_mw.csv', index=False)

# Boxplot confronto
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_cols = ['avg_community_smells', 'avg_trad_smells',
             'avg_turnover', 'avg_smelly_devs',
             'avg_st_congruence', 'avg_density']
for ax, col in zip(axes.flat, plot_cols):
    data   = [active_p[col].dropna(), failed_p[col].dropna()]
    labels = ['ACTIVE', 'FAILED']
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], ['steelblue', 'tomato']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    # Aggiungi p-value
    row_mw = mw_df[mw_df['feature'] == col]
    if not row_mw.empty:
        ax.set_title(f"{col.replace('avg_','').replace('_',' ').title()}\n"
                     f"p={row_mw['p_value'].values[0]:.4f} {row_mw['sig'].values[0]}")
    ax.grid(alpha=0.3)
plt.suptitle("Confronto ACTIVE vs INACTIVE — metriche SD/TD (livello progetto)", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'active_vs_failed_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Boxplot salvati.")


# Risultato dell'esecuzione di riferimento, 162 ACTIVE vs 170 INACTIVE:
# - INACTIVE con più community smells: +14,7%, p=0,009;
# - INACTIVE con più smelly developers: +18,2%, p<0,001;
# - INACTIVE con minore ST congruence: -29,6%, p<0,001;
# - INACTIVE con minore network density: -14,2%, p<0,001;
# - smell tradizionali medi inferiori negli INACTIVE: -18,9%, p=0,035;
# - turnover, churn, vulnerabilità, ML smell e smelly quitters non differiscono;
# - la dimensione media è diversa, ma il Mann-Whitney non è significativo
#   (p=0,224). La nuova definizione non rende più la taglia un discriminante
#   quasi meccanico come accadeva con la cessazione collettiva.


# ─────────────────────────────────────────────
# STEP 11 — HEATMAP CORRELAZIONE
# ─────────────────────────────────────────────
print("\n")
print("STEP 11 — Heatmap correlazione")

# matrice di correlazione di Pearson tra le principali metriche e has_failed, con maschera triangolare per leggibilità


corr_cols = [
    'community_smell_count', 'traditional_smell_count', 'ml_smell_count',
    'vulnerability_count', 'stqf_global_turnover', 'stqf_ratio_smelly_devs',
    'stqf_density', 'stqf_st_congruence', 'stqf_global_truck',
    'code_churn', 'commits_count', 'se_score', 'has_failed'
]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr() # Pearson

fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            annot_kws={'size': 9},
            cbar_kws={'label': 'Pearson r'})
ax.set_title("Matrice di correlazione — SD, TD e has_failed\n"
             "(332 progetti, 10K developer, ACTIVE vs INACTIVE)")
plt.tight_layout()
plt.savefig(OUT_DIR / 'correlation_heatmap.png', dpi=150)
plt.close()
print("Heatmap salvata.")


# Risultato dell'esecuzione di riferimento:
# la heatmap conferma correlazioni interne tra famiglie di metriche, per
# esempio community smells/ratio smelly devs e smell tradizionali/
# vulnerabilità. La relazione lineare diretta con l'abbandono del developer
# resta invece debole: i modelli temporali e multivariati sono più informativi
# della sola correlazione di Pearson.


# ─────────────────────────────────────────────
# STEP 12 — TRUCK FACTOR & OWNERSHIP 
# ─────────────────────────────────────────────
print("\n")
print("STEP 12 — Truck factor & ownership ")

# Non esiste ownership per-file: files_touched_count, lines_added/deleted, code_churn sono aggregati per developer.
# E' stato scelto il proxy standard. Per ogni progetto ordino i developer per commit totali e definisco core / truck-factor il gruppo minimo che copre ≥50% dei commit
# del progetto. Poi confronto core vs periferici e misuro la concentrazione dei contributi (indice di Gini) per progetto.


dev_proj = (df.groupby(['project_url', 'developer_id'])
              .agg(commits=('commits_count', 'sum'), churn=('code_churn', 'sum'),
                   files=('files_touched_count', 'sum'), failed=('has_failed', 'max'), pfail=('proj_failed', 'max')).reset_index())
_flags, _rows = [], []
for proj, g in dev_proj.groupby('project_url'):
    g = g.sort_values('commits', ascending=False); tot = g['commits'].sum()
    if tot == 0:
        g['is_core'] = 0
    else:
        cutoff = (g['commits'].cumsum() / tot >= 0.50).values.argmax()  # bus factor: min set >=50% commit
        g['is_core'] = 0; g.iloc[:cutoff + 1, g.columns.get_loc('is_core')] = 1
    _flags.append(g[['project_url', 'developer_id', 'is_core']])
    _rows.append({'project_url': proj, 'n_dev': len(g), 'truck_factor': int(g['is_core'].sum()),
                  'gini_commits': gini(g['commits']), 'proj_failed': int(g['pfail'].max())})
dev_proj = dev_proj.merge(pd.concat(_flags), on=['project_url', 'developer_id'])
proj_conc = pd.DataFrame(_rows)
c_core = dev_proj[dev_proj.is_core == 1]; c_peri = dev_proj[dev_proj.is_core == 0]
print(f"Core (truck-factor): {len(c_core):,} | Periferici: {len(c_peri):,}")
for m, lbl in [('commits', 'Commit'), ('churn', 'Churn'), ('files', 'File')]:
    _, p = _ss.mannwhitneyu(c_core[m], c_peri[m]); d = cliffs_delta(c_core[m].values, c_peri[m].values)
    print(f"  {lbl:8s} core_med={c_core[m].median():9.1f} peri_med={c_peri[m].median():7.1f} p={p:.1e} d={d:+.2f} ({eff_label(d)})")
print(f"  Fallimento: core={c_core['failed'].mean():.1%} periferici={c_peri['failed'].mean():.1%}")
g_act = proj_conc[proj_conc.proj_failed == 0]['gini_commits'].dropna()
g_fai = proj_conc[proj_conc.proj_failed == 1]['gini_commits'].dropna()
_, p_g = _ss.mannwhitneyu(g_act, g_fai)
print(f"  Gini contributi: ACTIVE={g_act.median():.3f} FAILED={g_fai.median():.3f} p={p_g:.1e}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].boxplot([np.log1p(c_core['commits']), np.log1p(c_peri['commits'])], tick_labels=['Core', 'Periferici']); ax[0].set_title('Commit (log)')
ax[1].boxplot([g_act, g_fai], tick_labels=['ACTIVE', 'FAILED']); ax[1].set_title(f'Gini per progetto (p={p_g:.1e})')
ax[2].bar(['Core', 'Periferici'], [c_core['failed'].mean(), c_peri['failed'].mean()], color=['#c0392b', '#2980b9']); ax[2].set_ylim(0, 1); ax[2].set_title('Tasso fallimento')
plt.tight_layout(); plt.savefig(OUT_DIR / 'truck_factor_ownership.png', dpi=110); plt.close()
proj_conc.to_csv(OUT_DIR / 'truck_factor_concentration.csv', index=False)
print("Truck factor salvato.")


# Risultato dell'esecuzione di riferimento:
# - 604 core e 10.756 periferici;
# - commit mediani 178 vs 2, churn 56.763 vs 57 e file 388 vs 3;
# - Cliff's delta tra 0,92 e 0,96: concentrazione molto elevata;
# - abbandono developer: 42,9% nei core e 87,5% nei periferici;
# - Gini dei commit quasi identico tra progetti ACTIVE e INACTIVE
#   (0,772 vs 0,769; p=0,56).
# L'ownership concentrata distingue nettamente core e periferia, ma non
# discrimina la nuova etichetta commit-based di progetto.


# ─────────────────────────────────────────────
# STEP 13 — SOGLIE SD/TD
# ─────────────────────────────────────────────
print("\n")
print("STEP 13 — Soglie SD/TD")

# 1. ROC + indice di Youden per il cutpoint ottimale e l'AUC univariata di ogni metrica;
# 2. tasso di fallimento per decile;
# 3. albero decisionale (max_depth=3, solo SD/TD, senza tenure/età). 



SDTD = {'community_smell_count': 'SD community', 'stqf_ratio_smelly_devs': 'SD smelly devs',
        'stqf_global_turnover': 'SD turnover', 'stqf_st_congruence': 'SD st-congruence',
        'traditional_smell_count': 'TD trad', 'ml_smell_count': 'TD ML', 'vulnerability_count': 'TD vuln'}
for c in SDTD: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
_rr = []
print(f"  {'metrica':16s} {'soglia':>9s} {'AUC':>6s} {'sens':>5s} {'spec':>5s}")
for c, lbl in SDTD.items():
    thr, a, se, sp = youden(df['has_failed'], df[c])
    print(f"  {lbl:16s} {thr:9.3f} {a:6.3f} {se:5.2f} {sp:5.2f}"); _rr.append({'metric': lbl, 'threshold': thr, 'auc': a})
pd.DataFrame(_rr).to_csv(OUT_DIR / 'thresholds_roc.csv', index=False)
_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, class_weight='balanced', random_state=0)
_tree.fit(df[list(SDTD)], df['has_failed'])
print(f"  Albero (solo SD/TD) AUC={roc_auc_score(df['has_failed'], _tree.predict_proba(df[list(SDTD)])[:,1]):.3f}")
print(export_text(_tree, feature_names=[SDTD[c] for c in SDTD]))
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for ax, (c, lbl) in zip(axs.ravel(), list(SDTD.items())[:4]):
    q = pd.qcut(df[c].rank(method='first'), 10, labels=False)
    ax.plot(range(1, 11), df.groupby(q)['has_failed'].mean().values, 'o-'); ax.set_title(lbl); ax.grid(alpha=.3)
plt.suptitle('Tasso di fallimento per decile'); plt.tight_layout()
plt.savefig(OUT_DIR / 'thresholds_failure_rate.png', dpi=110); plt.close()
print("Thresholds salvato.")


# Risultato dell'esecuzione di riferimento, outcome developer:
# - turnover è la sola metrica univariata moderatamente informativa
#   (AUC=0,608; soglia Youden circa 0,514);
# - le altre AUC sono vicine o inferiori a 0,50;
# - l'albero SD/TD raggiunge AUC=0,670, ma è una stima in-sample e va
#   interpretata come esplorativa, non come prestazione predittiva validata.


# ─────────────────────────────────────────────
# STEP 14 — SENTIMENT & EMOZIONI
# ─────────────────────────────────────────────
print("\n")
print("STEP 14 — Sentiment & Emozioni")

# sentiment_label è degenere: 24.912 "Neutral", 4 "Positive", 0 "Negative".
# Nel JSON però c'è sentiment_emotions con 6 emozioni (Anger, Fear, Sadness, Love, Joy, Surprise) è stato estratto e usato insieme a sentiment_score e se_score. 
# Confronto con Mann-Whitney + Cliff's delta.

print(
    "  sentiment_label degenere (>99.9% Neutral) -> "
    "escluso. Uso emozioni + score."
)

# Le emozioni sono contenute all'interno di developer_raw_json
RAW_JSON_COLUMN = 'developer_raw_json'

if RAW_JSON_COLUMN not in df.columns:
    raise KeyError(
        f"Colonna '{RAW_JSON_COLUMN}' non trovata nel dataset."
    )

# Converte ogni JSON testuale in un dizionario Python.
# safe_json è già definita nello STEP 2.
developer_json = df[RAW_JSON_COLUMN].apply(safe_json)


def extract_emotion(raw_data, emotion):
    """
    Estrae il valore di un'emozione dal JSON del developer.
    Restituisce NaN quando il dato non è disponibile.
    """
    if not isinstance(raw_data, dict):
        return np.nan

    emotions = raw_data.get('sentiment_emotions', {})

    if not isinstance(emotions, dict):
        return np.nan

    return emotions.get(emotion, np.nan)


# Estrazione delle sei emozioni
for emotion in EMO:
    column_name = f'emo_{emotion}'

    df[column_name] = pd.to_numeric(
        developer_json.map(
            lambda raw_data, current_emotion=emotion:
            extract_emotion(raw_data, current_emotion)
        ),
        errors='coerce'
    )

print("  Colonne emozionali estratte:")

for emotion in EMO:
    column_name = f'emo_{emotion}'

    print(
        f"    {column_name:15s}: "
        f"{df[column_name].notna().sum():,}/{len(df):,}"
    )

# Elenco dei segnali analizzati
SENT = {
    'sentiment_score': 'Sentiment',
    'se_score': 'SE score'
}

for emotion in EMO:
    column_name = f'emo_{emotion}'

    if (
        column_name in df.columns
        and df[column_name].notna().sum() >= 10
    ):
        SENT[column_name] = emotion

# Divisione tra osservazioni ACTIVE e FAILED
act = df[df['has_failed'] == 0].copy()
fai = df[df['has_failed'] == 1].copy()

sentiment_results = []

for column_name, label in SENT.items():

    active_values = pd.to_numeric(
        act[column_name],
        errors='coerce'
    ).dropna()

    failed_values = pd.to_numeric(
        fai[column_name],
        errors='coerce'
    ).dropna()

    if len(active_values) < 5 or len(failed_values) < 5:
        print(
            f"  {label:12s}: dati insufficienti "
            f"(ACTIVE={len(active_values)}, "
            f"FAILED={len(failed_values)})"
        )
        continue

    _, p_value = _ss.mannwhitneyu(
        active_values,
        failed_values,
        alternative='two-sided'
    )

    delta = cliffs_delta(
        active_values.values,
        failed_values.values
    )

    print(
        f"  {label:12s} "
        f"ACT={active_values.median():7.3f} "
        f"FAIL={failed_values.median():7.3f} "
        f"p={p_value:.1e} "
        f"d={delta:+.2f} ({eff_label(delta)})"
    )

    sentiment_results.append({
        'signal': label,
        'active_med': active_values.median(),
        'failed_med': failed_values.median(),
        'p_value': p_value,
        'cliffs_d': delta,
        'effect_size': eff_label(delta),
        'n_active': len(active_values),
        'n_failed': len(failed_values)
    })

pd.DataFrame(sentiment_results).to_csv(
    OUT_DIR / 'sentiment_active_vs_failed.csv',
    index=False
)

# ── Grafici ──

available_emotions = [
    emotion
    for emotion in EMO
    if (
        f'emo_{emotion}' in df.columns
        and df[f'emo_{emotion}'].notna().sum() >= 10
    )
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Profilo emotivo medio
if available_emotions:

    x = np.arange(len(available_emotions))
    width = 0.38

    active_means = [
        act[f'emo_{emotion}'].mean()
        for emotion in available_emotions
    ]

    failed_means = [
        fai[f'emo_{emotion}'].mean()
        for emotion in available_emotions
    ]

    axes[0].bar(
        x - width / 2,
        active_means,
        width,
        label='ACTIVE',
        color='#2980b9'
    )

    axes[0].bar(
        x + width / 2,
        failed_means,
        width,
        label='FAILED',
        color='#c0392b'
    )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        available_emotions,
        rotation=20
    )

    axes[0].set_ylabel('Valore medio')
    axes[0].set_title('Profilo emotivo medio')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

else:
    axes[0].text(
        0.5,
        0.5,
        'Dati emozionali non disponibili',
        ha='center',
        va='center'
    )
    axes[0].set_axis_off()

# Boxplot SE score
axes[1].boxplot(
    [
        pd.to_numeric(
            act['se_score'],
            errors='coerce'
        ).dropna(),

        pd.to_numeric(
            fai['se_score'],
            errors='coerce'
        ).dropna()
    ],
    tick_labels=['ACTIVE', 'FAILED']
)

axes[1].set_title('SE score')
axes[1].set_ylabel('Valore')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()

plt.savefig(
    OUT_DIR / 'sentiment_emotions.png',
    dpi=110
)

plt.close()

print("Sentiment analysis salvata.")


# Risultato dell'esecuzione di riferimento:
# le differenze emozionali sono statisticamente rilevabili per la numerosità
# del campione, ma tutte hanno Cliff's delta trascurabile. Solo SE score mostra
# un effetto piccolo (mediana 2,10 nei censurati vs 0,70 negli eventi;
# d=0,17). Le emozioni non costituiscono quindi un discriminante sostanziale.


# ─────────────────────────────────────────────
# STEP 15 — COX TEMPO-VARIANTE
# ─────────────────────────────────────────────
print("\n")
print("STEP 15 — Cox tempo-variante (pre-fallimento)")

CENSOR_GAP = INACTIVITY_THRESHOLD
if 'log_commits_count' not in df.columns:
    df['log_commits_count'] = np.log1p(pd.to_numeric(df['commits_count'], errors='coerce').fillna(0))

# esito canonico a livello (progetto, developer)
_ev = df.groupby(['project_url', 'developer_id']).agg(
    event=('has_failed', 'max'), n_win=('time_window_rank', 'nunique')).reset_index()

ADV = {'community_smell_count': 'SD community', 'stqf_global_turnover': 'SD turnover',
       'stqf_st_congruence': 'SD st-congr', 'traditional_smell_count': 'TD trad',
       'vulnerability_count': 'TD vuln', 'log_commits_count': 'Commit(log)',
       'se_score': 'SE score', 'sentiment_score': 'Sentiment'}
for c in ADV:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)


print(f"Esito canonico (censura a destra): eventi={_ev['event'].mean():.1%} censurati={1-_ev['event'].mean():.1%}")
_tv = _ev[_ev.n_win >= 3][['project_url', 'developer_id', 'event']]
_sub = df.merge(_tv[['project_url', 'developer_id']], on=['project_url', 'developer_id'])
_rows = []
for (proj, dev), grp in _sub.groupby(['project_url', 'developer_id']):
    grp = grp.sort_values('developer_tenure')
    ev = int(_tv[(_tv.project_url == proj) & (_tv.developer_id == dev)]['event'].iloc[0])
    ten = grp['developer_tenure'].values
    for j in range(len(grp)):
        start = ten[j]; stop = ten[j + 1] if j < len(grp) - 1 else ten[j] + 1
        if stop <= start: stop = start + 1
        r = {'id': hash((proj, dev)) & 0xffffffff, 'start': start, 'stop': stop,
             'event': ev if j == len(grp) - 1 else 0}
        for c in ADV: r[c] = grp[c].values[j]
        _rows.append(r)
_long = pd.DataFrame(_rows)
for c in ADV:
    s = _long[c].std()
    if s > 0: _long[c] = (_long[c] - _long[c].mean()) / s
_ctv = CoxTimeVaryingFitter(penalizer=0.1)
_ctv.fit(_long, id_col='id', event_col='event', start_col='start', stop_col='stop')
_res = _ctv.summary[['exp(coef)', 'p']].copy()
_res.index = [ADV.get(i, i) for i in _res.index]; _res = _res.reindex(_res['p'].sort_values().index)
print(f"  Dev={_long['id'].nunique():,} intervalli={len(_long):,} | HR (standardizzate, >1 = piu' rischio):")
for name, row in _res.iterrows():
    sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else ''
    print(f"   {name:16s} HR={row['exp(coef)']:.3f} p={row['p']:.4f} {sig}")
_res.to_csv(OUT_DIR / 'timevarying_cox.csv')
print("Analisi Cox tempo-variante conclusa")

# Risultato dell'esecuzione di riferimento:
# turnover corrente aumenta il rischio istantaneo (HR=1,109, p<0,001), mentre
# commit è fortemente protettivo (HR=0,778). Smell tradizionali e vulnerabilità
# hanno HR leggermente inferiori a 1: vicino all'uscita misurano soprattutto la
# riduzione dell'attività, non un effetto protettivo causale del debito.

# ─────────────────────────────────────────────
# STEP 16 — FIRMA PRE-FALLIMENTO
# ─────────────────────────────────────────────
print("\n")
print("STEP 16 — Firma pre-fallimento (traiettoria finestre finali)")


_aid = _ev[(_ev.n_win >= 3) & (_ev.event == 1)][['project_url', 'developer_id']]
_traj = df.merge(_aid, on=['project_url', 'developer_id']).sort_values(
    ['project_url', 'developer_id', 'time_window_rank']).copy()
_seq = _traj.groupby(['project_url', 'developer_id']).cumcount()
_tot = _traj.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('size')
_traj['pos'] = _seq - (_tot - 1)                       # 0 = ultima finestra, negativi = precedenti
SIG = ['community_smell_count', 'traditional_smell_count', 'stqf_global_turnover', 'log_commits_count', 'se_score']
_prof = _traj[_traj['pos'] >= -4].groupby('pos')[SIG].mean().round(3)
print(f"  Dev in abbandono con traiettoria: {_aid.shape[0]:,}")
print("  Media per posizione (0 = ultima finestra prima di sparire):")
print(_prof.to_string())
_first = _traj[_seq == 0].set_index(['project_url', 'developer_id'])[SIG]
_last  = _traj[_traj['pos'] == 0].set_index(['project_url', 'developer_id'])[SIG]
_pair = _first.add_suffix('_f').join(_last.add_suffix('_l')).dropna()
print(f"  Prima vs ultima finestra (Wilcoxon appaiato, n={len(_pair):,}):")
_sr = []
for c in SIG:
    a, b = _pair[c + '_f'], _pair[c + '_l']
    try: _, p = _ss.wilcoxon(a, b)
    except Exception: p = np.nan
    dpc = (b.mean() - a.mean()) / (abs(a.mean()) + 1e-9) * 100
    arrow = 'su' if b.median() > a.median() else 'giu' if b.median() < a.median() else '='
    print(f"   {c:26s} {arrow} prima={a.mean():6.3f} ultima={b.mean():6.3f} ({dpc:+.0f}%) p={p:.1e}")
    _sr.append({'metric': c, 'first_mean': a.mean(), 'last_mean': b.mean(), 'delta_pct': dpc, 'p': p})
pd.DataFrame(_sr).to_csv(OUT_DIR / 'prefailure_signature.csv', index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for c in ['traditional_smell_count', 'stqf_global_turnover', 'log_commits_count', 'se_score']:
    v = _prof[c] / (_prof[c].abs().max() + 1e-9)
    ax[0].plot(_prof.index, v, 'o-', label=c)
ax[0].set_xlabel('finestre prima della sparizione (0 = ultima)'); ax[0].set_ylabel('valore normalizzato')
ax[0].legend(fontsize=8); ax[0].set_title('Firma pre-fallimento (traiettoria)'); ax[0].grid(alpha=.3)
hr = _res['exp(coef)']; ax[1].barh(range(len(hr)), hr.values, color=['#c0392b' if v > 1 else '#2980b9' for v in hr.values])
ax[1].set_yticks(range(len(hr))); ax[1].set_yticklabels(hr.index); ax[1].axvline(1, color='k', ls='--')
ax[1].set_title('Cox tempo-variante — Hazard Ratio')
plt.tight_layout(); plt.savefig(OUT_DIR / 'prefailure_timevarying.png', dpi=110); plt.close()
print("Analisi pre-fallimento conclusa")

# Risultato dell'esecuzione di riferimento, 1.460 traiettorie di abbandono:
# dalla prima all'ultima finestra i commit calano del 30%, gli smell tecnici
# prodotti del 60% e SE score dell'8%, mentre il turnover cresce del 23%.
# Community smells rimane stabile (+2%, p=0,48). La firma pre-abbandono è
# quindi soprattutto un collasso dell'attività accompagnato da turnover.

# ─────────────────────────────────────────────
# STEP 17 — LIVELLO PROGETTO, CONTROLLO PER DIMENSIONE
# ─────────────────────────────────────────────
print("\n")
print("STEP 17 — Livello progetto, controllo per dimensione del team")


_p = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'), SD_community=('community_smell_count', 'mean'),
    SD_turnover=('stqf_global_turnover', 'mean'), SD_stcongr=('stqf_st_congruence', 'mean'),
    SD_smellyq=('stqf_ratio_smelly_quitters', 'mean'), TD_trad=('traditional_smell_count', 'mean'),
    TD_ml=('ml_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_failed=('proj_failed', 'max')).reset_index()
_p['log_n_dev'] = np.log1p(_p['n_dev'])
import statsmodels.api as _sm
from statsmodels.discrete.discrete_model import Logit as _Logit
SDTD = ['SD_community', 'SD_turnover', 'SD_stcongr', 'SD_smellyq', 'TD_trad', 'TD_ml', 'TD_vuln']
print(f"  ACTIVE={int((_p.proj_failed==0).sum())} FAILED={int((_p.proj_failed==1).sum())} | "
      f"n_dev mediana ACTIVE={_p[_p.proj_failed==0]['n_dev'].median():.0f} FAILED={_p[_p.proj_failed==1]['n_dev'].median():.0f}")
print("  Coefficiente logit: GREZZO vs AGGIUSTATO per log(n_dev):")
_A = []
for m in SDTD:
    try:
        mr = _Logit(_p['proj_failed'], _sm.add_constant(_p[[m]])).fit(disp=0)
        ma = _Logit(_p['proj_failed'], _sm.add_constant(_p[[m, 'log_n_dev']])).fit(disp=0)
        flip = '  <-- CAMBIA SEGNO' if np.sign(mr.params[m]) != np.sign(ma.params[m]) else ''
        print(f"   {m:13s} grezzo={mr.params[m]:+6.2f}(p={mr.pvalues[m]:.3f}) | +taglia={ma.params[m]:+6.2f}(p={ma.pvalues[m]:.3f}){flip}")
        _A.append({'metric': m, 'coef_raw': mr.params[m], 'p_raw': mr.pvalues[m], 'coef_adj': ma.params[m], 'p_adj': ma.pvalues[m]})
    except Exception as e:
        print(f"   {m}: {e}")
print("   [il confronto mostra quanto le associazioni cambiano dopo il controllo per la taglia]")
pd.DataFrame(_A).to_csv(OUT_DIR / 'project_size_controlled_logit.csv', index=False)
_p['size_bin'] = pd.cut(_p['n_dev'], [0, 10, 25, 60, 150, 1e9], labels=['1-10', '11-25', '26-60', '61-150', '150+'])
print("  Confronto ACTIVE-FAILED ENTRO fascia (diff mediana; * p<0.05):")
_B = []
for m in SDTD:
    ln = f"   {m:13s} "
    for b in _p['size_bin'].cat.categories:
        sb = _p[_p.size_bin == b]; a = sb[sb.proj_failed == 0][m]; f = sb[sb.proj_failed == 1][m]
        if len(a) >= 3 and len(f) >= 3:
            try: _, pp = _ss.mannwhitneyu(a, f)
            except Exception: pp = 1.0
            ln += f"{b}:{(a.median()-f.median()):+.2f}{'*' if pp<.05 else ''} "; _B.append({'metric': m, 'bin': b, 'diff': a.median()-f.median(), 'p': pp})
    print(ln)
pd.DataFrame(_B).to_csv(OUT_DIR / 'project_size_matched.csv', index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8)); _dA = pd.DataFrame(_A).set_index('metric'); xx = np.arange(len(_dA))
ax[0].bar(xx - .19, -np.log10(_dA['p_raw'] + 1e-300), .38, label='grezzo', color='#7f8c8d')
ax[0].bar(xx + .19, -np.log10(_dA['p_adj'] + 1e-300), .38, label='+ n_dev', color='#c0392b')
ax[0].axhline(-np.log10(.05), ls='--', color='k'); ax[0].set_xticks(xx); ax[0].set_xticklabels(_dA.index, rotation=30, ha='right')
ax[0].set_ylabel('-log10(p)'); ax[0].legend(); ax[0].set_title("Significativita' prima/dopo controllo taglia")
ax[1].boxplot([np.log1p(_p[_p.proj_failed == 0]['n_dev']), np.log1p(_p[_p.proj_failed == 1]['n_dev'])], tick_labels=['ACTIVE', 'FAILED'])
ax[1].set_ylabel('log(1+n_dev)'); ax[1].set_title('Dimensione team (il confondente)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'project_size_control.png', dpi=110); plt.close()
print("Analisi a livello progetto conclusa")

# Risultato dell'esecuzione di riferimento:
# - classi quasi bilanciate e mediane di taglia vicine: 14 developer negli
#   ACTIVE e 12 negli INACTIVE;
# - dopo il controllo per log(n_dev), turnover e smelly quitters cambiano
#   segno e diventano fattori di rischio (p=0,031 e p=0,043);
# - community smells aumenta il rischio, mentre ST congruence e le metriche
#   tecniche mostrano coefficienti negativi. Le direzioni tecniche suggeriscono
#   ancora un forte effetto di attività/esposizione e non vanno lette causalmente.



# ═══════════════════════════════════════════════════════════════════════════
# INATTIVITÀ A LIVELLO PROGETTO — MODELLI MULTIVARIATI
#
# RQ2: quali fattori socio-tecnici sono associati alla probabilità che un
# progetto abbia prodotto meno di 12 commit nelle quattro finestre terminate
# con il proprio commit più recente?
#
# L'outcome usa direttamente il volume recente di commit. Per evitare leakage:
# - i commit recenti e la media commit/finestra NON sono usati come predittori;
# - turnover e smelly quitters sono esclusi dai modelli finali perché sono
#   misure molto vicine alla perdita di partecipazione e possono incorporare
#   informazione contemporanea all'esito;
# - restano community smells, struttura della rete, debito tecnico e controlli
#   strutturali non costruiti con i commit recenti.
#
# STEP 18: Logit binario di progetto.
# STEP 19: Cox con covariate medie e durata fino al commit più recente.
# STEP 20: Cox tempo-variante con covariate progetto-finestra.
# I Cox sono esplorativi: con questa definizione, gli ACTIVE sono censurati alla
# propria finestra di commit più recente, non al cutoff globale.
# ═══════════════════════════════════════════════════════════════════════════
import statsmodels.api as _sm2
from statsmodels.discrete.discrete_model import Logit as _Logit2
from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
from lifelines import CoxPHFitter as _CoxPH
 
# predittori PULITI (no turnover/quitters): solo fattori non derivati dall'abbandono, VIF < soglia
PREDS = ['SD_community', 'SD_stcongr', 'SD_truck', 'TD_trad', 'TD_ml', 'TD_vuln', 'log_n_dev']
 
_pl = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'),
    SD_community=('community_smell_count', 'mean'), SD_stcongr=('stqf_st_congruence', 'mean'),
    SD_truck=('stqf_global_truck', 'mean'), TD_trad=('traditional_smell_count', 'mean'),
    TD_ml=('ml_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_first=('time_window_rank', 'min'), proj_last=('project_last_commit_rank', 'max'),
    proj_failed=('proj_failed', 'max')).reset_index()
_pl['log_n_dev'] = np.log1p(_pl['n_dev'])
_pl['duration'] = _pl['proj_last'] - _pl['proj_first'] + 1
_Z = _pl[PREDS].copy()                                     # standardizzo -> OR/HR per +1 dev. std.
for c in PREDS:
    s = _Z[c].std(); _Z[c] = (_Z[c] - _Z[c].mean()) / s if s > 0 else 0.0
_vif_df = pd.DataFrame({'f': PREDS, 'VIF': [round(_vif(_Z.values, i), 2) for i in range(len(PREDS))]})


# ─────────────────────────────────────────────
# STEP 18 — LOGIT MULTIVARIATO A LIVELLO PROGETTO
# ─────────────────────────────────────────────

print("\n")
print("STEP 18 — Logit multivariato a livello progetto")

print(f"Progetti: {len(_pl)} | ACTIVE {(_pl.proj_failed==0).sum()} | INACTIVE {(_pl.proj_failed==1).sum()}")
print("  ESCLUSI per outcome-adjacency: SD_turnover e SD_smellyq; esclusi anche i commit recenti usati nella target")
print(f"  VIF (tutti < {VIF_THRESHOLD}): " + ", ".join(f"{r.f}={r.VIF}" for r in _vif_df.itertuples()))
_ml = _Logit2(_pl['proj_failed'], _sm2.add_constant(_Z[PREDS])).fit(disp=0, maxiter=200)
print(f"  McFadden R2 = {1 - _ml.llf/_ml.llnull:.3f}  (OR per +1 dev. std.; OR>1 -> maggiori odds di inattivita')")
_rowsL = []
for p in PREDS:
    orr = np.exp(_ml.params[p]); pv = _ml.pvalues[p]
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    print(f"   {p:13s} OR={orr:5.2f}  p={pv:.3f}  {sig}")
    _rowsL.append({'predictor': p, 'coef': _ml.params[p], 'odds_ratio': orr, 'p_value': pv})
pd.DataFrame(_rowsL).to_csv(OUT_DIR / 'rq2_project_logit_multivariate.csv', index=False)
print("Analisi logit multivariata a livello progetto conclusa")


# Risultato dell'esecuzione di riferimento:
# McFadden R²=0,182. Risultano associati a minori odds di inattività:
# - ST congruence OR=0,48, p=0,001;
# - global truck OR=0,41, p<0,001;
# - traditional smells OR=0,64, p=0,021.
# Dimensione del team, community smells, ML smells e vulnerabilità non sono
# significativi nel modello congiunto. Le direzioni protettive di truck e
# smell tecnici non sono coerenti con una semplice ipotesi "più debito = più
# inattività" e richiedono cautela interpretativa.


# ─────────────────────────────────────────────
# STEP 19 — COX DI SOPRAVVIVENZA DEL PROGETTO
# ─────────────────────────────────────────────
 
print("\n")
print("STEP 19 — Cox di sopravvivenza a livello progetto")

_cox_pl = pd.concat([_pl[['duration', 'proj_failed']].rename(columns={'proj_failed': 'event'}), _Z[PREDS]], axis=1)
_cph_pl = _CoxPH(penalizer=0.1); _cph_pl.fit(_cox_pl, duration_col='duration', event_col='event')
print(f"  Progetti={len(_cox_pl)} | INACTIVE={int(_cox_pl['event'].sum())} | ACTIVE/censurati={int((_cox_pl['event']==0).sum())}")
print(f"  Concordance = {_cph_pl.concordance_index_:.3f}  (HR per +1 dev. std.; HR>1 -> evento di inattivita' osservato prima)")
_sm_pl = _cph_pl.summary[['exp(coef)', 'p']].reindex(_cph_pl.summary['p'].sort_values().index)
for name, row in _sm_pl.iterrows():
    sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else 'ns'
    print(f"   {name:13s} HR={row['exp(coef)']:5.2f}  p={row['p']:.3f}  {sig}")
_sm_pl.to_csv(OUT_DIR / 'rq2_project_cox.csv')
print("Analisi Cox a livello progetto conclusa")



# Risultato dell'esecuzione di riferimento:
# concordance=0,727. Team più grandi raggiungono più lentamente l'evento
# commit-based (HR=0,60, p<0,001); ML smells è associato a HR=0,79 (p=0,008).
# Gli altri fattori socio-tecnici non sono significativi. Il risultato è
# diverso dal Logit perché il Cox considera anche la durata fino all'ultimo
# commit e tratta gli ACTIVE come censurati.


# ─────────────────────────────────────────────
# STEP 20 — COX TEMPO-VARIANTE A LIVELLO PROGETTO
# ─────────────────────────────────────────────
 
print("\n")
print("STEP 20 — Cox tempo-variante a livello progetto")

_pw = df.groupby('project_url_x' if 'project_url_x' in df.columns else 'project_url')
_pw = df.groupby(['project_url', 'time_window_rank']).agg(
    n_dev=('developer_id', 'nunique'), SD_community=('community_smell_count', 'mean'),
    SD_stcongr=('stqf_st_congruence', 'mean'), SD_truck=('stqf_global_truck', 'mean'),
    TD_trad=('traditional_smell_count', 'mean'), TD_ml=('ml_smell_count', 'mean'),
    TD_vuln=('vulnerability_count', 'mean')).reset_index()
_pw['log_n_dev'] = np.log1p(_pw['n_dev'])
_pw['age'] = _pw['time_window_rank'] - _pw.groupby('project_url')['time_window_rank'].transform('min')
_pwin = df.groupby('project_url')['time_window_rank'].nunique()
_pfail = df.groupby('project_url')['proj_failed'].max()
_project_last_commit = df.groupby('project_url')['project_last_commit_rank'].max()
_rows_tv = []
for proj, g in _pw[_pw['project_url'].isin(_pwin[_pwin >= 3].index)].groupby('project_url'):
    g = g[g['time_window_rank'] <= _project_last_commit.loc[proj]].copy()
    g = g.sort_values('age'); ev = int(_pfail[proj]); ages = g['age'].values
    for j in range(len(g)):
        start = ages[j]; stop = ages[j + 1] if j < len(g) - 1 else ages[j] + 1
        if stop <= start: stop = start + 1
        r = {'id': hash(proj) & 0xffffffff, 'start': start, 'stop': stop, 'event': ev if j == len(g) - 1 else 0}
        for c in PREDS: r[c] = g[c].values[j]
        _rows_tv.append(r)
_ltv = pd.DataFrame(_rows_tv)
for c in PREDS:
    s = _ltv[c].std()
    if s > 0: _ltv[c] = (_ltv[c] - _ltv[c].mean()) / s
_ctv_pl = CoxTimeVaryingFitter(penalizer=0.1)
_ctv_pl.fit(_ltv, id_col='id', event_col='event', start_col='start', stop_col='stop')
print(f"  Progetti={_ltv['id'].nunique():,} | intervalli={len(_ltv):,} (metriche e dimensione VARIANO per finestra)")
_res_tv = _ctv_pl.summary[['exp(coef)', 'p']].reindex(_ctv_pl.summary['p'].sort_values().index)
for name, row in _res_tv.iterrows():
    sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else 'ns'
    print(f"   {name:13s} HR={row['exp(coef)']:5.2f}  p={row['p']:.3f}  {sig}")
_res_tv.to_csv(OUT_DIR / 'rq2_project_cox_timevarying.csv')
print("Analisi Cox a livello progetto conclusa")


# ─────────────────────────────────────────────
# grafici RQ2
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
_dl = pd.DataFrame(_rowsL).set_index('predictor')['odds_ratio']
ax[0].barh(range(len(_dl)), _dl.values, color=['#c0392b' if v > 1 else '#2980b9' for v in _dl.values])
ax[0].set_yticks(range(len(_dl))); ax[0].set_yticklabels(_dl.index); ax[0].axvline(1, color='k', ls='--')
ax[0].set_title('STEP 18 — Logit progetto leakage-free (OR, +1 SD)')
_hr = _sm_pl['exp(coef)']
ax[1].barh(range(len(_hr)), _hr.values, color=['#c0392b' if v > 1 else '#2980b9' for v in _hr.values])
ax[1].set_yticks(range(len(_hr))); ax[1].set_yticklabels(_hr.index); ax[1].axvline(1, color='k', ls='--')
ax[1].set_title('STEP 19 — Cox progetto leakage-free (HR, +1 SD)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_project_models.png', dpi=110); plt.close()
try:
    _fix = pd.read_csv(OUT_DIR / 'rq2_project_cox.csv', index_col=0)['exp(coef)']
    _idx = [i for i in PREDS if i in _res_tv.index and i in _fix.index]
    fig, ax = plt.subplots(figsize=(9, 4.6)); yy = np.arange(len(_idx)); w = .38
    ax.barh(yy + w/2, [_fix[i] for i in _idx], w, label='STEP 19 (medie fisse)', color='#7f8c8d')
    ax.barh(yy - w/2, [_res_tv['exp(coef)'][i] for i in _idx], w, label='STEP 20 (tempo-variante)', color='#c0392b')
    ax.axvline(1, color='k', ls='--'); ax.set_yticks(yy); ax.set_yticklabels(_idx)
    ax.set_xlabel('Hazard Ratio (+1 dev. std.)'); ax.legend(); ax.set_title('RQ2 leakage-free: covariate fisse vs tempo-varianti')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_timefixed_vs_timevarying.png', dpi=110); plt.close()
except Exception as _e:
    print("  (grafico confronto saltato)")
print("Confronto tra Step 18-19-20 completato.")

# Risultato dell'esecuzione di riferimento:
# nel Cox tempo-variante global_truck (HR=0,85, p<0,001) e log_n_dev
# (HR=0,86, p=0,001) risultano protettivi; le altre covariate non sono
# significative. La divergenza rispetto al Logit e al Cox statico mostra che
# le conclusioni di progetto sono sensibili alla forma temporale del modello,
# non che esista necessariamente un effetto cumulativo univoco.


# ═══════════════════════════════════════════════════════════════════════════
# La nuova definizione commit-based non rende più la dimensione del team parte
# meccanica dell'outcome. La scala può comunque confondere le associazioni, per
# cui viene controllata esplicitamente
# con piu' indicatori strutturali e verifichiamo se i fattori socio-tecnici
# (come bus factor, debito tecnico) mantengono un effetto AL NETTO della scala.
# Controlli di scala (VIF < soglia, non derivati dall'abbandono):
# - log_n_dev            : dimensione del team (n. contributori)
# - log_n_windows        : estensione della storia osservata del progetto
# - log_loc              : dimensione del codice (lines of code)
# L'eta' del progetto e' controllata dal Cox tramite l'asse del tempo (durata).
# ═══════════════════════════════════════════════════════════════════════════
import statsmodels.api as _sm3
from statsmodels.discrete.discrete_model import Logit as _Logit3
from lifelines import CoxPHFitter as _CoxPH3
 
_pl2 = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'), n_win=('time_window_rank', 'nunique'),
    tot_commits=('commits_count', 'sum'), loc=('project_loc', 'max') if 'project_loc' in df.columns else ('commits_count', 'sum'),
    SD_truck=('stqf_global_truck', 'mean'), SD_community=('community_smell_count', 'mean'),
    TD_trad=('traditional_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_first=('time_window_rank', 'min'), proj_last=('project_last_commit_rank', 'max'),
    proj_failed=('proj_failed', 'max')).reset_index()
_pl2['log_n_dev'] = np.log1p(_pl2['n_dev'])
_pl2['log_n_windows'] = np.log1p(_pl2['n_win'])
_pl2['log_loc'] = np.log1p(pd.to_numeric(_pl2['loc'], errors='coerce').fillna(0))
_pl2['duration'] = _pl2['proj_last'] - _pl2['proj_first'] + 1
 
SCALE = ['log_n_dev', 'log_n_windows', 'log_loc']       # controlli strutturali / scala
SOCTEC = ['SD_truck', 'SD_community', 'TD_trad', 'TD_vuln']    # fattori socio-tecnici (leakage-free)
ALLP = SOCTEC + SCALE
_Z2 = _pl2[ALLP].copy()
for c in ALLP:
    s = _Z2[c].std(); _Z2[c] = (_Z2[c] - _Z2[c].mean()) / s if s > 0 else 0.0



# ─────────────────────────────────────────────
# STEP 21 — RQ2 ROBUSTEZZA: CONTROLLI STRUTTURALI / DI SCALA
# ─────────────────────────────────────────────

print("\n")
print("STEP 21 — RQ2 robustezza: fattori socio-tecnici al netto della SCALA del progetto")

# ── STEP 21b — Logit con controlli di scala estesi ──
print(f"  Controlli di scala: {SCALE}")
_ml2 = _Logit3(_pl2['proj_failed'], _sm3.add_constant(_Z2[ALLP])).fit(disp=0, maxiter=200)
print(f"  [Logit] McFadden R2 = {1 - _ml2.llf/_ml2.llnull:.3f}  (OR per +1 dev. std.)")
for p in ALLP:
    orr = np.exp(_ml2.params[p]); pv = _ml2.pvalues[p]
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    tag = '(socio-tecnico)' if p in SOCTEC else '(scala)'
    print(f"    {p:20s} OR={orr:5.2f}  p={pv:.3f}  {sig:3s} {tag}")
 
# ── STEP 21b — Cox con controlli di scala estesi (eta' via asse del tempo) ──
_cox2 = pd.concat([_pl2[['duration', 'proj_failed']].rename(columns={'proj_failed': 'event'}), _Z2[ALLP]], axis=1)
_cph2 = _CoxPH3(penalizer=0.1); _cph2.fit(_cox2, duration_col='duration', event_col='event')
print(f"  [Cox] concordance = {_cph2.concordance_index_:.3f}  (HR per +1 dev. std.; eta' controllata dall'asse tempo)")
_s2 = _cph2.summary[['exp(coef)', 'p']]
for p in ALLP:
    hr = _s2.loc[p, 'exp(coef)']; pv = _s2.loc[p, 'p']
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    tag = '(socio-tecnico)' if p in SOCTEC else '(scala)'
    print(f"    {p:20s} HR={hr:5.2f}  p={pv:.3f}  {sig:3s} {tag}")
_s2.to_csv(OUT_DIR / 'rq2_robustness_scale_cox.csv')
pd.DataFrame([{'predictor': p, 'odds_ratio': np.exp(_ml2.params[p]), 'p_logit': _ml2.pvalues[p],
               'HR': _s2.loc[p, 'exp(coef)'], 'p_cox': _s2.loc[p, 'p'],
               'tipo': 'socio-tecnico' if p in SOCTEC else 'scala'} for p in ALLP]
            ).to_csv(OUT_DIR / 'rq2_robustness_scale.csv', index=False)


# ─────────────────────────────────────────────
# confronto: SD/TD con controllo MINIMO (solo n_dev, STEP 19) vs ESTESO (STEP 21)
print("  --- I fattori socio-tecnici SOPRAVVIVONO al controllo di scala esteso? ---")
try:
    _base = pd.read_csv(OUT_DIR / 'rq2_project_cox.csv', index_col=0)['exp(coef)']
    for p in SOCTEC:
        b = _base.get(p, np.nan); f = _s2.loc[p, 'exp(coef)']; pv = _s2.loc[p, 'p']
        verdetto = 'resta' if pv < .05 else 'si attenua/sparisce'
        print(f"    {p:13s} HR: n_dev-only={b:.2f}  ->  scala-estesa={f:.2f} (p={pv:.3f}) -> {verdetto}")
except Exception:
    pass
 
fig, ax = plt.subplots(figsize=(9, 4.6)); _hr2 = _s2.loc[ALLP, 'exp(coef)']
cols = ['#c0392b' if _s2.loc[p, 'p'] < .05 and _hr2[p] > 1 else ('#2980b9' if _s2.loc[p, 'p'] < .05 else '#bdc3c7') for p in ALLP]
ax.barh(range(len(ALLP)), [_hr2[p] for p in ALLP], color=cols)
ax.set_yticks(range(len(ALLP))); ax.set_yticklabels([f"{p} {'(ST)' if p in SOCTEC else '(scala)'}" for p in ALLP])
ax.axvline(1, color='k', ls='--'); ax.set_xlabel('Hazard Ratio (+1 dev. std.)')
ax.set_title('STEP 21 — RQ2: socio-tecnico al netto della scala (rosso=rischio sign., grigio=ns)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_robustness_scale.png', dpi=110); plt.close()
print("Step 21 (RQ2 robustezza scala) completato.")


# RISULTATI FINALI DELL'ESECUZIONE COMMIT-BASED
# ================================================================
#
# RQ1 — Quali fattori aumentano il rischio di abbandono del developer?
# ----------------------------------------------------------------
# L'unità è la coppia progetto-developer; l'evento è l'interruzione almeno
# quattro finestre prima dell'orizzonte del proprio progetto.
#
# Evidenze convergenti:
# - Logit early-stage: ratio smelly devs, smelly quitters, global truck,
#   communicability e bug introdotti aumentano le odds; commit e tenure sono
#   protettivi (McFadden R²=0,109).
# - Cox later-stage: global truck HR=2,13, turnover HR=1,71,
#   communicability HR=1,45 e smelly quitters HR=1,35; commit HR=0,65 e tenure
#   HR=0,89 (concordance=0,888).
# - Cox tempo-variante: turnover corrente HR=1,109; commit HR=0,778.
# - Firma pre-abbandono: turnover +23%, commit -30%, smell tecnici prodotti
#   -60% e SE score -8%.
# - Nessuna interazione SD × TD è significativa; sentiment ed emozioni hanno
#   effetti sostanzialmente trascurabili.
#
# Risposta RQ1:
# il segnale più coerente è sociale e dinamico: turnover, fragilità della rete
# e uscita di smelly developers accompagnano l'abbandono. Il debito tecnico
# statico non predice in modo robusto l'uscita individuale; vicino all'evento
# diminuisce insieme all'attività.
#
# RQ2 — Quali fattori distinguono i progetti ACTIVE dagli INACTIVE?
# ----------------------------------------------------------------
# L'unità è il progetto. INACTIVE significa meno di 12 commit nelle quattro
# finestre terminanti con il commit più recente. Risultano 170 INACTIVE e 162
# ACTIVE, senza ulteriori vincoli temporali.
#
# Evidenze:
# - confronto descrittivo: INACTIVE ha più community smells e smelly devs,
#   ma minore ST congruence, density e traditional smells; molte differenze
#   non sono significative e la dimensione non separa nettamente i gruppi;
# - Logit multivariato: ST congruence, global truck e traditional smells sono
#   associati a odds inferiori di inattività; log_n_dev non è significativo;
# - Cox statico: log_n_dev e ML smells hanno HR<1, mentre gli altri fattori non
#   risultano significativi;
# - Cox tempo-variante: global truck e log_n_dev hanno HR<1;
# - robustezza con scala estesa: nel Logit restano global truck, community
#   smells e traditional smells, ma nel Cox tutti i fattori socio-tecnici
#   diventano non significativi; domina log_n_windows (HR=0,33).
#
# Risposta RQ2:
# con la definizione commit-based non emerge un insieme stabile di metriche
# socio-tecniche che predica l'inattività in tutti i modelli. Alcune
# associazioni compaiono nel Logit, ma cambiano o scompaiono nei modelli di
# sopravvivenza e dopo i controlli di scala. La conclusione più prudente è che
# la bassa attività recente non è spiegata da una singola misura di debito:
# la relazione è sensibile al modello, alla scala e alla storia del progetto.
#
# Implicazione metodologica:
# la nuova definizione elimina la precedente dipendenza meccanica tra progetto
# fallito e assenza collettiva dei developer, produce classi bilanciate e rende
# più credibile il confronto di progetto. Al tempo stesso, poiché l'outcome è
# costruito con i commit recenti, le misure di attività basate sugli stessi
# commit devono restare escluse dai predittori per evitare leakage.


# ─────────────────────────────────────────────
# TERMINE ANALISI
# ─────────────────────────────────────────────
print("\n")
print("ANALISI COMPLETATA")

print(f"Output in: {OUT_DIR}")

for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:52s} ({f.stat().st_size:>9,} bytes)")
