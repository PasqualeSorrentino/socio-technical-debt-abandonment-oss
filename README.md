# Developers Leave, Projects Fade

### Socio-Technical Debt, Developer Departure, and Project Dormancy in Open-Source Machine Learning

Replication package for the paper by **Giammaria Giordano, Pasquale Sorrentino and Fabio Palomba** -
Software Engineering (SeSa) Lab, Department of Computer Science, University of Salerno, Italy.

---

## Abstract

**Context.** The continuity of open-source software depends on the voluntary participation of contributors: developers leave, and many projects fade until nobody maintains them. The causes may lie in *socio-technical debt*, which combines *social debt* and *technical debt*.

**Objective.** The two forms of debt are known to be related in terms of code quality. We investigate whether they are also associated with the departure of developers and the dormancy of projects, how much the answer depends on the definition of project dormancy, and how exits unfold.

**Method.** We analyze 332 engineered open-source ML projects from NICHE and 10,416 developers over quarterly windows. We model developer departure and project dormancy with logistic regression and (time-varying) Cox models, rebuild departure-based metrics in a lagged leave-one-out form, and compare temporal, maintenance-based, and combined definitions of project dormancy, controlling for scale and age. Robustness checks cover developer role, bots, proportional hazards, and within-project comparisons. A descriptive analysis covers the last activity of developers and projects and the fate of the debt left behind.

**Results.** Developer departure is associated with social rather than technical debt, and it is more likely where others have left. Higher team turnover and ratio of smelly quitters are associated with a higher risk of departure even when computed on the previous departures of other developers; the association distinguishes projects rather than unfolding as a chain within a project. Leavers are mostly casual contributors affected by smells of isolation, and core developers leave more often where the peripheral community is larger. Code smells and vulnerabilities are not associated with departure, but the debt they leave behind is rarely taken over and becomes orphaned in dormant projects, to the extent that the number of departures alone would imply. Projects fade through a long, unannounced tail of maintenance-only activity. Once scale is controlled, no socio-technical factor distinguishes projects that fade from those that stop while still maintained; under temporal definitions, technical debt relates to how early a project stops, not to whether it is still active once its age is controlled.

**Conclusion.** Social debt is associated with who leaves, technical debt with what remains, and project dormancy with maintenance continuity. Conclusions on project dormancy depend on how it is defined.

**Keywords:** Social Debt, Technical Debt, Community Smells, Open-Source Sustainability, Survival
Analysis, Empirical Software Engineering.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
└── analysis/
    ├── dataset/
    │   ├── mining.csv                 # longitudinal panel, 913 MB, Git LFS
    │   └── README.md                  # how to retrieve it
    ├── v2-maintenance-based/          # MAIN analysis -> RQ1 and RQ2 models of the paper
    │   ├── pipeline_v2_maintenance_based.py
    │   └── outputs/file/
    ├── v1-temporal/                   # SENSITIVITY analysis -> v1 term of Figures 3-4
    │   ├── pipeline_v1_temporal.py
    │   └── outputs/file/
    └── extended/                      # robustness checks, v3, descriptive analyses, RQ3
        ├── panel.py                   # shared data preparation (same panel as the v2 pipeline)
        ├── rq1_robustness.py          # Table A.1, lagged leave-one-out metrics, departure threshold
        ├── rq2_extras.py              # v1/v2/v3, cross-tabulation, age, bots, minimum detectable effect
        ├── descriptive_rq3.py         # Table 3, how developers leave and projects fade, RQ3
        ├── rq3_social.py              # Table 5, RQ3 social side: event study around core departures
        └── outputs/
```

Each pipeline is standalone, locates the dataset by walking up the directory tree, and writes every
table (`.csv`) and figure (`.png`) it produces into its own `outputs/file/` directory.

---

## The two analyses, and what each one produces

The paper compares two operationalizations of project dormancy, and this package contains one
pipeline for each. The two are **not** interchangeable: they classify projects differently and
therefore yield different RQ2 results. Both are required to reproduce the paper.

| Pipeline | Definition of project dormancy | Split | Role |
|---|---|---|---|
| **`v2-maintenance-based`** | Fewer than 12 commits in the four windows ending with the project's **most recent commit** | **218 active / 114 dormant** | **Main analysis.** Produces Tables 2 and 4 and the v2 bars of Figures 3-4 |
| **`v1-temporal`** | Last observable activity precedes the end of the observation period by at least four windows | 68 active / 264 dormant | **Sensitivity analysis.** Produces the v1 bars of Figures 3-4 |

A third, stricter definition (**v3**: active only if active under both v1 and v2; 41 active / 291 dormant) and the cross-tabulation of the definitions are computed in `extended/rq2_extras.py`.

**Figure 3 combines the v2 estimates from `v2-maintenance-based` with the v1 estimates from
`v1-temporal`**, and Figure 4 contrasts the two project splits: neither figure can be reproduced
from a single pipeline.

> **A note on terminology.** Following Valiev et al. (ESEC/FSE 2018), from whom the maintenance-based
> definition is taken, we refer to the project-level outcome as *dormancy* rather than abandonment.
> As those authors explicitly note, while all abandoned projects are dormant, not all dormant
> projects are abandoned, since some are simply feature-complete and require no further maintenance.
> The project-level results in this package therefore concern the cessation of maintenance activity,
> not the death of a project. At the developer level the paper uses the term *departure*, since the
> outcome describes the relation between a developer and a project, not the developer as a person
> (the pipelines still name it `has_failed`).

> **A note on the scale controls.** Under v1 the third structural control is the number of commits
> per window; under v2 it is the number of observed windows. The substitution is deliberate: commit
> intensity is what *defines* the v2 outcome, so using it as a control there would introduce leakage.
> The two columns of Figure 3 are therefore adjusted for slightly different control sets.

**RQ1 does not depend on the project-dormancy definition.** The developer-level outcome is identical
in both pipelines, so `later_stage_cox_results.csv`, `timevarying_cox.csv`,
`prefailure_signature.csv` and `cox_interactions_results.csv` can be taken from either of them. The
early-stage logistic regression is the one exception: `v2-maintenance-based` aggregates the early
stage to one row per project-developer pair and standardizes the predictors, so its odds ratios are
expressed per one standard deviation and are not comparable with those of `v1-temporal`. The paper
reports the `v2-maintenance-based` specification.

---

## Replication guide

### 1. Prerequisites

* **Python 3.13** (the study used 3.13.9)
* **Git LFS** - mandatory, the dataset is stored as an LFS object
* Several GB of free RAM, since the panel is loaded in memory

### 2. Clone the repository and fetch the dataset

Do **not** use the "Download ZIP" button on GitHub: it does not resolve Git LFS pointers and you
would obtain a 134-byte placeholder instead of the 913 MB panel.

```bash
git lfs install
git clone <REPOSITORY-URL>
cd <REPOSITORY-NAME>
git lfs pull
```

Verify that the dataset is the real file and not a pointer:

```bash
ls -l analysis/dataset/mining.csv      # expected: ~913 MB, not 134 bytes
```

If the size is 134 bytes, see `analysis/dataset/README.md`.

### 3. Install the dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the pipelines

From the repository root:

```bash
# Main analysis (v2) - RQ1 models and RQ2 project-level models
python analysis/v2-maintenance-based/pipeline_v2_maintenance_based.py

# Sensitivity analysis (v1) - needed for Figures 3 and 4
python analysis/v1-temporal/pipeline_v1_temporal.py

# Extended analyses (run rq2_extras.py before descriptive_rq3.py, which reuses its project labels)
python analysis/extended/rq1_robustness.py
python analysis/extended/rq2_extras.py
python analysis/extended/descriptive_rq3.py
python analysis/extended/rq3_social.py
python analysis/extended/rq1_extra.py   # after rq1_robustness.py; needs R with the survival package
python analysis/extended/rq1_signature_loo.py   # Figure 2 with the lagged leave-one-out turnover
```

`rq1_extra.py` adds the gamma-frailty Cox model (through `frailty_cox.R`), the models that
measure covariates before the time at risk (first later-stage window; time-varying covariates
lagged by one window), and a Benjamini-Hochberg correction pooled over all developer-level
models. If `Rscript` is not on the path, the frailty model is skipped.

To repeat the role-based analyses with an 80% core threshold, prefix the extended scripts with
`CORE_THRESHOLD=0.8`; outputs go to `analysis/extended/outputs_core80/`.

To repeat them with a departure threshold of eight windows (two years) for core developers
only, prefix `rq1_robustness.py`, `descriptive_rq3.py` and `rq3_social.py` with
`CORE_DEPARTURE_WINDOWS=8`; outputs go to `analysis/extended/outputs_coredep8/`
(copy `outputs/rq2_project_labels.csv` there first, since `descriptive_rq3.py` reuses it).
In this run the check against Table 2 is skipped, because the relabeling changes the outcome.

Each pipeline takes well under a minute once the panel is loaded. `rq1_robustness.py` first
re-estimates Table 2 and stops if it does not match the paper, so every extended analysis is
guaranteed to start from the same panel as the main models.

### 5. Where each result of the paper comes from

Paths are relative to `<pipeline>/outputs/file/` for the two pipelines and to `extended/outputs/`
for the extended analyses.

| Paper item | Script | File |
|---|---|---|
| Table 2 — RQ1, Cox model (later stage) | v2-maintenance-based | `later_stage_cox_results.csv` |
| RQ1, early-stage logistic regression | v2-maintenance-based | `early_stage_logit_results.csv` |
| Table A.1 — RQ1 robustness (no tenure, clustering, tenure strata, bots, role and team size, active projects, project strata, LOO) | extended/rq1_robustness.py | `rq1_cox_robustness_long.csv` |
| RQ1, Benjamini-Hochberg correction | extended/rq1_robustness.py | `rq1_benjamini_hochberg.csv` |
| RQ1, proportional hazards test | extended/rq1_robustness.py | `rq1_schoenfeld_test.csv` |
| RQ1, time-varying Cox (original and LOO) | extended/rq1_robustness.py | `rq1_timevarying_cox.csv` (original also in v2 `timevarying_cox.csv`) |
| RQ1, early stage clustered and with role/team size | extended/rq1_robustness.py | `rq1_early_stage_robustness.csv` |
| Lagged leave-one-out metrics, correlation with the originals | extended/rq1_robustness.py | `rq1_loo_correlations.csv` |
| RQ1, interaction terms | v2-maintenance-based | `cox_interactions_results.csv` |
| Section 3.3 — validation of the departure threshold | extended/rq1_robustness.py | `rq1_threshold_returns.csv` |
| Figure 2 — pre-departure signature | v2-maintenance-based | `prefailure_signature.csv` |
| Table 3 — last contribution of developers and final window of projects | extended/descriptive_rq3.py | `desc_last_contribution_categories.csv` |
| Casual contributors, community smells at exit | extended/descriptive_rq3.py | `desc_casual_contributors.csv`, `desc_smells_last_window.csv` |
| Table 4 — reference logit | v2-maintenance-based | `rq2_project_logit_multivariate.csv` |
| Table 4 — scale-adjusted logit and Cox | v2-maintenance-based | `rq2_robustness_scale.csv` |
| RQ2, time-varying Cox | v2-maintenance-based | `rq2_project_cox_timevarying.csv` |
| Figure 3 — v2 bars | v2-maintenance-based | `rq2_robustness_scale.csv` |
| Figure 3 — v1 bars | **v1-temporal** | `rq2_robustness_scale_cox.csv` |
| Figure 4 — team size as a confounder | v2-maintenance-based **and** v1-temporal | `project_level_outcomes.csv` of both |
| Descriptive comparison, active vs. dormant | v2-maintenance-based | `active_vs_failed_mw.csv` |
| Ownership concentration and Gini index | v2-maintenance-based | `truck_factor_concentration.csv` |
| Project-size control (sign inversions) | v1-temporal | `project_size_controlled_logit.csv` |
| v1/v2/v3 labels and cross-tabulation | extended/rq2_extras.py | `rq2_project_labels.csv`, `rq2_crosstab_v1_v2.csv` |
| RQ2, six factors under v1/v2/v3 with age | extended/rq2_extras.py | `rq2_six_factor_models.csv` |
| RQ2, minimum detectable effect | extended/rq2_extras.py | `rq2_minimum_detectable_effect.csv` |
| How projects fade (final window, end-of-life notices) | extended/descriptive_rq3.py | `desc_project_final_window.csv` |
| Activity trajectories | extended/descriptive_rq3.py | `desc_trajectories.csv`, `desc_trajectories_tests.csv` |
| Contrasting cases | extended/descriptive_rq3.py | `desc_contrasting_cases*.csv` |
| RQ3 — debt left behind, orphaned debt and permutation baseline | extended/descriptive_rq3.py | `rq3_debt_left_behind.csv`, `rq3_orphaned_share_by_project.csv` |
| Table 5 — RQ3, community after a core departure (event study) | extended/rq3_social.py | `rq3_social_event_study.csv`, `rq3_social_counts.csv` |
| Codebook validation samples (stratified and random) | extended/descriptive_rq3.py | `desc_codebook_validation_sample.csv`, `desc_codebook_holdout_sample.csv` |
| VIF-based variable selection | v2-maintenance-based | `vif_results.csv` |

> Some output files and column names still carry the earlier wording (`proj_failed`,
> `active_vs_failed_mw.csv`, `rq2_sensitivity_v1_cox.csv`). These are the names the pipelines
> produce and they are kept as they are, so that the file names in this table match what a
> replication run writes to disk. They refer to the dormancy outcome described above.

### 6. Reproducibility notes

* **Windows are calendar quarters within each project.** The quarterly windows of each project start
  in a project-specific month, so the panel contains 154 distinct window identifiers. Every window is
  indexed by its calendar quarter (absolute month of its start / 3), so that two consecutive windows
  of a project are always one unit apart and the four-window thresholds really last twelve months.
  An earlier version of the package ranked the 154 identifiers globally, which shortened every
  threshold to about one third of its nominal length (see the Lessons Learned of the paper).
* The models are **deterministic**. The only random step is the $k$-means clustering of the
  activity trajectories, which uses a fixed seed (`random_state=0`, 50 initializations).
* Minor differences in the last decimal digits may appear across BLAS implementations or library
  versions. The Cox models use a ridge penalization of 0.1, which makes confidence intervals and
  *p*-values approximate.
* Running a pipeline **overwrites** the committed outputs in its folder. Work on a copy if you want
  to compare a fresh run against the archived results.

---

## Dataset provenance

The sample is derived from **NICHE** (Widyasari et al., MSR 2023), a curated dataset of engineered
machine learning projects written in Python. We retained only the 441 projects labelled as
*engineered*, and further kept those whose history could be reconstructed through quarterly windows,
with more than one contributor, and for which the extraction of all metrics succeeded. This yields
the 332 projects analysed in the paper.

Metrics were extracted with **Designite** (traditional code smells), **dslinter** (ML-specific
smells), **Bandit** (security weaknesses), the **RSZZ** variant of SZZ (bug-inducing commits) and
**csDetector** (community smells and socio-technical quality factors).

---

## Citation

```bibtex
@article{developers_leave,
  title   = {Developers Leave, Projects Fade: Socio-Technical Debt, Developer Abandonment
             and Project Dormancy in Open-Source Machine Learning Projects},
  author  = {Giordano, Giammaria and Sorrentino, Pasquale and Palomba, Fabio},
  note    = {Software Engineering (SeSa) Lab, University of Salerno}
}
```

## Authors

| | Role | GitHub |
|---|---|---|
| **Giammaria Giordano** | Conceptualization, methodology, supervision | [@<handle-giordano>](https://github.com/giammariagiordano) |
| **Pasquale Sorrentino** | Formal analysis, investigation, data curation, writing | [@<handle-sorrentino>](https://github.com/PasqualeSorrentino) |
| **Fabio Palomba** | Supervision, resources, review |  |

## Contact

Corresponding author: Giammaria Giordano - `giammaria.giordano@unipegaso.it`
