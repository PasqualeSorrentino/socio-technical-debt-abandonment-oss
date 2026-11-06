# Developers Leave, Projects Fade

### On the Relationship between Socio-Technical Debt and Abandonment in Open-Source Machine Learning Projects

Replication package for the paper by **Pasquale Sorrentino, Giammaria Giordano and Fabio Palomba** -
Software Engineering (SeSa) Lab, Department of Computer Science, University of Salerno, Italy.

---

## Abstract

**Context.** Open-source software is a load-bearing component of the digital infrastructure, yet its
continuity depends on the voluntary participation of contributors. Many projects decline over time
until they are abandoned. The causes are not only technical. They emerge from the interplay between
the code and the community that develops it, i.e., from the *socio-technical debt* that combines
social debt and technical debt.

**Objective.** While social and technical debt have been shown to be coupled in terms of code
quality, the extent to which they relate to the abandonment of developers and of entire projects has
not been systematically assessed. This paper investigates whether socio-technical debt is associated
with abandonment at these two levels of granularity, and how much the answer depends on the way
failure is defined.

**Method.** We analyze **332 engineered open-source ML projects** from the NICHE dataset, involving
**10,416 developers** observed over quarterly windows. We model developer abandonment and project
inactivity as right-censored time-to-event processes, combining logistic regression and
(time-varying) Cox models. At the project level, we compare two operationalizations of failure, i.e.,
a temporal one and one based on recent maintenance intensity, while controlling for project scale.

**Result.** At the developer level, social debt indicators, and in particular team turnover and the
share of non-core contributors, are consistently associated with a higher risk of abandonment,
although part of this association is expected by construction. Technical debt shows no association.
At the project level, the effects observed under the temporal definition vanish under the
maintenance-based one once scale is controlled. We found no consistent evidence of a socio-technical
effect, while scale and continuity indicators carry most of the signal.

**Conclusion.** Project abandonment does not appear to be the sum of individual abandonments or a
direct product of socio-technical debt, and conclusions on project failure are sensitive to how
failure is defined. We conclude by distilling lessons learned, implications, and future research
directions.

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
    ├── v2-maintenance-based/          # MAIN analysis -> results reported in the paper
    │   ├── pipeline_v2_maintenance_based.py
    │   └── outputs/file/
    └── v1-temporal/                   # SENSITIVITY analysis -> v1 term of Figure 3
        ├── pipeline_v1_temporal.py
        └── outputs/file/
```

Each pipeline is standalone, locates the dataset by walking up the directory tree, and writes every
table (`.csv`) and figure (`.png`) it produces into its own `outputs/file/` directory.

---

## The two analyses, and what each one produces

The paper compares two operationalizations of project failure, and this package contains one
pipeline for each. The two are **not** interchangeable: they classify projects differently and
therefore yield different RQ2 results. Both are required to reproduce the paper.

| Pipeline | Definition of project failure | Split | Role |
|---|---|---|---|
| **`v2-maintenance-based`** | Fewer than 12 commits in the four windows ending with the project's **most recent commit** | **162 active / 170 inactive** | **Main analysis.** Produces Tables 2–3 and the v2 bars of Figures 3-4 |
| **`v1-temporal`** | Last observable activity precedes the end of the observation period by at least four windows | 38 active / 294 inactive | **Sensitivity analysis.** Produces the v1 bars of Figures 3-4 |

**Figure 3 combines the v2 estimates from `v2-maintenance-based` with the v1 estimates from
`v1-temporal`**, and Figure 4 contrasts the two project splits: neither figure can be reproduced
from a single pipeline.

> **A note on the scale controls.** Under v1 the third structural control is the number of commits
> per window; under v2 it is the number of observed windows. The substitution is deliberate: commit
> intensity is what *defines* the v2 outcome, so using it as a control there would introduce leakage.
> The two columns of Figure 3 are therefore adjusted for slightly different control sets.

**RQ1 does not depend on the project-failure definition.** The developer-level outcome is identical
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
# Main analysis (v2) - everything reported in the paper except the v1 bars
python analysis/v2-maintenance-based/pipeline_v2_maintenance_based.py

# Sensitivity analysis (v1) - needed for Figures 3 and 4
python analysis/v1-temporal/pipeline_v1_temporal.py
```

Loading and parsing the 913 MB panel dominates the start-up time; each pipeline then runs the full
sequence of descriptive, regression and survival analyses and regenerates its output folder.

### 5. Where each result of the paper comes from

Paths are relative to `<pipeline>/outputs/file/`.

| Paper item | Pipeline | File |
|---|---|---|
| Table 2 — RQ1, Cox model (later stage) | v2-maintenance-based | `later_stage_cox_results.csv` |
| RQ1, early-stage logistic regression | v2-maintenance-based | `early_stage_logit_results.csv` |
| RQ1, time-varying Cox | v2-maintenance-based | `timevarying_cox.csv` |
| RQ1, interaction terms | v2-maintenance-based | `cox_interactions_results.csv` |
| Figure 2 — pre-abandonment signature | v2-maintenance-based | `prefailure_signature.csv` |
| Table 3 — reference logit | v2-maintenance-based | `rq2_project_logit_multivariate.csv` |
| Table 3 — scale-adjusted logit and Cox | v2-maintenance-based | `rq2_robustness_scale.csv` |
| RQ2, time-varying Cox | v2-maintenance-based | `rq2_project_cox_timevarying.csv` |
| Figure 3 — v2 bars | v2-maintenance-based | `rq2_robustness_scale.csv` |
| Figure 3 — v1 bars | **v1-temporal** | `rq2_robustness_scale_cox.csv` |
| Figure 4 — team size as a confounder | v2-maintenance-based **and** v1-temporal | `project_level_outcomes.csv` of both |
| Descriptive comparison, active vs. inactive | v2-maintenance-based | `active_vs_failed_mw.csv` |
| Ownership concentration and Gini index | v2-maintenance-based | `truck_factor_concentration.csv` |
| Project-size control (sign inversions) | v2-maintenance-based | `project_size_controlled_logit.csv` |
| VIF-based variable selection | v2-maintenance-based | `vif_results.csv` |

### 6. Reproducibility notes

* The pipelines are **deterministic**: no random seed is involved, so repeated runs on the same
  dataset return identical coefficients.
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
@article{palomba_developers_leave,
  title   = {Developers Leave, Projects Fade: On the Relationship between Socio-Technical
             Debt and Abandonment in Open-Source Machine Learning Projects},
  author  = {Sorrentino, Pasquale and Giordano, Giammaria and Palomba, Fabio},
  note    = {Software Engineering (SeSa) Lab, University of Salerno}
}
```

## License

TBD.

## Contact

Corresponding author: Giammaria Giordano - `giagiordano@unisa.it`
