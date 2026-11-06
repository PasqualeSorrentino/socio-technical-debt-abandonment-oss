# Dataset

`mining.csv` is the longitudinal panel used by all pipelines: **24,916 observations**
(developer x project x quarterly window), 76 raw attributes, **913 MB**.

## The file is tracked with Git LFS

The 134-byte file you may see here is a **pointer**, not the data.
**Downloading the repository as a ZIP from the GitHub web interface does not resolve
LFS pointers.** To obtain the actual dataset you must clone:

```bash
git lfs install
git clone <REPOSITORY-URL>
cd <REPOSITORY-NAME>
git lfs pull
```

Verify before running any pipeline:

```bash
ls -l analysis/dataset/mining.csv       # expected: ~913 MB, not 134 bytes
head -c 60 analysis/dataset/mining.csv
```

If the first line reads `version https://git-lfs.github.com/spec/v1`, the pointer has not
been resolved and `git lfs pull` has not run correctly.
