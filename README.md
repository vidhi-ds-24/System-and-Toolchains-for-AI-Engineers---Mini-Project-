# Miniproject

06-763 — Systems and Toolchains for AI Engineers, Fall 2026.

[Project requirements](https://kitchingroup.cheme.cmu.edu/f26-06763/course/miniproject.html)

```text
data/       Datasets
src/        Source code
results/    Output files
```

```sh
make setup
make data
make evidence IDS="id1 id2 id3 id4" TEAM="Team name"
```

The virtual environment is `.venv`; activating it displays `(miniproj)`.

## Ridge Training

```sh
make ridge
```

Training uses normal runs 1–300. Each channel is standardized with the mean and
sample standard deviation (`ddof=1`) of all training rows. Within each run,
the previous two standardized rows form 104 features to predict the current
52 channels using `Ridge(alpha=1.0)`. The first two samples have no prediction.
Residual standard deviations are calculated on the lagged training rows.
The model and all scales are saved together in `results/ridge_model.npz`.

The scoring teammate can load this artifact and construct aligned inputs:

```python
from src.ridge import build_lagged, load_model

model = load_model("results/ridge_model.npz")
keys, X, y = build_lagged(raw_frame, model.mean, model.std)
predictions = model.predict(X)  # y and predictions are both standardized
residual_std = model.residual_std
```

For scoring, select normal runs 301–500 and faults 1–20, runs 1–20.
`raw_frame` contains unstandardized channel values and row keys. `keys` stays
aligned with `X`, `y`, and predictions. Channel order is `xmeas_1`–`xmeas_41`,
then `xmv_1`–`xmv_11`. Scoring, contributions, and evaluation are separate tasks.

AI assistance: Codex helped implement and test the Ridge training and handoff code;
include actual AI use in the final report.
