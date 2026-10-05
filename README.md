# Miniproject

06-763 — Systems and Toolchains for AI Engineers, Fall 2026.

[Project requirements](https://kitchingroup.cheme.cmu.edu/f26-06763/course/miniproject.html)

```text
data/       Datasets
src/        Source code
results/    Output files
```

```sh
make setup        # Create miniproj (.venv), lock and install dependencies
make data         # Download the official datasets and verify checksums
make pca          # Generate PCA scores; also runs make data
make ridge        # Train Ridge and save results/ridge_model.npz
make ridge-score  # Generate results/scores_ridge.parquet
make ridge-plot   # Generate results/ridge_fault1_plot.png
```

After all required result files are ready, generate the evidence PDF:

```sh
make evidence IDS="id1 id2 id3 id4" TEAM="Team name"
```
