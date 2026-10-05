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
make pca-evaluate # Write T2/SPE rows to results/thresholds.csv and results/detection.csv
make pca-diagnose # Run pca-evaluate, then write SPE rows to results/contributions.csv
make pca-plot     # Run pca-evaluate, then generate results/pca_fault1_plot.png
make ridge        # Train Ridge and save results/ridge_model.npz
make ridge-score  # Generate results/scores_ridge.parquet
make ridge-plot   # Generate results/ridge_fault1_plot.png
make ridge-evaluate  # Write Ridge rows to results/thresholds.csv and results/detection.csv
make ridge-diagnose  # Run ridge-evaluate, then write Ridge rows to results/contributions.csv
```

After all required result files are ready, generate the evidence PDF:

```sh
make evidence IDS="id1 id2 id3 id4" TEAM="Team name"
```
