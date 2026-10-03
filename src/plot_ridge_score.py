import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def main():
    df = pl.read_parquet("results/scores_ridge.parquet")
    validation = df.filter(
        (pl.col("faultNumber") == 0)
        & pl.col("simulationRun").is_between(301, 400)
    )
    threshold = np.quantile(validation["score"].to_numpy(), 0.99)
    run1 = df.filter(
        (pl.col("faultNumber") == 1) & (pl.col("simulationRun") == 1)
    ).sort("sample")
    plt.figure(figsize=(12, 4))
    plt.plot(run1["sample"], run1["score"], lw=1, label="Ridge score")
    plt.axhline(
        threshold, color="red", linestyle="--",
        label=f"Validation 99th percentile ({threshold:.2f})",
    )
    plt.title("Ridge Monitor Score — simulationRun 1 (Fault 1)")
    plt.ylabel("Score (sum squared normalized residual)")
    plt.xlabel("Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ridge_fault1_plot.png", dpi=150)
    plt.close()
    print("Plot saved to results/ridge_fault1_plot.png")


if __name__ == "__main__":
    main()
