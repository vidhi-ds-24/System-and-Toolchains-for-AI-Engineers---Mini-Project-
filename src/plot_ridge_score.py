import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def main():
    df = pl.read_parquet("results/scores_ridge.parquet")
    run1 = df.filter(pl.col("simulationRun") == 1)
    plt.figure(figsize=(12,4))
    plt.plot(np.arange(len(run1)), run1["ridge_score"], lw=1)
    plt.title("Ridge Monitor Score — simulationRun 1 (Fault 1)")
    plt.ylabel("Score (sum squared normalized residual)")
    plt.xlabel("Sample index")
    plt.tight_layout()
    plt.savefig("results/ridge_fault1_plot.png", dpi=150)
    plt.close()
    print("Plot saved to results/ridge_fault1_plot.png")


if __name__ == "__main__":
    main()
