import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
    scores = pl.read_parquet("results/scores_pca.parquet")
    thresholds = {
        row["detector"]: row["threshold"]
        for row in pl.read_csv("results/thresholds.csv").iter_rows(named=True)
    }
    run1 = scores.filter(
        (pl.col("faultNumber") == 1) & (pl.col("simulationRun") == 1)
    ).sort("sample")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, column in ((axes[0], "T2"), (axes[1], "SPE")):
        threshold = thresholds[column]
        ax.plot(run1["sample"], run1[column], lw=1, label=column)
        ax.axhline(
            threshold, color="red", linestyle="--",
            label=f"Validation 99th percentile ({threshold:.2f})",
        )
        ax.axvline(20, color="gray", linestyle=":", label="Fault starts after sample 20")
        ax.set_ylabel(column)
        ax.legend()
    axes[0].set_title("PCA monitor — Fault 1, simulationRun 1")
    axes[1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig("results/pca_fault1_plot.png", dpi=150)
    plt.close()
    print("Plot saved to results/pca_fault1_plot.png")


if __name__ == "__main__":
    main()