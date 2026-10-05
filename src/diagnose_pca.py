"""Rank SPE channel contributions over post-fault alarm samples."""

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from src.evaluate_pca import alarm_flags, write_pca_rows
from src.pca import (
    CHANNELS,
    DATA_DIR,
    FAULTY_FILE,
    FAULT_FREE_FILE,
    RESULTS_DIR,
    standardize,
    to_matrix,
)


def main() -> None:
    fault_free = pl.read_parquet(DATA_DIR / FAULT_FREE_FILE)
    faulty = pl.read_parquet(DATA_DIR / FAULTY_FILE)
    train = fault_free.filter(pl.col("simulationRun").is_between(1, 300))
    train_matrix = to_matrix(train)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0, ddof=1)
    train_z = standardize(train, mean, std)

    full_model = PCA(svd_solver="full").fit(train_z)
    k = int(np.argmax(np.cumsum(full_model.explained_variance_ratio_) >= 0.90) + 1)
    model = PCA(n_components=k, svd_solver="full").fit(train_z)
    faulty_z = standardize(faulty, mean, std)

    spe_threshold = (
        pl.read_csv(RESULTS_DIR / "thresholds.csv")
        .filter(pl.col("detector") == "SPE")["threshold"]
        .item()
    )
    reconstruction = model.inverse_transform(model.transform(faulty_z))
    channel_contrib = (faulty_z - reconstruction) ** 2
    contrib_frame = faulty.select(["faultNumber", "simulationRun", "sample"]).with_columns(
        pl.Series(f"c{i}", channel_contrib[:, i]) for i in range(len(CHANNELS))
    )

    totals = {}
    for (fault, _run), frame in contrib_frame.sort(["faultNumber", "simulationRun", "sample"]).group_by(
        ["faultNumber", "simulationRun"], maintain_order=True
    ):
        samples = frame["sample"].to_numpy()
        values = frame.select([f"c{i}" for i in range(len(CHANNELS))]).to_numpy()
        alarms = alarm_flags(values.sum(axis=1), spe_threshold)
        selected = alarms & (samples > 20)
        total, count = totals.get(fault, (np.zeros(len(CHANNELS)), 0))
        totals[fault] = (total + values[selected].sum(axis=0), count + int(selected.sum()))

    output_rows = []
    for fault in range(1, 21):
        total, count = totals[fault]
        if count == 0:
            continue
        mean_contrib = total / count
        for rank, index in enumerate(np.argsort(-mean_contrib, kind="stable")[:5], start=1):
            output_rows.append({
                "fault": fault,
                "detector": "SPE",
                "rank": rank,
                "channel": CHANNELS[index],
                "contribution": float(mean_contrib[index]),
            })

    contributions = pl.DataFrame(output_rows, schema={
        "fault": pl.Int64,
        "detector": pl.String,
        "rank": pl.Int64,
        "channel": pl.String,
        "contribution": pl.Float64,
    })
    write_pca_rows(RESULTS_DIR / "contributions.csv", contributions, ["fault", "detector", "rank"])
    print(pl.read_csv(RESULTS_DIR / "contributions.csv").group_by("detector").len().sort("detector"))


if __name__ == "__main__":
    main()