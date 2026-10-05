"""Rank Ridge channel contributions over post-fault alarm samples."""

import numpy as np
import polars as pl

from src.evaluate_ridge import ROOT, alarm_flags, write_ridge_rows
from src.ridge import CHANNELS, KEYS, build_lagged, load_model


def main() -> None:
    results = ROOT / "results"
    model = load_model(results / "ridge_model.npz")
    faulty = pl.read_parquet(
        ROOT / "data" / "tep_faulty_training_runs01-20.parquet"
    ).filter(
        pl.col("faultNumber").is_between(1, 20)
        & pl.col("simulationRun").is_between(1, 20)
    )
    keys, features, targets = build_lagged(faulty, model.mean, model.std)
    contributions = ((targets - model.predict(features)) / model.residual_std) ** 2

    scores = pl.read_parquet(results / "scores_ridge.parquet")
    aligned = (
        keys.with_row_index("_row")
        .join(scores, on=KEYS, how="left", validate="1:1")
        .sort(KEYS)
    )
    if aligned["score"].null_count():
        raise ValueError("Ridge scores are missing faulty samples; run make ridge-score")
    if not np.allclose(
        aligned["score"].to_numpy(),
        contributions[aligned["_row"].to_numpy()].sum(axis=1),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("Saved scores do not match the Ridge model; run make ridge-score")
    threshold = (
        pl.read_csv(results / "thresholds.csv")
        .filter(pl.col("detector") == "ridge")["threshold"]
        .item()
    )

    rows = []
    for fault_rows in aligned.partition_by("faultNumber", maintain_order=True):
        total = np.zeros(len(CHANNELS))
        count = 0
        for run in fault_rows.partition_by("simulationRun", maintain_order=True):
            # Determine alarms before filtering the post-fault samples.
            selected = alarm_flags(run["score"].to_numpy(), threshold)
            selected &= run["sample"].to_numpy() > 20
            indices = run["_row"].to_numpy()[selected]
            total += contributions[indices].sum(axis=0)
            count += len(indices)
        if count == 0:
            continue
        mean = total / count
        for rank, channel in enumerate(np.argsort(-mean, kind="stable")[:5], start=1):
            rows.append({
                "fault": int(fault_rows["faultNumber"][0]),
                "detector": "ridge",
                "rank": rank,
                "channel": CHANNELS[channel],
                "contribution": float(mean[channel]),
            })

    output = pl.DataFrame(rows, schema={
        "fault": pl.Int64,
        "detector": pl.String,
        "rank": pl.Int64,
        "channel": pl.String,
        "contribution": pl.Float64,
    })
    write_ridge_rows(results / "contributions.csv", output, ["fault", "detector", "rank"])
    print(f"Wrote {output.height} Ridge rows to results/contributions.csv")


if __name__ == "__main__":
    main()
