"""Evaluate PCA scores while preserving other detectors' result rows."""

from pathlib import Path

import numpy as np
import polars as pl

from src.pca import RESULTS_DIR


def alarm_flags(values: np.ndarray, threshold: float) -> np.ndarray:
    # A point alarms only when it and the two points before it are all strictly above the line.
    # The window is computed inside one already-sorted run, so it cannot cross into another run.
    above = np.asarray(values) > threshold
    alarms = np.zeros(len(above), dtype=bool)
    if len(above) >= 3:
        alarms[2:] = above[2:] & above[1:-1] & above[:-2]
    return alarms


def write_pca_rows(path: Path, rows: pl.DataFrame, sort_columns: list[str]) -> None:
    # Drop only the PCA detectors this call is about to rewrite. Ridge rows stay.
    pca_detectors = rows["detector"].unique().to_list()
    if path.exists():
        existing = pl.read_csv(path, schema_overrides=rows.schema)
        kept = existing.filter(~pl.col("detector").is_in(pca_detectors))
        rows = pl.concat([kept.select(rows.columns), rows], how="vertical")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(sort_columns).write_csv(path)


def detection_rows(scores: pl.DataFrame, column: str, threshold: float, detector: str) -> list[dict]:
    rates = {fault: [] for fault in range(1, 21)}
    delays = {fault: [] for fault in range(1, 21)}
    false_alarms = 0
    test_samples = 0

    ordered = scores.sort(["faultNumber", "simulationRun", "sample"])
    for (fault, run), frame in ordered.group_by(["faultNumber", "simulationRun"], maintain_order=True):
        samples = frame["sample"].to_numpy()
        if not np.array_equal(samples, np.arange(1, 501)):
            raise ValueError(f"Fault {fault}, run {run} must have samples 1–500 exactly once")
        alarms = alarm_flags(frame[column].to_numpy(), threshold)
        if fault == 0:
            if 401 <= run <= 500:
                false_alarms += int(alarms.sum())
                test_samples += len(alarms)
            continue
        post_onset = samples > 20
        rates[fault].append(float(alarms[post_onset].mean()))
        hits = np.flatnonzero(alarms & post_onset)
        if len(hits):
            delays[fault].append(float((samples[hits[0]] - 20) * 3))

    rows = [{
        "fault": 0,
        "detector": detector,
        "detection_rate": false_alarms / test_samples,
        "median_delay_min": None,
        "runs_missed": None,
    }]
    for fault in range(1, 21):
        rows.append({
            "fault": fault,
            "detector": detector,
            "detection_rate": float(np.mean(rates[fault])),
            "median_delay_min": float(np.median(delays[fault])) if delays[fault] else None,
            "runs_missed": len(rates[fault]) - len(delays[fault]),
        })
    return rows


def main() -> None:
    scores = pl.read_parquet(RESULTS_DIR / "scores_pca.parquet")
    validation_scores = scores.filter(
        (pl.col("faultNumber") == 0) & pl.col("simulationRun").is_between(301, 400)
    )
    thresholds = pl.DataFrame({
        "detector": ["T2", "SPE"],
        "threshold": [
            float(np.quantile(validation_scores["T2"].to_numpy(), 0.99)),
            float(np.quantile(validation_scores["SPE"].to_numpy(), 0.99)),
        ],
    })
    write_pca_rows(RESULTS_DIR / "thresholds.csv", thresholds, ["detector"])

    threshold_lookup = dict(zip(thresholds["detector"].to_list(), thresholds["threshold"].to_list()))
    detection = pl.DataFrame(
        detection_rows(scores, "T2", threshold_lookup["T2"], "T2")
        + detection_rows(scores, "SPE", threshold_lookup["SPE"], "SPE"),
        schema={
            "fault": pl.Int64,
            "detector": pl.String,
            "detection_rate": pl.Float64,
            "median_delay_min": pl.Float64,
            "runs_missed": pl.Int64,
        },
    )
    write_pca_rows(RESULTS_DIR / "detection.csv", detection, ["fault", "detector"])
    print(pl.read_csv(RESULTS_DIR / "thresholds.csv"))
    print(pl.read_csv(RESULTS_DIR / "detection.csv").group_by("detector").len().sort("detector"))


if __name__ == "__main__":
    main()