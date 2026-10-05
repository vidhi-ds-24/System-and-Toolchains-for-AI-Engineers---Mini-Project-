"""Evaluate Ridge scores while preserving other detectors' result rows."""

from pathlib import Path

import numpy as np
import polars as pl

from src.ridge import KEYS

ROOT = Path(__file__).resolve().parent.parent


def alarm_flags(values: np.ndarray, threshold: float) -> np.ndarray:
    """Require the current and previous two scores in one run to exceed threshold."""
    above = np.asarray(values) > threshold
    alarms = np.zeros(len(above), dtype=bool)
    if len(above) >= 3:
        alarms[2:] = above[2:] & above[1:-1] & above[:-2]
    return alarms


def write_ridge_rows(
    path: Path, rows: pl.DataFrame, sort_columns: list[str]
) -> None:
    """Replace Ridge rows in a shared CSV, keeping existing other-detector rows."""
    if path.exists():
        existing = pl.read_csv(path, schema_overrides=rows.schema)
        existing = existing.filter(pl.col("detector") != "ridge")
        rows = pl.concat([existing.select(rows.columns), rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(sort_columns).write_csv(path)


def main() -> None:
    results_dir = ROOT / "results"
    scores = pl.read_parquet(results_dir / "scores_ridge.parquet").select(
        KEYS + ["score"]
    ).sort(KEYS)
    expected_runs = {(0, run) for run in range(301, 501)} | {
        (fault, run) for fault in range(1, 21) for run in range(1, 21)
    }
    if set(scores.select(KEYS[:2]).unique().iter_rows()) != expected_runs:
        raise ValueError("Ridge scores must cover normal runs 301–500 and faults 1–20, runs 1–20")
    if not np.isfinite(scores["score"].to_numpy()).all():
        raise ValueError("Ridge scores must be finite")

    validation = scores.filter(
        (pl.col("faultNumber") == 0)
        & pl.col("simulationRun").is_between(301, 400)
    )
    threshold = float(np.quantile(validation["score"].to_numpy(), 0.99))
    rates = {fault: [] for fault in range(1, 21)}
    delays = {fault: [] for fault in range(1, 21)}
    false_alarms = 0
    test_samples = 0

    for (fault, run), frame in scores.group_by(KEYS[:2], maintain_order=True):
        samples = frame["sample"].to_numpy()
        if not np.array_equal(samples, np.arange(3, 501)):
            raise ValueError(f"Fault {fault}, run {run} must have Ridge samples 3–500 exactly once")
        alarms = alarm_flags(frame["score"].to_numpy(), threshold)
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
        "detector": "ridge",
        "detection_rate": false_alarms / test_samples,
        "median_delay_min": None,
        "runs_missed": None,
    }]
    for fault in range(1, 21):
        rows.append({
            "fault": fault,
            "detector": "ridge",
            "detection_rate": float(np.mean(rates[fault])),
            "median_delay_min": float(np.median(delays[fault])) if delays[fault] else None,
            "runs_missed": len(rates[fault]) - len(delays[fault]),
        })

    thresholds = pl.DataFrame({"detector": ["ridge"], "threshold": [threshold]})
    detection = pl.DataFrame(rows, schema={
        "fault": pl.Int64,
        "detector": pl.String,
        "detection_rate": pl.Float64,
        "median_delay_min": pl.Float64,
        "runs_missed": pl.Int64,
    })
    write_ridge_rows(results_dir / "thresholds.csv", thresholds, ["detector"])
    write_ridge_rows(results_dir / "detection.csv", detection, ["fault", "detector"])
    print(f"Updated Ridge threshold in results/thresholds.csv: {threshold:.6f}")
    print("Updated 21 Ridge rows in results/detection.csv")


if __name__ == "__main__":
    main()
