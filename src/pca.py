"""PCA fault detection on the Tennessee Eastman Process.

Script version of notebooks/PCA_v0.1.ipynb. Run from the project root:

    uv run python src/pca.py

Steps: verify the data, split the fault-free runs, standardize with training
statistics, pick k for 90% variance, fit PCA, score T2 and SPE, and write
results/scores_pca.parquet.

The Ridge model can reuse the preprocessing with:

    from pca import CHANNELS, to_matrix, standardize
"""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FAULT_FREE_FILE = "tep_fault_free_training.parquet"
FAULTY_FILE = "tep_faulty_training_runs01-20.parquet"

# 41 measurements, then 11 manipulated variables. The Ridge pair must use the same order.
CHANNELS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

# Expected shape of each file: row count and the (min, max) of each identifier column.
EXPECTED = {
    FAULT_FREE_FILE: {
        "rows": 250_000,
        "simulationRun": (1, 500),
        "sample": (1, 500),
    },
    FAULTY_FILE: {
        "rows": 200_000,
        "faultNumber": (1, 20),
        "simulationRun": (1, 20),
        "sample": (1, 500),
    },
}

VARIANCE_TARGET = 0.90


def verify_data() -> None:
    # Official checksum file from the assignment. The evidence script rejects files that do not match it.
    sums_path = DATA_DIR / "SHA256SUMS"
    if not sums_path.exists():
        raise SystemExit("data/SHA256SUMS is missing. Run `make data` first.")
    expected_hashes = {}
    for line in sums_path.read_text().splitlines():
        digest, name = line.split()
        expected_hashes[name.removeprefix("*")] = digest

    problems = []
    for name, checks in EXPECTED.items():
        path = DATA_DIR / name
        print(name)
        if not path.exists():
            problems.append(f"{name} is missing")
            continue

        # Hash the file the same way the assignment's shasum check does.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest == expected_hashes.get(name)
        print("  checksum", "OK" if ok else "DOES NOT MATCH")
        if not ok:
            problems.append(f"{name} checksum does not match")

        frame = pl.read_parquet(path)
        ok = frame.height == checks["rows"]
        print("  rows", frame.height, "OK" if ok else f"expected {checks['rows']}")
        if not ok:
            problems.append(f"{name} has {frame.height} rows")

        missing = [column for column in CHANNELS if column not in frame.columns]
        print("  52 channels", "OK" if not missing else f"missing {missing}")
        if missing:
            problems.append(f"{name} is missing channels {missing}")

        for column, bounds in checks.items():
            # "rows" was already checked above. It is one number, not a (low, high) pair.
            if column == "rows":
                continue
            if column not in frame.columns:
                problems.append(f"{name} is missing column {column}")
                continue
            seen = (frame[column].min(), frame[column].max())
            print(f"  {column}", seen, "OK" if seen == bounds else f"expected {bounds}")
            if seen != bounds:
                problems.append(f"{name} {column} range is {seen}")

    if problems:
        raise SystemExit("Data check failed:\n  " + "\n  ".join(problems))


def to_matrix(frame: pl.DataFrame) -> np.ndarray:
    # select(CHANNELS) locks the column order. PCA needs a numeric array, one row per snapshot.
    return frame.select(CHANNELS).to_numpy().astype(np.float64)


def standardize(frame: pl.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # z = (x - mean) / std, always with the training mean and standard deviation.
    # Recomputing them on new rows would center the new data on itself and hide the shift we want to detect.
    return (to_matrix(frame) - mean) / std


def score_block(model: PCA, standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # transform projects each standardized snapshot onto the k directions: t = P.T @ z.
    scores = model.transform(standardized)
    # explained_variance_ is lambda_i. Dividing by it makes a quiet direction count more than a noisy one.
    t2 = np.sum(scores**2 / model.explained_variance_, axis=1)
    # inverse_transform(scores) is P @ P.T @ z, the reconstruction named in the assignment.
    reconstruction = model.inverse_transform(scores)
    # SPE is the squared length of the part those k directions could not rebuild.
    spe = np.sum((standardized - reconstruction) ** 2, axis=1)
    return t2, spe


def pack_scores(frame: pl.DataFrame, t2: np.ndarray, spe: np.ndarray, fault_number: int | None) -> pl.DataFrame:
    # Fault-free blocks are stored with faultNumber 0. The faulty block keeps its own faultNumber.
    if fault_number is None:
        faults = frame.get_column("faultNumber")
    else:
        faults = pl.Series("faultNumber", [fault_number] * frame.height)
    return pl.DataFrame(
        {
            "faultNumber": faults,
            "simulationRun": frame.get_column("simulationRun"),
            "sample": frame.get_column("sample"),
            "T2": t2,
            "SPE": spe,
        }
    )


def main() -> None:
    verify_data()

    fault_free = pl.read_parquet(DATA_DIR / FAULT_FREE_FILE)
    faulty = pl.read_parquet(DATA_DIR / FAULTY_FILE)

    # Runs 1-300: train. The mean, standard deviation, and PCA fit use only these runs.
    # Runs 301-400: validation. These scores are used later to set the alarm threshold.
    # Runs 401-500: test. These scores are used later to measure false alarms.
    train = fault_free.filter(pl.col("simulationRun").is_between(1, 300))
    validation = fault_free.filter(pl.col("simulationRun").is_between(301, 400))
    test = fault_free.filter(pl.col("simulationRun").is_between(401, 500))
    print("\nrows: train", train.height, "validation", validation.height, "test", test.height, "faulty", faulty.height)

    train_matrix = to_matrix(train)
    mean = train_matrix.mean(axis=0)
    # ddof=1 divides by n - 1. The assignment requires that sample standard deviation.
    std = train_matrix.std(axis=0, ddof=1)
    train_z = standardize(train, mean, std)

    # Fit all 52 components only to read every eigenvalue.
    # svd_solver="full" is exact. The evidence check allows 0.1% error, and the randomized solver is an approximation.
    full_model = PCA(svd_solver="full").fit(train_z)
    cumulative = np.cumsum(full_model.explained_variance_ratio_)
    # Smallest k whose eigenvalues add up to at least 90%.
    k = int(np.argmax(cumulative >= VARIANCE_TARGET) + 1)
    print("k =", k)
    print("variance kept =", round(float(cumulative[k - 1]), 6))

    # The monitoring model keeps k components. Keeping all 52 would rebuild every snapshot exactly, and SPE would be 0.
    model = PCA(n_components=k, svd_solver="full").fit(train_z)

    # Training rows are omitted. The score file contains validation, test, and every faulty row.
    scored_pieces = []
    for frame, fault_number in ((validation, 0), (test, 0), (faulty, None)):
        t2, spe = score_block(model, standardize(frame, mean, std))
        scored_pieces.append(pack_scores(frame, t2, spe, fault_number))
    scores = pl.concat(scored_pieces)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "scores_pca.parquet"
    scores.write_parquet(out_path)
    print("wrote", scores.height, "rows to", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
