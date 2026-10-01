"""Train the Ridge forecaster and provide lagged inputs for the scoring stage."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

CHANNELS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
KEYS = ["faultNumber", "simulationRun", "sample"]


@dataclass
class RidgeModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    residual_std: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict standardized targets from standardized lag-1/lag-2 features."""
        return features @ self.coef.T + self.intercept


def prepare_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Normal files may omit faultNumber; preserve keys and numeric channel order."""
    if "faultNumber" not in frame.columns:
        frame = frame.with_columns(pl.lit(0).alias("faultNumber"))
    frame = frame.select(KEYS + CHANNELS).sort(KEYS)
    if frame.select(KEYS).null_count().to_numpy().any():
        raise ValueError("Row keys must not contain nulls")
    if not np.isfinite(frame.select(CHANNELS).to_numpy()).all():
        raise ValueError("Channel values must be finite")
    return frame


def build_lagged(
    frame: pl.DataFrame, mean: np.ndarray, std: np.ndarray
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return aligned (keys, X, y), with X=[z(t-1), z(t-2)] and y=z(t).

    Supply raw channel values and the training mean/std. The first two samples
    of each (faultNumber, simulationRun) are omitted. Rows are sorted internally.
    """
    frame = prepare_frame(frame)
    if mean.shape != (52,) or std.shape != (52,):
        raise ValueError("Mean and standard deviation must each have 52 entries")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Standardization parameters must be finite, with positive std")

    keys, features, targets = [], [], []
    for run in frame.partition_by(KEYS[:2], maintain_order=True):
        if np.any(np.diff(run["sample"].to_numpy()) != 1):
            raise ValueError("Each run must have consecutive, unique sample numbers")
        if len(run) < 3:
            continue
        z = (run.select(CHANNELS).to_numpy() - mean) / std
        keys.append(run.select(KEYS).slice(2))
        features.append(np.hstack([z[1:-1], z[:-2]]))
        targets.append(z[2:])

    if not keys:
        return frame.select(KEYS).head(0), np.empty((0, 104)), np.empty((0, 52))
    return pl.concat(keys), np.vstack(features), np.vstack(targets)


def fit_ridge(frame: pl.DataFrame) -> RidgeModel:
    """Fit only normal runs 1–300, even if other runs are present in the input."""
    frame = prepare_frame(frame)
    train = frame.filter(
        (pl.col("faultNumber") == 0) & pl.col("simulationRun").is_between(1, 300)
    )
    if train.is_empty():
        raise ValueError("No normal training rows from runs 1–300")

    values = train.select(CHANNELS).to_numpy()
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    _, features, targets = build_lagged(train, mean, std)
    if len(features) < 2:
        raise ValueError("At least two lagged training rows are required")

    estimator = Ridge(alpha=1.0).fit(features, targets)
    residual_std = (targets - estimator.predict(features)).std(axis=0, ddof=1)
    if not np.isfinite(residual_std).all() or (residual_std <= 0).any():
        raise ValueError("Training residual standard deviations must be positive and finite")
    return RidgeModel(mean, std, estimator.coef_, estimator.intercept_, residual_std)


def save_model(model: RidgeModel, path: str | Path) -> None:
    """Save coefficients and all training scales together, without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            channels=np.array(CHANNELS),
            mean=model.mean,
            std=model.std,
            coef=model.coef,
            intercept=model.intercept,
            residual_std=model.residual_std,
        )


def load_model(path: str | Path) -> RidgeModel:
    with np.load(path, allow_pickle=False) as saved:
        if saved["channels"].tolist() != CHANNELS:
            raise ValueError("Saved channel order does not match CHANNELS")
        return RidgeModel(*(saved[name] for name in (
            "mean", "std", "coef", "intercept", "residual_std"
        )))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tep_fault_free_training.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/ridge_model.npz"))
    args = parser.parse_args()
    model = fit_ridge(pl.read_parquet(args.data))
    save_model(model, args.output)
    print("Fitted Ridge(alpha=1.0) on normal runs 1–300: 104 inputs, 52 targets")
    print(f"Saved model, standardization, and residual scales to {args.output}")


if __name__ == "__main__":
    main()
