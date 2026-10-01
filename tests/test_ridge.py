import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from src.ridge import CHANNELS, KEYS, build_lagged, fit_ridge, load_model, save_model


class RidgeTests(unittest.TestCase):
    def test_lags_keep_fault_and_run_boundaries(self):
        rows = []
        for fault, run, offset in [(0, 1, 0), (0, 2, 100), (1, 1, 200)]:
            for sample in range(1, 5):
                row = dict(zip(KEYS, (fault, run, sample)))
                row.update({name: offset + 10 * sample + j for j, name in enumerate(CHANNELS)})
                rows.append(row)
        mean, std = np.full(52, 5.0), np.full(52, 2.0)
        keys, x, y = build_lagged(pl.DataFrame(rows[::-1]), mean, std)
        self.assertEqual(keys.rows(), [(0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (1, 1, 3), (1, 1, 4)])
        for i, (fault, run, sample) in enumerate(keys.rows()):
            offset = {(0, 1): 0, (0, 2): 100, (1, 1): 200}[fault, run]
            expected = lambda t: (offset + 10 * t + np.arange(52) - mean) / std
            np.testing.assert_array_equal(x[i, :52], expected(sample - 1))
            np.testing.assert_array_equal(x[i, 52:], expected(sample - 2))
            np.testing.assert_array_equal(y[i], expected(sample))

    def test_training_excludes_validation_test_and_faults(self):
        rng = np.random.default_rng(42)
        rows = []
        for fault, run in [(0, 1), (0, 2), (0, 301), (0, 401), (1, 1)]:
            for sample in range(1, 16):
                row = dict(zip(KEYS, (fault, run, sample)))
                row.update(zip(CHANNELS, rng.normal(size=52)))
                rows.append(row)
        frame = pl.DataFrame(rows)
        train = frame.filter((pl.col("faultNumber") == 0) & (pl.col("simulationRun") <= 300))
        model = fit_ridge(frame)
        other = fit_ridge(train.drop("faultNumber"))
        for name in ("mean", "std", "coef", "intercept", "residual_std"):
            np.testing.assert_array_equal(getattr(model, name), getattr(other, name))
        values = train.select(CHANNELS).to_numpy()
        np.testing.assert_allclose(model.mean, values.mean(axis=0))
        np.testing.assert_allclose(model.std, values.std(axis=0, ddof=1))
        _, x, y = build_lagged(train, model.mean, model.std)
        np.testing.assert_allclose(model.residual_std, (y - model.predict(x)).std(axis=0, ddof=1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ridge.npz"
            save_model(model, path)
            restored = load_model(path)
            np.testing.assert_array_equal(restored.predict(x), model.predict(x))
            np.testing.assert_array_equal(restored.residual_std, model.residual_std)

    def test_missing_or_duplicate_samples_are_rejected(self):
        for samples in ([1, 2, 4], [1, 2, 2]):
            frame = pl.DataFrame({
                "simulationRun": [1] * 3,
                "sample": samples,
                **{name: [1.0, 2.0, 3.0] for name in CHANNELS},
            })
            with self.assertRaisesRegex(ValueError, "consecutive, unique"):
                build_lagged(frame, np.zeros(52), np.ones(52))


if __name__ == "__main__":
    unittest.main()
