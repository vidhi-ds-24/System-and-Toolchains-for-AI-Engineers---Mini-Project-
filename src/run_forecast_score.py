"""
Forecast monitor scoring: compute the Ridge anomaly score.
Produce results/scores_ridge.parquet
"""
from pathlib import Path
import polars as pl
import numpy as np
from src.ridge import load_model, build_lagged

def main():
    # Load the pre‑trained ridge model
    model_path = Path("results/ridge_model.npz")
    model = load_model(model_path)

    # Score normal validation/test runs and the required faulty runs only.
    df_faultfree = pl.read_parquet("data/tep_fault_free_training.parquet").filter(
        (pl.col("faultNumber") == 0)
        & pl.col("simulationRun").is_between(301, 500)
    )
    df_faulty = pl.read_parquet("data/tep_faulty_training_runs01-20.parquet").filter(
        pl.col("faultNumber").is_between(1, 20)
        & pl.col("simulationRun").is_between(1, 20)
    )
    df_all = pl.concat([df_faultfree, df_faulty])

    # build lagged features & targets; automatically drops first‑two samples per run
    keys_df, X, y_true_std = build_lagged(df_all, model.mean, model.std)

    # model predict (predict standardized targets)
    y_pred_std = model.predict(X)

    # residual in standardized space
    residual = y_true_std - y_pred_std

    # per‑channel normalized residual: divide by training‑set residual_std
    norm_residual = residual / model.residual_std[None, :]

    # total score = sum over 52 channels of squared normalized residual
    total_score = np.sum(np.square(norm_residual), axis=1)

    # Write the four columns required by the evidence script.
    scores_df = keys_df.with_columns(pl.Series("score", total_score))
    out_path = Path("results/scores_ridge.parquet")
    scores_df.write_parquet(out_path)
    print(f"Scoring complete. Saved {scores_df.height:,} rows to {out_path}")

if __name__ == "__main__":
    main()
