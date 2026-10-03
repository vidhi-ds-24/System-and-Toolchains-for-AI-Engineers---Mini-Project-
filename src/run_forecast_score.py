"""
Forecast monitor scoring: compute Ridge‑based anomaly score and per‑channel contributions.
Produce results/scores_ridge.parquet
"""
from pathlib import Path
import polars as pl
import numpy as np
from src.ridge import load_model, build_lagged, KEYS, CHANNELS

def main():
    # Load the pre‑trained ridge model
    model_path = Path("results/ridge_model.npz")
    model = load_model(model_path)

    # Load full dataset (fault‑free + faulty training runs 01‑20)
    df_faultfree = pl.read_parquet("data/tep_fault_free_training.parquet")
    df_faulty = pl.read_parquet("data/tep_faulty_training_runs01-20.parquet")
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

    # per‑channel contribution: squared normalized residual
    contrib_schema = [f"contrib_{ch}" for ch in CHANNELS]
    contrib_df = pl.DataFrame(
        np.square(norm_residual),
        schema=contrib_schema
    )
    contrib_df = pl.concat([keys_df, contrib_df], how="horizontal_extend")
    contrib_df = contrib_df.with_columns(pl.Series("ridge_score", total_score))

    # Write output parquet for group project
    out_path = Path("results/scores_ridge.parquet")
    contrib_df.write_parquet(out_path)
    print(f"Scoring complete. Saved scores + per‑channel contributions to {out_path}")

if __name__ == "__main__":
    main()

