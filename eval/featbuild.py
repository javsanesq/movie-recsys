"""Build (user x candidate) feature matrix. Shared across train/eval/serve so
features are identical everywhere (no train/serve skew)."""
import numpy as np, polars as pl
from config import GENRES

GENRE_COLS = [f"genre_{g}" for g in GENRES]
AFF_COLS = [f"genre_affinity_{g}" for g in GENRES]

def feature_columns(items_df: pl.DataFrame) -> list[str]:
    item_cols = (["n_ratings", "avg_rating", "std_rating", "release_year"]
                 + GENRE_COLS + [c for c in items_df.columns if c.startswith("genome_")])
    user_cols = (["u_n_ratings", "u_avg_rating", "u_std_rating",
                  "days_since_last", "pct_high_ratings"] + AFF_COLS)
    cross_cols = ["genre_match", "rating_delta", "popularity_rank", "user_item_era_diff"]
    return item_cols + user_cols + cross_cols

def build_matrix(user_row: dict, cand_movieids: np.ndarray,
                 items_df: pl.DataFrame, max_item_ratings: int) -> tuple[np.ndarray, list[str]]:
    cands = items_df.filter(pl.col("movieId").is_in(cand_movieids.tolist()))
    order = pl.DataFrame({"movieId": cand_movieids, "_o": np.arange(len(cand_movieids))})
    cands = cands.join(order, on="movieId", how="inner").sort("_o").drop("_o")

    aff = np.array([user_row.get(c, 0.0) for c in AFF_COLS], dtype=np.float32)
    genre_mat = cands.select(GENRE_COLS).to_numpy().astype(np.float32)
    genre_match = genre_mat @ aff

    n = cands.height
    feat = {
        "n_ratings": cands["n_ratings"].to_numpy(),
        "avg_rating": cands["avg_rating"].to_numpy(),
        "std_rating": cands["std_rating"].to_numpy(),
        "release_year": cands["release_year"].fill_null(0).to_numpy(),
    }
    for c in GENRE_COLS:
        feat[c] = cands[c].to_numpy()
    for c in cands.columns:
        if c.startswith("genome_"):
            feat[c] = cands[c].to_numpy()
    feat["u_n_ratings"] = np.full(n, user_row["n_ratings"])
    feat["u_avg_rating"] = np.full(n, user_row["avg_rating"])
    feat["u_std_rating"] = np.full(n, user_row["std_rating"])
    feat["days_since_last"] = np.full(n, user_row["days_since_last"])
    feat["pct_high_ratings"] = np.full(n, user_row["pct_high_ratings"])
    for c in AFF_COLS:
        feat[c] = np.full(n, user_row.get(c, 0.0))
    feat["genre_match"] = genre_match
    feat["rating_delta"] = cands["avg_rating"].to_numpy() - user_row["avg_rating"]
    feat["popularity_rank"] = cands["n_ratings"].to_numpy() / max(max_item_ratings, 1)
    user_year = 2017 - (user_row["days_since_last"] / 365.0)
    feat["user_item_era_diff"] = np.abs(cands["release_year"].fill_null(0).to_numpy() - user_year)

    cols = feature_columns(items_df)
    X = np.column_stack([np.asarray(feat[c], dtype=np.float32) for c in cols])
    return X, cands["movieId"].to_list()
