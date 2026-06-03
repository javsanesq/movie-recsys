"""Per-user features computed on TRAIN split only."""
import polars as pl
from config import DATA_RAW, SPLITS, FEATURES, GENRES, TRAIN_END

def main():
    train = pl.read_parquet(SPLITS / "train.parquet")
    movies = pl.read_csv(DATA_RAW / "movies.csv").select("movieId", "genres")
    t = train.join(movies, on="movieId", how="left")

    base = train.group_by("userId").agg(
        pl.len().alias("n_ratings"),
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").std().fill_null(0.0).alias("std_rating"),
        pl.col("timestamp").max().alias("last_ts"),
        (pl.col("rating") >= 4.0).mean().alias("pct_high_ratings"),
    ).with_columns(
        ((TRAIN_END - pl.col("last_ts")) / 86400.0).alias("days_since_last")
    ).drop("last_ts")

    ga = t.group_by("userId").agg([
        pl.col("genres").str.contains(g, literal=True).cast(pl.Int8).mean().alias(f"genre_affinity_{g}")
        for g in GENRES
    ])
    base = base.join(ga, on="userId", how="left")

    base = base.fill_null(0.0)
    base.write_parquet(FEATURES / "users.parquet")
    print(f"users: {base.shape}")

if __name__ == "__main__":
    main()
