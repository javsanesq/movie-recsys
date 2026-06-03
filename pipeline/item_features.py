"""Per-item features computed on TRAIN split only, merged with genome top-20."""
import polars as pl
from config import (DATA_RAW, SPLITS, FEATURES, GENRES, BAYES_MIN_COUNT)

def extract_year(titles: pl.Series) -> pl.Series:
    return titles.str.extract(r"\((\d{4})\)", 1).cast(pl.Int32, strict=False)

def bayesian_avg(local_mean: float, count: int, global_mean: float, prior: int) -> float:
    return (count * local_mean + prior * global_mean) / (count + prior)

def main():
    train = pl.read_parquet(SPLITS / "train.parquet")
    movies = pl.read_csv(DATA_RAW / "movies.csv")
    global_mean = train["rating"].mean()

    agg = train.group_by("movieId").agg(
        pl.len().alias("n_ratings"),
        pl.col("rating").mean().alias("local_mean"),
        pl.col("rating").std().fill_null(0.0).alias("std_rating"),
    )
    agg = agg.with_columns(
        ((pl.col("n_ratings") * pl.col("local_mean") + BAYES_MIN_COUNT * global_mean)
         / (pl.col("n_ratings") + BAYES_MIN_COUNT)).alias("avg_rating")
    ).drop("local_mean")

    items = movies.join(agg, on="movieId", how="inner")
    items = items.with_columns(extract_year(pl.col("title")).alias("release_year"))
    for g in GENRES:
        items = items.with_columns(
            pl.col("genres").str.contains(g, literal=True).cast(pl.Int8).alias(f"genre_{g}"))

    genome = pl.read_parquet(FEATURES / "genome_top20.parquet")
    items = items.join(genome, on="movieId", how="left")
    genome_cols = [c for c in items.columns if c.startswith("genome_")]
    items = items.with_columns([pl.col(c).fill_null(0.0) for c in genome_cols])
    items.write_parquet(FEATURES / "items.parquet")
    print(f"items: {items.shape}, global_mean={global_mean:.3f}")

if __name__ == "__main__":
    main()
