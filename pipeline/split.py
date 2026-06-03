"""Temporal train/val/test split + filtering. Streams with Polars."""
import polars as pl
from config import (DATA_RAW, SPLITS, TRAIN_END, VAL_END,
                    MIN_USER_TRAIN_RATINGS, MIN_ITEM_TRAIN_RATINGS)

def assign_split(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("timestamp") < TRAIN_END).then(pl.lit("train"))
          .when(pl.col("timestamp") < VAL_END).then(pl.lit("val"))
          .otherwise(pl.lit("test")).alias("split")
    )

def main():
    ratings = pl.scan_csv(DATA_RAW / "ratings.csv").collect(engine="streaming")
    ratings = assign_split(ratings)
    train = ratings.filter(pl.col("split") == "train")

    good_users = (train.group_by("userId").len()
                  .filter(pl.col("len") >= MIN_USER_TRAIN_RATINGS)["userId"])
    good_items = (train.group_by("movieId").len()
                  .filter(pl.col("len") >= MIN_ITEM_TRAIN_RATINGS)["movieId"])
    ratings = ratings.filter(
        pl.col("userId").is_in(good_users.to_list()) &
        pl.col("movieId").is_in(good_items.to_list()))

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        part = ratings.filter(pl.col("split") == name)
        part.write_parquet(SPLITS / f"{name}.parquet")
        print(f"{name}: {part.height:,} rows, "
              f"{part['userId'].n_unique():,} users, {part['movieId'].n_unique():,} items")

if __name__ == "__main__":
    main()
