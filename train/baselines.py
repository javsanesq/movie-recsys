"""Popularity baseline: most-rated TRAIN items, descending movieId list."""
import numpy as np, polars as pl
from config import SPLITS, MODELS

def popularity_ranking() -> np.ndarray:
    train = pl.read_parquet(SPLITS / "train.parquet")
    pop = train.group_by("movieId").len().sort("len", descending=True)
    return pop["movieId"].to_numpy()

def main():
    ranking = popularity_ranking()
    np.save(MODELS / "popularity.npy", ranking)
    print(f"popularity ranking: {len(ranking)} items, top5={ranking[:5].tolist()}")

if __name__ == "__main__":
    main()
