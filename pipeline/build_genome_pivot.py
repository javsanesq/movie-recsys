"""Top-20 genome tags pivoted per movie. Filters tags BEFORE pivoting (8 GB safe)."""
import polars as pl
from config import DATA_RAW, FEATURES, GENOME_TOP_TAGS

def main():
    scores = DATA_RAW / "genome-scores.csv"
    top = (pl.scan_csv(scores)
           .group_by("tagId").agg(pl.col("relevance").mean().alias("m"))
           .sort("m", descending=True).head(GENOME_TOP_TAGS)
           .collect(engine="streaming"))
    top_ids = top["tagId"].to_list()
    tags = pl.read_csv(DATA_RAW / "genome-tags.csv").filter(pl.col("tagId").is_in(top_ids))
    id_to_name = {r["tagId"]: r["tag"] for r in tags.iter_rows(named=True)}

    sub = (pl.scan_csv(scores).filter(pl.col("tagId").is_in(top_ids))
           .collect(engine="streaming"))
    pivot = sub.pivot(values="relevance", index="movieId", on="tagId")
    pivot = pivot.rename({str(t): f"genome_{id_to_name[t]}" for t in top_ids if str(t) in pivot.columns})
    pivot.write_parquet(FEATURES / "genome_top20.parquet")
    print(f"genome_top20: {pivot.shape}, tags={list(id_to_name.values())}")

if __name__ == "__main__":
    main()
