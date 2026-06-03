import polars as pl
from pipeline.item_features import extract_year, bayesian_avg

def test_extract_year():
    s = pl.Series(["Toy Story (1995)", "No Year Movie", "Blade Runner (1982)"])
    assert extract_year(s).to_list() == [1995, None, 1982]

def test_bayesian_avg_shrinks_low_count():
    val = bayesian_avg(local_mean=5.0, count=1, global_mean=3.0, prior=25)
    assert 3.0 < val < 3.2
