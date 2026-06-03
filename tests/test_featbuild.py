import numpy as np, polars as pl
from config import FEATURES
from eval.featbuild import build_matrix, feature_columns

def test_build_matrix_shapes_and_order():
    items = pl.read_parquet(FEATURES / "items.parquet")
    users = pl.read_parquet(FEATURES / "users.parquet")
    user_row = users.row(0, named=True)
    cand = items["movieId"].head(50).to_numpy()
    X, mids = build_matrix(user_row, cand, items, int(items["n_ratings"].max()))
    assert X.shape[0] == len(mids)
    assert X.shape[1] == len(feature_columns(items))
    assert X.dtype == np.float32
    assert set(mids).issubset(set(cand.tolist()))
    assert not np.isnan(X).any()
