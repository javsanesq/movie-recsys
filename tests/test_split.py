import polars as pl
from pipeline.split import assign_split
from config import TRAIN_END, VAL_END

def test_assign_split_boundaries():
    df = pl.DataFrame({"timestamp": [TRAIN_END - 1, TRAIN_END, VAL_END - 1, VAL_END]})
    out = assign_split(df)["split"].to_list()
    assert out == ["train", "val", "val", "test"]
