"""Stage 2: build training data from ALS candidates, train LightGBM LambdaRank.

Samples RANKER_USER_SAMPLE users to fit 8 GB. Training candidates are NOT
seen-filtered (so train-rated items provide positive labels). Early-stopping
validation uses a held-out slice of TRAIN users (labeled by train ratings) so
the NDCG signal is dense — the temporal val/test splits are reserved for the
final unbiased model comparison in eval/evaluate.py.

Built feature matrices are cached to `_ranker_cache.npz` so LightGBM
hyperparameters can be re-tuned without repeating the ~14 min candidate build.
"""
import json, numpy as np, joblib, lightgbm as lgb, polars as pl
from config import (SPLITS, MODELS, FEATURES, N_CANDIDATES, RANKER_USER_SAMPLE)
from eval.featbuild import build_matrix, feature_columns
from train.retrieval import retrieve, load_item_vectors

CACHE = MODELS / "_ranker_cache.npz"
VAL_USERS = 3000  # held-out train users for early-stopping


def load_state():
    als = joblib.load(MODELS / "als.pkl")
    item_vectors = load_item_vectors()
    idx_to_movieid = np.load(MODELS / "idx_to_movieid.npy")
    idx_to_userid = np.load(MODELS / "idx_to_userid.npy")
    uidx = {int(u): i for i, u in enumerate(idx_to_userid.tolist())}
    items = pl.read_parquet(FEATURES / "items.parquet")
    users = pl.read_parquet(FEATURES / "users.parquet")
    user_dict = {int(r["userId"]): r for r in users.iter_rows(named=True)}
    return als, item_vectors, idx_to_movieid, uidx, items, user_dict


def build_dataset(als, item_vectors, idx_to_movieid, uidx, items, user_dict,
                  user_ids, rating_map, max_item_ratings):
    Xs, ys, groups = [], [], []
    for uid in user_ids:
        if uid not in user_dict:
            continue
        cands, _ = retrieve(als, item_vectors, idx_to_movieid, uidx[uid], N_CANDIDATES, seen=None)
        if len(cands) == 0:
            continue
        X, mids = build_matrix(user_dict[uid], cands, items, max_item_ratings)
        y = np.array([min(4, int(np.floor(rating_map.get(uid, {}).get(m, 0.0)))) for m in mids],
                     dtype=np.int32)
        Xs.append(X); ys.append(y); groups.append(len(mids))
    return np.vstack(Xs), np.concatenate(ys), np.array(groups)


def build_cache():
    als, item_vectors, idx_to_movieid, uidx, items, user_dict = load_state()
    max_item_ratings = int(items["n_ratings"].max())
    train = pl.read_parquet(SPLITS / "train.parquet")
    pos = train.group_by("userId").agg([pl.col("movieId"), pl.col("rating")]).to_dict(as_series=False)
    rating_map = {u: dict(zip(m, r)) for u, m, r in zip(pos["userId"], pos["movieId"], pos["rating"])}

    uids = train["userId"].unique().to_numpy()
    uids = uids[np.isin(uids, list(uidx.keys()))]
    rng = np.random.default_rng(0); rng.shuffle(uids)
    fit_uids = uids[:RANKER_USER_SAMPLE]
    val_uids = uids[RANKER_USER_SAMPLE:RANKER_USER_SAMPLE + VAL_USERS]

    Xtr, ytr, gtr = build_dataset(als, item_vectors, idx_to_movieid, uidx, items, user_dict, fit_uids, rating_map, max_item_ratings)
    Xva, yva, gva = build_dataset(als, item_vectors, idx_to_movieid, uidx, items, user_dict, val_uids, rating_map, max_item_ratings)
    np.savez(CACHE, Xtr=Xtr, ytr=ytr, gtr=gtr, Xva=Xva, yva=yva, gva=gva)
    return Xtr, ytr, gtr, Xva, yva, gva


def main():
    if CACHE.exists():
        d = np.load(CACHE)
        Xtr, ytr, gtr, Xva, yva, gva = d["Xtr"], d["ytr"], d["gtr"], d["Xva"], d["yva"], d["gva"]
        print("loaded cached ranker data")
    else:
        Xtr, ytr, gtr, Xva, yva, gva = build_cache()
    print(f"train X={Xtr.shape}, val X={Xva.shape}", flush=True)

    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr)
    dval = lgb.Dataset(Xva, label=yva, group=gva, reference=dtrain)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [5, 10, 20],
        "label_gain": [0, 1, 3, 7, 15], "num_leaves": 63, "learning_rate": 0.05,
        "min_child_samples": 50, "verbose": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(25)])
    booster.save_model(str(MODELS / "ranker.txt"))

    items = pl.read_parquet(FEATURES / "items.parquet")
    cols = feature_columns(items)
    imp = dict(zip(cols, booster.feature_importance(importance_type="gain").tolist()))
    (MODELS / "feature_importance.json").write_text(json.dumps(imp))
    (MODELS / "feature_columns.json").write_text(json.dumps(cols))
    print("best score:", dict(booster.best_score["valid_0"]))


if __name__ == "__main__":
    main()
