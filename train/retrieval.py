"""Stage 1: ALS confidence model -> item vectors -> Faiss IndexFlatIP (raw dot).

Retrieval uses the raw inner product (matching ALS's training objective, not
cosine) and filters out items the user already rated in train, so candidate
slots go to novel items. Both choices materially improve Recall@200 on the
temporal val split (see RESULTS.md).
"""
import numpy as np, joblib, faiss, polars as pl
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from config import (SPLITS, MODELS, FEATURES, ALS_FACTORS, ALS_ITERATIONS,
                    ALS_REGULARIZATION, ALS_ALPHA, N_CANDIDATES, POSITIVE_THRESHOLD)

USER_SEEN = FEATURES / "user_seen.parquet"


def build_matrix():
    train = pl.read_parquet(SPLITS / "train.parquet")
    users = train["userId"].unique().sort().to_list()
    items = train["movieId"].unique().sort().to_list()
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {m: i for i, m in enumerate(items)}
    rows = train["userId"].replace_strict(uidx).to_numpy()
    cols = train["movieId"].replace_strict(iidx).to_numpy()
    conf = (1.0 + ALS_ALPHA * train["rating"].to_numpy()).astype(np.float32)
    mat = csr_matrix((conf, (rows, cols)), shape=(len(users), len(items)))
    return mat, np.array(users), np.array(items), uidx, iidx


def build_index(item_factors: np.ndarray) -> faiss.IndexFlatIP:
    """Raw inner-product index over ALS item vectors (no normalization)."""
    iv = np.ascontiguousarray(item_factors.astype(np.float32))
    index = faiss.IndexFlatIP(iv.shape[1])
    index.add(iv)
    return index


def retrieve(model, index, idx_to_movieid, user_idx, k, seen=None):
    """Top-k candidate movieIds + scores for a user, excluding seen items.

    `seen` is a set of movieIds the user already rated in train. Over-fetches by
    len(seen) so k novel items survive the filter.
    """
    uvec = model.user_factors[user_idx].astype(np.float32).reshape(1, -1)
    over = min(k + (len(seen) if seen else 0), index.ntotal)
    scores, idxs = index.search(uvec, over)
    mids, scores = idx_to_movieid[idxs[0]], scores[0]
    if seen:
        keep = np.array([m not in seen for m in mids.tolist()])
        mids, scores = mids[keep], scores[keep]
    return mids[:k], scores[:k]


def save_user_seen():
    """Persist each user's train-rated movieIds for seen-filtering at serve time."""
    train = pl.read_parquet(SPLITS / "train.parquet")
    seen = train.group_by("userId").agg(pl.col("movieId").alias("seen"))
    seen.write_parquet(USER_SEEN)
    return seen


def load_user_seen() -> dict[int, set]:
    seen = pl.read_parquet(USER_SEEN)
    return {int(u): set(m) for u, m in zip(seen["userId"], seen["seen"])}


def main():
    user_items, idx_to_userid, idx_to_movieid, uidx, iidx = build_matrix()
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS, iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION, use_gpu=False)
    model.fit(user_items)

    index = build_index(model.item_factors)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "als.pkl")
    faiss.write_index(index, str(MODELS / "faiss.index"))
    np.save(MODELS / "idx_to_movieid.npy", idx_to_movieid)
    np.save(MODELS / "idx_to_userid.npy", idx_to_userid)
    save_user_seen()

    recall = eval_recall(model, index, idx_to_movieid, idx_to_userid, uidx)
    print(f"Recall@{N_CANDIDATES} = {recall:.4f}")


def eval_recall(model, index, idx_to_movieid, idx_to_userid, uidx, sample=5000):
    seen_map = load_user_seen()
    mid_set = set(idx_to_movieid.tolist())
    val = pl.read_parquet(SPLITS / "val.parquet").filter(pl.col("rating") >= POSITIVE_THRESHOLD)
    val = val.filter(pl.col("userId").is_in(list(uidx.keys())))
    user_pos = val.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    rng = np.random.default_rng(0)
    pairs = list(zip(user_pos["userId"], user_pos["movieId"]))
    rng.shuffle(pairs)
    hits = total = 0
    for uid, pos in pairs[:sample]:
        pos = [m for m in pos if m in mid_set]
        if not pos:
            continue
        cands, _ = retrieve(model, index, idx_to_movieid, uidx[uid],
                            N_CANDIDATES, seen_map.get(uid))
        retrieved = set(cands.tolist())
        hits += sum(1 for m in pos if m in retrieved)
        total += len(pos)
    return hits / max(total, 1)


if __name__ == "__main__":
    main()
