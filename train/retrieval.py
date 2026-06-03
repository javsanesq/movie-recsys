"""Stage 1: ALS confidence model -> exact top-k inner-product retrieval.

Retrieval uses the raw inner product (matching ALS's training objective, not
cosine) and filters out items the user already rated in train, so candidate
slots go to novel items. Both choices materially improve Recall@200 on the
temporal val split (see RESULTS.md).

Search is exact brute-force numpy (`item_vectors @ user_vec` + argpartition).
At ~18k items this is ~2 ms — an ANN index (Faiss) adds no value at this scale,
and crucially numpy avoids the duplicate-libomp conflict between Faiss and
LightGBM that deadlocks when both run in one process (eval + serving do).
"""
import numpy as np, joblib, polars as pl
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from config import (SPLITS, MODELS, FEATURES, ALS_FACTORS, ALS_ITERATIONS,
                    ALS_REGULARIZATION, ALS_ALPHA, N_CANDIDATES, POSITIVE_THRESHOLD)

USER_SEEN = FEATURES / "user_seen.parquet"
ITEM_VECTORS = MODELS / "item_vectors.npy"


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


def build_item_vectors(item_factors: np.ndarray) -> np.ndarray:
    """C-contiguous float32 item matrix for brute-force inner-product search."""
    return np.ascontiguousarray(item_factors.astype(np.float32))


def retrieve(model, item_vectors, idx_to_movieid, user_idx, k, seen=None):
    """Top-k candidate movieIds + scores for a user, excluding seen items.

    Exact inner-product search. `seen` is a set of movieIds the user already
    rated in train; over-fetches by len(seen) so k novel items survive.
    """
    uvec = model.user_factors[user_idx].astype(np.float32)
    scores = item_vectors @ uvec
    over = min(k + (len(seen) if seen else 0), scores.shape[0])
    part = np.argpartition(-scores, over - 1)[:over]
    part = part[np.argsort(-scores[part])]
    mids, sc = idx_to_movieid[part], scores[part]
    if seen:
        keep = np.array([m not in seen for m in mids.tolist()])
        mids, sc = mids[keep], sc[keep]
    return mids[:k], sc[:k]


def save_user_seen():
    """Persist each user's train-rated movieIds for seen-filtering at serve time."""
    train = pl.read_parquet(SPLITS / "train.parquet")
    seen = train.group_by("userId").agg(pl.col("movieId").alias("seen"))
    seen.write_parquet(USER_SEEN)
    return seen


def load_user_seen() -> dict[int, set]:
    seen = pl.read_parquet(USER_SEEN)
    return {int(u): set(m) for u, m in zip(seen["userId"], seen["seen"])}


def load_item_vectors() -> np.ndarray:
    return np.load(ITEM_VECTORS)


def main():
    user_items, idx_to_userid, idx_to_movieid, uidx, iidx = build_matrix()
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS, iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION, use_gpu=False)
    model.fit(user_items)

    item_vectors = build_item_vectors(model.item_factors)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "als.pkl")
    np.save(ITEM_VECTORS, item_vectors)
    np.save(MODELS / "idx_to_movieid.npy", idx_to_movieid)
    np.save(MODELS / "idx_to_userid.npy", idx_to_userid)
    save_user_seen()

    recall = eval_recall(model, item_vectors, idx_to_movieid, idx_to_userid, uidx)
    print(f"Recall@{N_CANDIDATES} = {recall:.4f}")


def eval_recall(model, item_vectors, idx_to_movieid, idx_to_userid, uidx, sample=5000):
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
        cands, _ = retrieve(model, item_vectors, idx_to_movieid, uidx[uid],
                            N_CANDIDATES, seen_map.get(uid))
        retrieved = set(cands.tolist())
        hits += sum(1 for m in pos if m in retrieved)
        total += len(pos)
    return hits / max(total, 1)


if __name__ == "__main__":
    main()
