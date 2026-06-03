"""3-way offline eval (popularity / als_only / als_lgbm) -> artifacts/metrics.json.
Eval candidates are seen-filtered (novel items); labels are TEST positives."""
import json, numpy as np, joblib, lightgbm as lgb, polars as pl
from config import (SPLITS, MODELS, FEATURES, METRICS_JSON, N_CANDIDATES,
                    EVAL_USER_SAMPLE, POSITIVE_THRESHOLD)
from eval.featbuild import build_matrix
from eval.metrics import ndcg_at_k, recall_at_k, precision_at_k, average_precision_at_k
from train.retrieval import retrieve, load_user_seen, load_item_vectors

def main():
    als = joblib.load(MODELS / "als.pkl")
    item_vectors = load_item_vectors()
    ranker = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
    idx_to_movieid = np.load(MODELS / "idx_to_movieid.npy")
    idx_to_userid = np.load(MODELS / "idx_to_userid.npy")
    uidx = {int(u): i for i, u in enumerate(idx_to_userid.tolist())}
    pop = np.load(MODELS / "popularity.npy").tolist()
    items = pl.read_parquet(FEATURES / "items.parquet")
    users = pl.read_parquet(FEATURES / "users.parquet")
    user_dict = {int(r["userId"]): r for r in users.iter_rows(named=True)}
    seen_map = load_user_seen()
    max_item_ratings = int(items["n_ratings"].max())

    test = pl.read_parquet(SPLITS / "test.parquet").filter(pl.col("rating") >= POSITIVE_THRESHOLD)
    posdict = test.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    relevant = {u: set(m) for u, m in zip(posdict["userId"], posdict["movieId"])}
    cand_users = [u for u, m in zip(posdict["userId"], posdict["movieId"])
                  if len(m) >= 3 and u in uidx and u in user_dict]
    rng = np.random.default_rng(0); rng.shuffle(cand_users)
    cand_users = cand_users[:EVAL_USER_SAMPLE]

    METRIC_KEYS = ("ndcg@5", "ndcg@10", "recall@10", "precision@10", "map@10")
    acc = {m: {k: [] for k in METRIC_KEYS} for m in ("popularity", "als_only", "als_lgbm")}
    rec_lists = {m: [] for m in acc}

    for uid in cand_users:
        rel = relevant[uid]
        useen = seen_map.get(uid, set())
        cands, _ = retrieve(als, item_vectors, idx_to_movieid, uidx[uid], N_CANDIDATES, useen)
        if len(cands) == 0:
            continue
        X, mids = build_matrix(user_dict[uid], cands, items, max_item_ratings)
        scores = ranker.predict(X)
        order = np.argsort(scores)[::-1]
        ranked = {
            "popularity": [m for m in pop if m not in useen][:50],
            "als_only": cands.tolist(),
            "als_lgbm": [mids[i] for i in order],
        }
        for m, r in ranked.items():
            acc[m]["ndcg@5"].append(ndcg_at_k(r, rel, 5))
            acc[m]["ndcg@10"].append(ndcg_at_k(r, rel, 10))
            acc[m]["recall@10"].append(recall_at_k(r, rel, 10))
            acc[m]["precision@10"].append(precision_at_k(r, rel, 10))
            acc[m]["map@10"].append(average_precision_at_k(r, rel, 10))
            rec_lists[m].append(set(r[:10]))

    def coverage(lists):
        seen = set().union(*lists) if lists else set()
        return len(seen) / items.height
    def personalization(lists):
        if len(lists) < 2:
            return 0.0
        idx = rng.choice(len(lists), size=min(500, len(lists)), replace=False)
        sub = [lists[i] for i in idx]; sims = []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                u = len(sub[i] | sub[j]); sims.append(len(sub[i] & sub[j]) / u if u else 0.0)
        return 1.0 - float(np.mean(sims)) if sims else 0.0

    models = {}
    for m in acc:
        models[m] = {k: round(float(np.mean(acc[m][k])), 4) for k in METRIC_KEYS}
        models[m]["coverage@10"] = round(coverage(rec_lists[m]), 4)
        models[m]["personalization@10"] = round(personalization(rec_lists[m]), 4)

    fi = json.load(open(MODELS / "feature_importance.json"))
    total = sum(fi.values()) or 1.0
    fi_norm = dict(sorted(((k, round(v / total, 4)) for k, v in fi.items()), key=lambda x: -x[1])[:10])
    out = {"n_users_evaluated": len(cand_users), "models": models, "feature_importance": fi_norm}
    METRICS_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(models, indent=2))

if __name__ == "__main__":
    main()
