"""Loads all artifacts once; serves two-stage recommendations.

Mirrors eval/evaluate.py exactly (seen-filtered retrieve -> build_matrix ->
ranker) so serving and offline evaluation use identical logic — no train/serve
skew.
"""
import json
import numpy as np
import joblib
import lightgbm as lgb
import polars as pl

from config import MODELS, FEATURES, METRICS_JSON, N_CANDIDATES, GENRES
from eval.featbuild import build_matrix
from train.retrieval import retrieve, load_user_seen, load_item_vectors


class Recommender:
    def __init__(self):
        self.als = joblib.load(MODELS / "als.pkl")
        self.item_vectors = load_item_vectors()
        self.ranker = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
        self.idx_to_movieid = np.load(MODELS / "idx_to_movieid.npy")
        idx_to_userid = np.load(MODELS / "idx_to_userid.npy")
        self.uidx = {int(u): i for i, u in enumerate(idx_to_userid.tolist())}
        self.items = pl.read_parquet(FEATURES / "items.parquet")
        self.users = pl.read_parquet(FEATURES / "users.parquet")
        self.user_dict = {int(r["userId"]): r for r in self.users.iter_rows(named=True)}
        self.item_dict = {int(r["movieId"]): r for r in self.items.iter_rows(named=True)}
        self.seen = load_user_seen()
        self.max_item_ratings = int(self.items["n_ratings"].max())
        self.metrics = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
        self.loaded = True

    def known_user(self, uid: int) -> bool:
        return uid in self.uidx

    def _format(self, movieids, scores, retr):
        out = []
        for m, s, r in zip(movieids, scores, retr):
            it = self.item_dict.get(int(m))
            if not it:
                continue
            out.append({
                "movie_id": int(m),
                "title": it["title"],
                "genres": it["genres"].split("|"),
                "score": float(s),
                "retrieval_score": float(r),
            })
        return out

    def recommend(self, uid: int, k: int = 10, stage: str = "both"):
        cands, retr = retrieve(self.als, self.item_vectors, self.idx_to_movieid,
                               self.uidx[uid], N_CANDIDATES, self.seen.get(uid))
        if len(cands) == 0:
            return []
        if stage == "retrieval":
            return self._format(cands[:k], retr[:k], retr[:k])

        retr_map = {int(m): float(r) for m, r in zip(cands, retr)}
        X, mids = build_matrix(self.user_dict[uid], cands, self.items, self.max_item_ratings)
        scores = self.ranker.predict(X)
        order = np.argsort(scores)[::-1][:k]
        chosen = [mids[i] for i in order]
        return self._format(chosen, [scores[i] for i in order],
                            [retr_map[int(m)] for m in chosen])

    def user_profile(self, uid: int):
        r = self.user_dict[uid]
        aff = {g: float(r[f"genre_affinity_{g}"]) for g in GENRES}
        top = sorted(aff, key=aff.get, reverse=True)[:5]
        return {
            "user_id": uid,
            "n_ratings": int(r["n_ratings"]),
            "avg_rating": float(r["avg_rating"]),
            "top_genres": top,
            "genre_affinity": aff,
        }

    def movie_info(self, mid: int):
        r = self.item_dict[mid]
        yr = r.get("release_year")
        return {
            "movie_id": mid,
            "title": r["title"],
            "genres": r["genres"].split("|"),
            "avg_rating": float(r["avg_rating"]),
            "n_ratings": int(r["n_ratings"]),
            "release_year": int(yr) if yr is not None else None,
        }
