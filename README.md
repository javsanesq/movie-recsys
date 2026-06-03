# movie-recsys

A production-style **two-stage movie recommender** on MovieLens 25M: ALS retrieval → LightGBM LambdaRank re-ranking, served by a FastAPI inference API behind a vanilla-JS inspection dashboard, with correct temporal evaluation and one-command Docker deployment.

```
GET /recommend/{user_id}
   │
   ├─ Stage 1 — ALS (implicit ALS, 128 factors)
   │     user vector · item matrix  →  exact top-200 inner-product search
   │     (excludes already-rated items)                      ~2 ms
   │
   └─ Stage 2 — LightGBM LambdaRank
         re-rank the 200 candidates on 73 user/item/cross features → top-K
```

The re-ranker beats both a popularity baseline and ALS-only on every accuracy metric — **NDCG@10 0.115 vs 0.072 vs 0.066**. Full numbers and design rationale in **[RESULTS.md](RESULTS.md)**; a from-scratch walkthrough in **[output/doc/movie-recsys-technical-deep-dive.docx](output/doc/)**.

## Quick start (Docker)

Trained artifacts (`artifacts/models/`, `artifacts/features/`) are gitignored — too large for the repo — so produce them once, then the image bakes them in:

```bash
make setup data pipeline train eval   # one-time: build the artifacts (~30 min)
docker compose up --build             # dashboard → :80   API → :8000
```

On a clone where the artifacts already exist, `docker compose up --build` serves immediately. The dashboard's three panels: user profile + genre affinity, live recommendations with a Stage-1/both toggle, and the offline model-comparison + feature-importance charts.

## Train from scratch

Requires [`uv`](https://docs.astral.sh/uv/). The pipeline provisions Python 3.11 and all deps.

```bash
make setup       # uv sync (Python 3.11 venv)
make data        # download MovieLens 25M (~250 MB)
make pipeline    # temporal split + user/item/genome features  -> Parquet
make train       # ALS + Faiss-free retrieval, LambdaRank ranker, popularity baseline
make eval        # offline evaluation -> artifacts/metrics.json
make api         # serve locally on :8000
make test        # pytest
```

Tuned for an 8 GB machine: genome tags are streamed and reduced before pivoting, the ranker trains on a 20k-user sample, and built feature matrices are cached so LightGBM can be re-tuned without rebuilding candidates.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /recommend/{id}?k=10&stage=both\|retrieval` | top-K recs with ranker `score` + `retrieval_score` |
| `GET /users/{id}` | profile: n_ratings, avg rating, top genres, genre affinity |
| `GET /movies/{id}` | title, genres, Bayesian avg rating, year |
| `GET /metrics` | offline model comparison + feature importance |
| `GET /health` | liveness |

## Design highlights

- **Temporal split** (train `<2017`, val `2017`, test `≥2018`) — no future leakage; all features computed on train only.
- **Two-stage retrieval + ranking** — the industry-standard pattern; Stage 1 maximizes recall, Stage 2 maximizes precision.
- **Honest metrics** — proper ranking measures (NDCG, MAP, Recall, Precision) plus coverage and personalization, on a 3,646-user test sample.
- **Exact numpy retrieval instead of Faiss** — equivalent at 18k items and avoids a fatal `faiss`/`lightgbm` OpenMP conflict; see RESULTS.md.
- **No train/serve skew** — training, evaluation, and serving share one `build_matrix` feature builder.

## Layout

```
data/ pipeline/   raw download + temporal split + feature engineering
train/            retrieval (ALS), ranker (LambdaRank), baselines
eval/             metrics, shared feature builder, offline evaluation
api/              FastAPI two-stage inference service
ui/               vanilla-JS dashboard + nginx config
artifacts/        models + features (gitignored) · metrics.json (committed)
```

## License

MIT — see [LICENSE](LICENSE).
