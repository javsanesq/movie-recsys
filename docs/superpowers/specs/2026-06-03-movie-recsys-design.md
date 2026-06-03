# movie-recsys — Design Spec

**Date:** 2026-06-03
**Author:** Javier Sánchez Esquivel
**Status:** Approved pending user review

A portfolio-grade, two-stage movie recommendation system: ALS retrieval → Faiss
ANN → LightGBM LambdaRank re-ranking, served by FastAPI behind a vanilla-JS
inspection dashboard, deployed via Docker Compose. Companion to `rag-assistant`
and `tabular-ai-analyst` and styled to match them.

The full phase-by-phase implementation detail (code samples, exact API
signatures, feature tables) lives in the source plan at
`~/Desktop/movie-recsys.md`. This spec records the **decisions and deltas** that
turn that plan into the thing actually built on this machine.

---

## Goal & Success Criteria

Demonstrate production-grade RecSys engineering for a CV: correct temporal
evaluation, feature engineering, ANN retrieval, learning-to-rank re-ranking, an
inference API, and an inspection dashboard — all reproducible.

**Done when:**
- Pipeline turns raw MovieLens 25M into Parquet features with no temporal leakage.
- Stage-1 ALS achieves **Recall@200 ≥ 0.50** on the validation set.
- Offline eval shows **NDCG@10: als_lgbm > als_only > popularity** (real numbers
  in committed `artifacts/metrics.json`).
- FastAPI serves `/recommend` in **< 200 ms** with models pre-loaded.
- Dashboard loads, shows user profile + recommendations + metrics, and toggles
  Stage-1-only vs both-stages.
- `docker compose up --build` serves API (8000) + UI (80) end-to-end, **verified
  by actually building and running the images**.
- `output/doc/movie-recsys-technical-deep-dive.docx` explains the whole system.

---

## Key Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| **Dataset scale** | Full ml-25m for retrieval & eval; **sample ~40k users** for ranker training | Real portfolio numbers, but the ranker feature matrix (200 cands × 150k users) would OOM on 8 GB. Sampling is standard practice — documented. |
| **Execution depth** | Build **+ train + verify** | Produce real artifacts, real `metrics.json`, working API & dashboard. |
| **Repo** | Public, `movie-recsys` under `javsanesq` | Matches existing CV repos. |
| **Study guide** | `.docx` deep-dive, **built incrementally** per phase | Styled on `rag-assistant-technical-deep-dive.docx`. |
| **Deployment** | **Full Docker**: write config, build images, verify `compose up` serves | Proves end-to-end like the other repos. |

---

## Environment & Tooling

- **`uv` provisions an isolated Python 3.11 venv.** System Python is 3.14.4 —
  no faiss/lightgbm/implicit wheels exist for it. 3.11 has arm64-mac wheels for
  the whole stack.
- **`pyproject.toml` + `uv.lock`** instead of raw `requirements.txt`
  (reproducible; modern convention). A `requirements.txt` is also exported for
  the Dockerfile's `pip install`.
- Core deps: `faiss-cpu`, `lightgbm`, `implicit`, `scipy`, `polars`, `pandas`,
  `pyarrow`, `numpy`, `fastapi`, `uvicorn`, `joblib`, `python-docx`.

## Memory Strategy — the binding constraint (8 GB RAM, M2)

This is the part the original plan under-specifies. On 8 GB:

1. **Genome pivot:** never materialize the dense 30M-row × 1128-tag matrix.
   Stream `genome-scores.csv` with Polars, compute mean relevance per tag,
   select **top-20 tags**, *then* filter rows to those tags and pivot. Peak
   memory stays in the low hundreds of MB.
2. **Ratings:** Polars lazy/streaming scans for the temporal split and for
   per-user / per-item aggregations. Never hold all 25M rows + derived copies.
3. **ALS:** `scipy.sparse.csr_matrix` (~150k users × 27k items, sparse) is small.
   ALS itself is the lightest heavy-step.
4. **Ranker training matrix (the bomb):** sample ~40k train users → ~8M rows ×
   ~70 float32 features ≈ 2 GB. Build in chunks, cast to float32, free
   intermediates. LightGBM trains from a single `lgb.Dataset`.
5. **Eval:** cap at ~5k test users (those with ≥ 3 positive test items) for
   tractable runtime. Documented as a sample, not the full population.

Every compromise above is written into `RESULTS.md` and the README as a
deliberate, justified trade-off — not hidden.

---

## Architecture

```
GET /recommend/{user_id}?k=10&stage=both|retrieval|ranker
   │
   ├─ Stage 1: ALS user vector → Faiss IndexFlatIP (cosine via L2-normalized IP)
   │           → top-200 candidate movieIds (ms)
   │
   └─ Stage 2: build (user × candidate) feature matrix → LightGBM LambdaRank
               → sort → top-K
```

**Services (Docker Compose):**
- `api` — FastAPI on 8000, models loaded once at startup, artifacts baked into image.
- `ui` — nginx on 80, serves vanilla-JS dashboard, proxies `/api/*` → `api:8000`.

Training/pipeline are standalone scripts, not services. Outputs to `artifacts/`
(gitignored except `artifacts/metrics.json`).

## Repository Structure (aligned to house style)

Matches `rag-assistant` / `tabular-ai-analyst` conventions (adds Makefile,
LICENSE, SECURITY.md, scripts/, output/doc/ beyond the bare plan):

```
movie-recsys/
├── data/{download.py, raw/(gitignored)}
├── pipeline/{split.py, user_features.py, item_features.py, build_genome_pivot.py}
├── train/{retrieval.py, ranker.py, baselines.py}
├── eval/{metrics.py, evaluate.py}
├── api/{main.py, recommender.py, schemas.py}
├── ui/{index.html, app.js, style.css, nginx.conf}
├── artifacts/{models/, features/, metrics.json(committed)}
├── output/doc/movie-recsys-technical-deep-dive.docx
├── docs/superpowers/specs/   (this spec)
├── scripts/                  (helper run scripts)
├── tests/                    (pipeline + API smoke tests)
├── pyproject.toml, uv.lock, requirements.txt
├── Dockerfile.api, docker-compose.yml
├── Makefile, README.md, RESULTS.md, SECURITY.md, LICENSE, .gitignore
```

---

## Phases, Verification & Commit Cadence

One commit per completed phase (~7–9 commits total). A study-guide section is
drafted as each phase lands.

| # | Phase | Verification gate | Commit |
|---|---|---|---|
| 1 | Scaffold + data pipeline | splits + 3 parquet feature files exist, row counts printed, no leakage | "scaffold + data pipeline" |
| 2 | ALS + Faiss retrieval | `als.pkl`, `faiss.index`, `idx_to_movieid.npy`; **Recall@200 ≥ 0.50** | "ALS retrieval + Faiss" |
| 3 | LightGBM ranker + baselines + eval | `ranker.txt`; `metrics.json` shows als_lgbm > als_only > popularity | "LambdaRank ranker + offline eval" |
| 4 | FastAPI inference | `/health`, `/recommend`, `/users`, `/movies`, `/metrics` work; `/recommend` < 200 ms | "FastAPI inference service" |
| 5 | Vanilla-JS dashboard | profile loads, recs render, stage toggle works, error states clean | "inspection dashboard" |
| 6 | Docker + docs | `docker compose up --build` serves both; README + RESULTS done | "docker compose + docs" |

Final commit(s): study-guide docx polish, metrics.json, RESULTS.md.

## Out of Scope (YAGNI)

- No real-time / online learning; batch-trained artifacts only.
- No cold-start model — users/items below thresholds excluded from eval and
  documented (production note: fallback to popularity + content-based).
- No auth, rate limiting, or multi-user serving concerns (single-reviewer demo).
- No GPU; `use_gpu=False` throughout.
- No A/B infra, no feature store — Parquet on disk is the "store".

## Risks

- **8 GB OOM** during ranker build or genome pivot → mitigated by streaming +
  user sampling + float32; if a step still spikes, reduce sample size and note it.
- **Faiss/implicit wheel issues on 3.11/arm64** → verified available; uv pins them.
- **Long training runtime** ties up the machine → acceptable per "build+train+verify".
