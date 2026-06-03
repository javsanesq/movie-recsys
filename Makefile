.PHONY: setup data pipeline train eval api test up down clean
export PYTHONPATH := .

setup:           ## install deps into a 3.11 venv
	uv sync

data:            ## download MovieLens 25M
	uv run python data/download.py

pipeline:        ## temporal split + feature engineering
	uv run python pipeline/split.py
	uv run python pipeline/build_genome_pivot.py
	uv run python pipeline/item_features.py
	uv run python pipeline/user_features.py

train:           ## ALS retrieval + LambdaRank ranker + popularity baseline
	uv run python train/retrieval.py
	uv run python train/ranker.py
	uv run python train/baselines.py

eval:            ## offline evaluation -> artifacts/metrics.json
	uv run python eval/evaluate.py

api:             ## run the API locally (http://localhost:8000)
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

test:            ## run the test suite
	uv run pytest -q

up:              ## build + run the full stack (API :8000, dashboard :80)
	docker compose up --build

down:
	docker compose down

clean:           ## remove the ranker build cache
	rm -f artifacts/models/_ranker_cache.npz
