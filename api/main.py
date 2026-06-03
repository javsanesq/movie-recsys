from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from api.recommender import Recommender
from api import schemas

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rec"] = Recommender()
    yield
    state.clear()


app = FastAPI(title="movie-recsys", lifespan=lifespan)


@app.get("/health", response_model=schemas.Health)
def health():
    return {"status": "ok", "models_loaded": bool(state.get("rec"))}


@app.get("/recommend/{user_id}", response_model=schemas.RecommendResponse)
def recommend(user_id: int,
              k: int = Query(10, ge=1, le=100),
              stage: str = Query("both", pattern="^(both|retrieval|ranker)$")):
    rec = state["rec"]
    if not rec.known_user(user_id):
        raise HTTPException(404, f"user {user_id} not found")
    s = "retrieval" if stage == "retrieval" else "both"
    return {"user_id": user_id, "stage": stage, "recommendations": rec.recommend(user_id, k, s)}


@app.get("/users/{user_id}", response_model=schemas.UserProfile)
def user(user_id: int):
    rec = state["rec"]
    if user_id not in rec.user_dict:
        raise HTTPException(404, f"user {user_id} not found")
    return rec.user_profile(user_id)


@app.get("/movies/{movie_id}", response_model=schemas.MovieInfo)
def movie(movie_id: int):
    rec = state["rec"]
    if movie_id not in rec.item_dict:
        raise HTTPException(404, f"movie {movie_id} not found")
    return rec.movie_info(movie_id)


@app.get("/metrics")
def metrics():
    return state["rec"].metrics
