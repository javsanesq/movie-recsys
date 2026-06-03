from pydantic import BaseModel


class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float
    retrieval_score: float


class RecommendResponse(BaseModel):
    user_id: int
    stage: str
    recommendations: list[Recommendation]


class UserProfile(BaseModel):
    user_id: int
    n_ratings: int
    avg_rating: float
    top_genres: list[str]
    genre_affinity: dict[str, float]


class MovieInfo(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    avg_rating: float
    n_ratings: int
    release_year: int | None


class Health(BaseModel):
    status: str
    models_loaded: bool
