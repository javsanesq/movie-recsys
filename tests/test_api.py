import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from config import MODELS


def _known_user():
    return int(np.load(MODELS / "idx_to_userid.npy")[0])


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["models_loaded"] is True


def test_recommend_both():
    with TestClient(app) as c:
        r = c.get(f"/recommend/{_known_user()}?k=10")
        assert r.status_code == 200
        assert len(r.json()["recommendations"]) == 10


def test_recommend_unknown_user():
    with TestClient(app) as c:
        assert c.get("/recommend/999999999").status_code == 404


def test_stage_retrieval_differs():
    with TestClient(app) as c:
        u = _known_user()
        both = [x["movie_id"] for x in c.get(f"/recommend/{u}?stage=both").json()["recommendations"]]
        retr = [x["movie_id"] for x in c.get(f"/recommend/{u}?stage=retrieval").json()["recommendations"]]
        assert both != retr
