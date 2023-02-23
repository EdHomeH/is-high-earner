import json

from fastapi.testclient import TestClient

from main import app, User

client = TestClient(app)


def test_get_success():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello!"

    
def test_get_fail():
    r = client.get("/somerandomtext/")
    assert r.status_code == 404


def test_post_fail():
    data = {"feature_1": -5, "feature_2": "test string"}
    r = client.post("/predict/", content=json.dumps(data))
    assert r.status_code == 422


def test_post_success():
    data = User.Config.schema_extra['example']
    
    r = client.post("/predict/", content=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == False
