import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_success():
    r = client.get("/")
    assert r.status_code == 200
    assert r.content == b'"Hello!"'

    
def test_get_fail():
    r = client.get("/somerandomtext/")
    assert r.status_code == 404


def test_post_fail():
    data = {"feature_1": -5, "feature_2": "test string"}
    r = client.post("/predict/", content=json.dumps(data))
    assert r.status_code == 422

def test_post_success():
    data = {
      "age": 39,
      "workclass": "State-gov",
      "fnlght": 77516,
      "education": "Bachelors",
      "education_num": 13,
      "marital_status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital_gain": 2174,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States"
    }
    r = client.post("/predict/", content=json.dumps(data))
    assert r.status_code == 200
    assert r.content == b'false'

