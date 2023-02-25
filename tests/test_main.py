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


def test_post_is_not_high_earner():
    data = User.Config.schema_extra['example']
    
    r = client.post("/predict/", content=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == False


def test_post_is_high_earner():
    data = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlght": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    
    r = client.post("/predict/", content=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == True
