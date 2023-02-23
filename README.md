# IS HIGH EARNER

This repository is an example template of how to deploy a model that predicts if a user is a high earner or not.

### How to run:

```
uvicorn main:app
```
This will start a [FastAPI](https://fastapi.tiangolo.com/) server on http://127.0.0.1:8000/ that will execute a prediction by executing:
```
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 39,
  "workclass": "State-gov",
  "fnlght": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 3174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}'
```

The data used to train the model is from [UCI](https://archive.ics.uci.edu/ml/datasets/census+income) and tries to predict if the salary of the user is above 50k a year.



### To execute tests:

```
python -m pytest
```

### To test if the app deployed correctly:
```
python is_deployed.py
```

This is the third project in the [MLOps Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) program
