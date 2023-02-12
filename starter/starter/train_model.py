# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics


data = pd.read_csv('starter/data/census.csv', sep=', ')
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

trained_model = train_model(X_train, y_train)


X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=trained_model.predict(X_test))
print("precision: ", precision)
print("recall: ", recall)
print("fbeta: ", fbeta)
