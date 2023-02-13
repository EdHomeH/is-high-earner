# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import pickle


data_path = 'data/census.csv'
output_model_path = 'model/output/logistic.sav'

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


def main():

    data = pd.read_csv(data_path, sep=', ')
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    trained_model = train_model(X_train, y_train)
    print(type(trained_model))
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=inference(trained_model, X_test))

    print("precision: ", precision)
    print("recall: ", recall)
    print("fbeta: ", fbeta)

    pickle.dump(trained_model, open(output_model_path, 'wb'))


if __name__ == "__main__":
    main()
