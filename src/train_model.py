# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.data import process_data
from src.ml.model import train_model, compute_model_metrics, inference, save_model


random_state=123

data_path = 'data/census.csv'
output_model_path = 'src/output/logistic.sav'

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
label = "salary"


def main():

    data = pd.read_csv(data_path, sep=', ', engine='python')
    train, test = train_test_split(data, test_size=0.20, random_state=random_state)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    trained_model = train_model(X_train, y_train)

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
    )
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=inference(trained_model, X_test))

    print("precision: ", precision)
    print("recall: ", recall)
    print("fbeta: ", fbeta)

    save_model(trained_model, output_model_path)



if __name__ == "__main__":
    main()
