
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model.ml.model import train_model
from model.ml.data import process_data


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


def test_train_model(data):

    train, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    trained_model = train_model(X_train, y_train)

    assert type(trained_model) == LogisticRegression
