
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.ml.model import train_model, inference, compute_model_metrics, load_model
from src.ml.data import process_data
from src.train_model import cat_features, label, random_state


def test_train_model_type(data):

    train, _ = train_test_split(data, test_size=0.20, random_state=random_state)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    trained_model = train_model(X_train, y_train)

    assert type(trained_model) == LogisticRegression


def test_inference(data):

    _, test = train_test_split(data, test_size=0.20, random_state=random_state)

    trained_model, encoder, lb = load_model('src/output/', return_encoder_and_lbl_binarizer=True)

    X_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
    )

    assert inference(trained_model, X_test[:1]) == [0]
    assert inference(trained_model, X_test[15:16]) == [1]
 

def test_compute_model_metrics(data):

    _, test = train_test_split(data, test_size=0.20, random_state=random_state)

    trained_model, encoder, lb = load_model('src/output/', return_encoder_and_lbl_binarizer=True)

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
    )

    precision, recall, fbeta = compute_model_metrics(y=y_test[:1], preds=inference(trained_model, X_test[:1]))
    
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
