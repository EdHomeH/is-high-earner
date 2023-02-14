import pytest
from sklearn.model_selection import train_test_split

from src.ml.data import process_data
from src.train_model import cat_features, label


def test_process_data(data):
    
    train, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    assert X_train.shape[0] == 26048
    assert X_train.shape[1] == 108
    assert y_train.shape[0] == 26048
    with pytest.raises(IndexError):
        y_train.shape[1]
 