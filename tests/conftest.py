import pytest
import pandas as pd
from sklearn.model_selection import train_test_split


random_state = 123
data_path = 'data/census.csv'

@pytest.fixture
def data():
    return pd.read_csv(data_path, sep=', ', engine='python')


@pytest.fixture()
def train_data(data):
    train, _ = train_test_split(data, test_size=0.20, random_state=random_state)
    return train


@pytest.fixture
def test_data(data):
    _, test = train_test_split(data, test_size=0.20, random_state=random_state)
    return test 
