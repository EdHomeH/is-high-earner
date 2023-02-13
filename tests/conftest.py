import pytest
import pandas as pd


data_path = 'data/census.csv'


@pytest.fixture
def data():
    return pd.read_csv(data_path, sep=', ')
