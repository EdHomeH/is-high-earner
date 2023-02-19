from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from src.ml.model import load_model, inference
from src.ml.data import process_data
from src.train_model import output_model_path, cat_features, label


class User(BaseModel):
    age: int
    workclass: str
    fnlght: int
    education: str
    education_num: int = Field(alias = 'education-num')
    marital_status: str = Field(alias = 'marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias = 'capital-gain')
    capital_loss: int = Field(alias = 'capital-loss')
    hours_per_week: int = Field(alias = 'hours-per-week')
    native_country: str = Field(alias = 'native-country')

    class Config:
        schema_extra = {
            "example": {
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
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


app = FastAPI()


@app.get("/")
async def hello():
    return "Hello!"


@app.post("/predict/")
async def is_high_earner(user: User):

    model, encoder, lb = load_model(output_model_path, return_encoder_and_lbl_binarizer=True)
    
    user_df = pd.DataFrame([user.dict(by_alias=True)])

    X_test, _, _, _ = process_data(
        user_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    if inference(model, X_test).tolist()[0] == 1:
       return True 
    return False
