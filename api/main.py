
import os
import sys

from fastapi import FastAPI
from starlette.responses import JSONResponse

from api.models.models import HousePricing
from mlops_project.predictor.api_predict import ModelAPIPredictor

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(current_dir)


app = FastAPI()

"""
PARAMETER VALUES
Values are required after de endpoint.
"""


@app.get('/', status_code=200)
async def healthcheck():
    return 'HousePricing Regressor is ready to go!'


@app.post('/predict')
def extract_name(housepricing_features: HousePricing):
    predictor = ModelAPIPredictor("C:/Users/usuario/Documents/GitHub/mlops_project/mlops_project/models/random_forest_output.pkl")
    X = [
        housepricing_features.crim,
        housepricing_features.zn,
        housepricing_features.indus,
        housepricing_features.chas,
        housepricing_features.nox,
        housepricing_features.rm,
        housepricing_features.age,
        housepricing_features.dis,
        housepricing_features.rad,
        housepricing_features.tax,
        housepricing_features.ptratio,
        housepricing_features.b,
        housepricing_features.lstat
    ]
    prediction = predictor.predict([X])
    return JSONResponse(f"Resultado predicción: {prediction}")
