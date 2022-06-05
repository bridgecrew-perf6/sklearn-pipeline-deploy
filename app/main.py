import logging
import os
import sys

import joblib
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware, db

from models import StrokeModel
from schema import StrokeSchema

# Configure Logging
log_format = (
    "[%(asctime)s] - p%(process)s %(name)s %(lineno)d - %(levelname)s:%(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


# Load Environment File
load_dotenv(".env")


# Configure FastAPI
app = FastAPI()
app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])


@app.get("/")
async def index():
    return {"message": "Hello, world."}


@app.post("/predict/")
async def predict(features: StrokeSchema):

    features_dict = features.dict()  # Convert JSON to Python dictionary object.
    features_dataframe = pd.DataFrame(
        [features_dict]
    )  # Create Pandas DataFrame from dictionary.
    print(features_dataframe)
    pipeline = joblib.load(
        os.environ["PIPELINE_PATH"]
    )  # Load pipeline using environment file path.
    prediction = pipeline.predict(features_dataframe)[
        0
    ].item()  # Generate prediction using features_dataframe.
    try:
        prediction_probability = (
            pipeline.predict_proba(features_dataframe)[0].max().item()
        )  # Generate prediction probability using features_dataframe.
    except AttributeError:
        logger.error("Method predict_proba is not available for selected model architecture.")
        prediction_probability = None  # Use dummy value.

    db_results = StrokeModel(
        gender=features.gender,
        age=features.age,
        hypertension=features.hypertension,
        heart_disease=features.heart_disease,
        ever_married=features.ever_married,
        work_type=features.work_type,
        residence_type=features.residence_type,
        avg_glucose_level=features.avg_glucose_level,
        bmi=features.bmi,
        smoking_status=features.smoking_status,
        stroke_prediction=prediction,
        stroke_prediction_probability=prediction_probability,
    )

    db.session.add(db_results)  # Add results to database.
    db.session.commit()  # Commit results to database.

    return prediction_probability


# Running FastAPI Server Locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
