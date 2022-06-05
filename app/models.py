from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class StrokeModel(Base):
    __tablename__ = "stroke_predictions"

    # Identification
    id = Column(Integer, primary_key=True, index=True)

    # Input Features
    gender = Column(String)
    age = Column(Integer)
    hypertension = Column(Integer)
    heart_disease = Column(Integer)
    ever_married = Column(String)
    work_type = Column(String)
    residence_type = Column(String)
    avg_glucose_level = Column(Float)
    bmi = Column(Float)
    smoking_status = Column(String)

    # Pipeline Predictions
    stroke_prediction = Column(Integer)
    stroke_prediction_probability = Column(Float)

    # Logging
    time_created = Column(DateTime(timezone=True), server_default=func.now())
