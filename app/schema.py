from pydantic import BaseModel


class StrokeSchema(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

    class Config:
        orm_mode = True
