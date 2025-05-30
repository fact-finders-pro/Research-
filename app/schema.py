from pydantic import BaseModel

class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add additional features as needed