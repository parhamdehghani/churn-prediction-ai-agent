import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Pydantic Model for Input Data Validation ---
class ChurnFeatures(BaseModel):
    transaction_count: int | None = None
    total_plan_days: int | None = None
    total_amount_paid: float | None = None
    total_songs_completed: int | None = None
    total_songs_985_completed: int | None = None
    total_unique_songs: int | None = None
    total_secs_played: float | None = None
    listening_day_count: int | None = None
    age_cleaned: int | None = None
    is_male: int
    is_female: int

# --- 2. Load the MLflow Pipeline Model ---
# This assumes the MLflow server is running
mlflow.set_tracking_uri("http://mlflow:5000")
logged_model_uri = "runs:/046054e68a604cd4a19760cfe140e180/spark-xgb-pipeline-model-best"
model = mlflow.pyfunc.load_model(logged_model_uri)

# --- 3. Create the FastAPI App ---
app = FastAPI(title="Churn Prediction API")

@app.get("/health", tags=["Health Check"])
def health_check():
    """Health check endpoint to ensure the API is running."""
    return {"status": "ok"}

@app.post("/predict", tags=["Prediction"])
def predict(features: ChurnFeatures):
    """
    Takes user features and returns a churn prediction (1 for churn, 0 for stay).
    """
    # Convert the input to a Pandas DataFrame, as expected by the model
    feature_df = pd.DataFrame([features.dict()])
    
    # Make a prediction
    prediction = model.predict(feature_df)
    
    # Return the prediction as an integer
    return {"churn_prediction": int(prediction[0])}
