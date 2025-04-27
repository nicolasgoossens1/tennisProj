from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from model.neural_model import TennisPredictor, prepare_features
from model.prepare_data import fetch_recent_matches

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
input_size = 20  # Replace with the actual input size of your model
model = TennisPredictor(input_size)
model.load_state_dict(torch.load("tennis_predictor.pth"))
model.eval()

# Define the input schema
class MatchFeatures(BaseModel):
    player1: str
    player2: str
    surface: str

@app.post("/predict")
def predict(match: MatchFeatures):
    # Fetch recent matches
    recent_matches = fetch_recent_matches(5000)
    if recent_matches.empty:
        raise HTTPException(status_code=400, detail="No recent matches found.")

    # Prepare features for the prediction
    features, _ = prepare_features(recent_matches)
    features_tensor = torch.tensor([features], dtype=torch.float32)

    # Make a prediction
    with torch.no_grad():
        output = model(features_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0  # 1 = Player 1 wins, 0 = Player 2 wins

    return {"probability": probability, "prediction": prediction}