import pandas as pd
import joblib
from .prepare_data import fetch_recent_matches, calculate_win_percentages, win_percentage_surface, average_odds, head2head
from src.data_processing import get_connection

# Load the trained model and feature names
model = joblib.load('tennis_model.pkl')
feature_names = joblib.load('feature_names.pkl')

def predict_match(player1, player2, surface, n_recent_matches=5000):
    """
    Predict the winner of a match.
    """
    # Fetch recent matches and calculate win percentages
    recent_matches = fetch_recent_matches(n_recent_matches)
    if recent_matches.empty:
        raise ValueError("No recent matches found. Cannot calculate win percentages.")

    win_percentages = calculate_win_percentages(recent_matches)
   

    # Get recent win percentages for both players
    player1_recent_win_percentage = win_percentages.get(player1, 0)
    player2_recent_win_percentage = win_percentages.get(player2, 0)

    # Calculate additional features
    player1_surface_win_percentage = win_percentage_surface(player1, surface, recent_matches)
    player2_surface_win_percentage = win_percentage_surface(player2, surface, recent_matches)
    player1_avg_odds, player2_avg_odds = average_odds(player1, player2, recent_matches)
    head2head_player1_wins, head2head_player2_wins = head2head(player1, player2, recent_matches)

    # Create the feature dictionary
    features = {
        'recent_win_percentage_player1': player1_recent_win_percentage,
        'recent_win_percentage_player2': player2_recent_win_percentage,
        'player1_surface_win_percentage': player1_surface_win_percentage,
        'player2_surface_win_percentage': player2_surface_win_percentage,
        'player1_avg_odds': player1_avg_odds or 0,
        'player2_avg_odds': player2_avg_odds or 0,
        'head2head_player1_wins': head2head_player1_wins,
        'head2head_player2_wins': head2head_player2_wins,
    }

    # Debug: Print the feature dictionary
    print("Feature dictionary:")
    print(features)

    # Convert features to a DataFrame
    features_df = pd.DataFrame([features])

    # Align features with the model's expected feature names
    #features_df = features_df.reindex(columns=feature_names, fill_value=0)

    # Debug: Print the aligned feature DataFrame
    print("Aligned feature DataFrame:")
    print(features_df)

    # Make the prediction
    prediction = model.predict(features_df)
    return player1 if prediction[0] == 1 else player2

# Example usage
if __name__ == "__main__":
    player1 = "Sinner J."
    player2 = "Alcaraz C."
    surface = "Hard"
    try:
        winner = predict_match(player1, player2, surface)
        print(f"Predicted winner: {winner}")
    except ValueError as e:
        print(f"Error: {e}")