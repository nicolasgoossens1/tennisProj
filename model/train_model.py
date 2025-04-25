import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from prepare_data import fetch_recent_matches, calculate_win_percentages
from src.data_processing import get_connection
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_random_forest():
    """
    Train a Random Forest model to predict match winners.
    """
    # Load match data
    logging.info("Loading match data from the database...")
    conn = get_connection()
    matches = pd.read_sql_query("SELECT * FROM atp_tennis_with_loser", conn)
    conn.close()
    logging.info(f"Loaded {len(matches)} matches.")

    # Fetch recent matches
    n = 5000  # Number of recent matches to fetch
    logging.info(f"Fetching the most recent {n} matches...")
    recent_matches = fetch_recent_matches(n)

    if recent_matches.empty:
        logging.error("No recent matches found. Exiting.")
        return

    # Calculate win percentages
    logging.info("Calculating win percentages for players...")
    win_percentages = calculate_win_percentages(recent_matches)

    # Add features to the dataset
    logging.info("Adding features to the dataset...")
    matches['recent_win_percentage_player1'] = matches['Player_1'].map(win_percentages).fillna(0)
    matches['recent_win_percentage_player2'] = matches['Player_2'].map(win_percentages).fillna(0)
    matches['player1_surface_win_percentage'] = 0  # Placeholder for surface-specific win percentage
    matches['player2_surface_win_percentage'] = 0  # Placeholder for surface-specific win percentage
    matches['player1_avg_odds'] = 0  # Placeholder for average odds
    matches['player2_avg_odds'] = 0  # Placeholder for average odds
    matches['head2head_player1_wins'] = 0  # Placeholder for head-to-head wins
    matches['head2head_player2_wins'] = 0  # Placeholder for head-to-head wins

    # Add the target variable
    matches['winner'] = (matches['Winner'] == matches['Player_1']).astype(int)

    # Drop irrelevant columns
    logging.info("Dropping irrelevant columns...")
    X = matches[[
        'recent_win_percentage_player1',
        'recent_win_percentage_player2',
        'player1_surface_win_percentage',
        'player2_surface_win_percentage',
        'player1_avg_odds',
        'player2_avg_odds',
        'head2head_player1_wins',
        'head2head_player2_wins'
    ]]
    y = matches['winner']

    # Split the dataset into training and testing sets
    logging.info("Splitting the dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # Train the Random Forest model
    logging.info("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.2f}")

    # Save the model and feature names
    logging.info("Saving the model and feature names...")
    joblib.dump(model, 'tennis_model.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    logging.info("Model and feature names saved successfully.")

if __name__ == "__main__":
    train_random_forest()