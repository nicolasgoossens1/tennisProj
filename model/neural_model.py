import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from prepare_data import fetch_recent_matches, calculate_win_percentages, win_percentage_surface, head2head, calculate_recent_win_percentage

# Define the neural network
class TennisPredictor(nn.Module):
    def __init__(self, input_size):
        super(TennisPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability of Player 1 winning
        )

    def forward(self, x):
        return self.fc(x)

# Prepare features and labels
def prepare_features(matches):
    """
    Prepare features for the neural network.
    """
    features = []
    labels = []

    for _, match in matches.iterrows():
        player1 = match['Player_1']
        player2 = match['Player_2']
        surface = match['Surface']
        round_ = match['Round']
        series = match['Series']
        court = match['Court']

        # Numerical features
        normalized_rank_1 = match['Rank_1'] / 1000
        normalized_rank_2 = match['Rank_2'] / 1000
        normalized_pts_1 = match['Pts_1'] / 1000
        normalized_pts_2 = match['Pts_2'] / 1000
        normalized_odd_1 = match['Odd_1'] / 10
        normalized_odd_2 = match['Odd_2'] / 10

        # Categorical features (one-hot encoding)
        surface_encoded = one_hot_encode(surface, ['Hard', 'Clay', 'Grass', 'Carpet'])
        round_encoded = one_hot_encode(round_, ['R32', 'QF', 'SF', 'F'])
        series_encoded = one_hot_encode(series, ['Grand Slam', 'Masters', 'ATP500', 'ATP250'])
        court_encoded = [1 if court == 'Indoor' else 0]

        # Player stats
        recent_win_pct_1 = calculate_recent_win_percentage(player1, matches)
        recent_win_pct_2 = calculate_recent_win_percentage(player2, matches)
        surface_win_pct_1 = win_percentage_surface(player1, surface, matches)
        surface_win_pct_2 = win_percentage_surface(player2, surface, matches)
        head_to_head_1, head_to_head_2 = head2head(player1, player2, matches)

        # Combine all features
        feature_vector = [
            normalized_rank_1, normalized_rank_2,
            normalized_pts_1, normalized_pts_2,
            normalized_odd_1, normalized_odd_2,
            *surface_encoded, *round_encoded, *series_encoded, *court_encoded,
            recent_win_pct_1, recent_win_pct_2,
            surface_win_pct_1, surface_win_pct_2,
            head_to_head_1, head_to_head_2
        ]
        features.append(feature_vector)

        # Label: 1 if Player_1 is the winner, else 0
        labels.append(1 if match['Winner'] == player1 else 0)

    return np.array(features), np.array(labels)

def one_hot_encode(value, categories):
    """
    One-hot encode a categorical value.
    """
    return [1 if value == category else 0 for category in categories]

def main():
    # Fetch and preprocess data
    print("Fetching recent matches...")
    matches = fetch_recent_matches(5000)

    if matches.empty:
        print("No matches found. Exiting...")
        return

    print("Preparing features and labels...")
    features, labels = prepare_features(matches)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Initialize the model
    input_size = X_train.shape[1]
    model = TennisPredictor(input_size)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("Training the model...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # Evaluate the model
    print("Evaluating the model...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions > 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy:.2%}")

    # Save the model
    torch.save(model.state_dict(), "tennis_predictor.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()