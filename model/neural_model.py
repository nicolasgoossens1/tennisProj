import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from prepare_data import fetch_recent_matches, calculate_win_percentages, win_percentage_surface, average_odds, head2head, player_rankings

# Define a simple neural network
class TennisPredictor(nn.Module):
    def __init__(self, input_size):
        super(TennisPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability of player1 winning
        )

    def forward(self, x):
        return self.fc(x)

# Prepare features and labels
def prepare_features(player1, player2, surface, recent_matches):
    win_percentages = calculate_win_percentages(recent_matches)
    player1_recent_win_percentage = win_percentages.get(player1, 0)
    player2_recent_win_percentage = win_percentages.get(player2, 0)
    player1_surface_win_percentage = win_percentage_surface(player1, surface, recent_matches)
    player2_surface_win_percentage = win_percentage_surface(player2, surface, recent_matches)
    player1_avg_odds, player2_avg_odds = average_odds(player1, player2, recent_matches)
    head2head_player1_wins, head2head_player2_wins = head2head(player1, player2, recent_matches)

    features = [
        player1_recent_win_percentage,
        player2_recent_win_percentage,
        player1_surface_win_percentage,
        player2_surface_win_percentage,
        player1_avg_odds or 0,
        player2_avg_odds or 0,
        head2head_player1_wins,
        head2head_player2_wins,
    ]
    rankings = player_rankings(recent_matches)
    player1_rank = rankings.get(player1, len(rankings) + 1)  # Default to lowest rank if not found
    player2_rank = rankings.get(player2, len(rankings) + 1)
    features.extend([player1_rank, player2_rank])
    
    return features

def prepare_dataset(recent_matches):
    data = []
    labels = []
    winner_count = 0
    loser_count = 0

    for _, match in recent_matches.iterrows():
        player1 = match['Winner']
        player2 = match['Loser']
        surface = match['Surface']
        label = 1  # Winner is player1

        # Count winners and losers
        winner_count += 1
        loser_count += 1

        # Positive example: Winner is player1
        features = prepare_features(player1, player2, surface, recent_matches)
        data.append(features)
        labels.append(label)

        # Negative example: Winner is player2 (swap players)
        features = prepare_features(player2, player1, surface, recent_matches)
        data.append(features)
        labels.append(0)

    print(f"Winner count: {winner_count}, Loser count: {loser_count}")
    return data, labels

def main():
    # Fetch recent matches
    recent_matches = fetch_recent_matches(5000)

    # Prepare dataset
    data, labels = prepare_dataset(recent_matches)

    # Normalize features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    print(data[:5])  # Print the first 5 feature sets

    # Convert data and labels to a DataFrame
    dataset_df = pd.DataFrame(data, columns=[
        "Player1_RecentWin%", "Player2_RecentWin%", 
        "Player1_SurfaceWin%", "Player2_SurfaceWin%", 
        "Player1_AvgOdds", "Player2_AvgOdds", 
        "Head2Head_Player1Wins", "Head2Head_Player2Wins",
        "Player1_Rank", "Player2_Rank"
    ])
    dataset_df['Label'] = labels

    # Save the dataset to a CSV file for inspection
    dataset_df.to_csv("dataset.csv", index=False)

    # Print the first few rows of the dataset
    print(dataset_df.head())

    # Split into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for batch training
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    input_size = train_data.shape[1]
    model = TennisPredictor(input_size)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(100):
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            output = model(batch_features)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "neural_model.pth")
    print(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    print("Model saved!")

def evaluate_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        predictions = (predictions > 0.5).float()  # Convert probabilities to binary predictions
        accuracy = (predictions == test_labels).float().mean().item()
        print(f"Test Accuracy: {accuracy:.2%}")

def cross_validate(model, data, labels, k=5):
    """
    Perform k-fold cross-validation.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Convert to tensors
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

        # Train the model
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        for epoch in range(50):  # Shorter training for cross-validation
            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = model(test_data)
            predictions = (predictions > 0.5).float()
            accuracy = (predictions == test_labels).float().mean().item()
            accuracies.append(accuracy)

    print(f"Cross-Validation Accuracy: {sum(accuracies) / len(accuracies):.2%}")

def hyperparameter_tuning(data, labels):
    """
    Perform hyperparameter tuning for the neural network.
    """
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Testing with lr={lr}, batch_size={batch_size}")
            # Train and evaluate the model
            # (Reuse the training loop and evaluation code here)
            # Update best_accuracy and best_params if needed

if __name__ == "__main__":
    main()