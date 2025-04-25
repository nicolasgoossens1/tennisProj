from src.data_processing import (
    head2head, win_percentage, average_odds, average_points_difference,
    win_percentage_surface, plot_player_ranking
)

# Example usage
player1 = "Federer R."
player2 = "Nadal R."

# Head-to-head record
player1_wins, player2_wins = head2head(player1, player2)
print(f"Head-to-head record between {player1} and {player2}: {player1_wins} - {player2_wins}")

# Win percentage
print(f"Win percentage of {player1}: {win_percentage(player1) * 100:.2f}%")

# Average odds
avg_odds_player1, avg_odds_player2 = average_odds(player1, player2)
print(f"Average odds for {player1}: {avg_odds_player1}")
print(f"Average odds for {player2}: {avg_odds_player2}")

# Average points difference
avg_points_diff = average_points_difference(player1, player2)
print(f"Average points difference between {player1} and {player2}: {avg_points_diff}")

# Win percentage on surface
surface = "Hard"
print(f"Win percentage of {player1} on {surface} surface: {win_percentage_surface(player1, surface) * 100:.2f}%")

# Plot player ranking
start_date = "2010-01-01"
end_date = "2018-12-31"
plot_player_ranking(player1, start_date, end_date)