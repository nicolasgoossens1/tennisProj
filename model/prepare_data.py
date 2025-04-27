import os
import sys
import pandas as pd
from tqdm import tqdm

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data_processing import get_connection

def fetch_recent_matches(n):
    try:
        conn = get_connection()
        query = """
        SELECT 
            Player_1, 
            Player_2, 
            Date, 
            Surface, 
            Round, 
            Series, 
            Court, 
            Rank_1, 
            Rank_2, 
            Pts_1, 
            Pts_2, 
            Odd_1, 
            Odd_2,
            Winner,
            Loser
        FROM atp_tennis_with_loser
        ORDER BY Date DESC
        LIMIT ?;
        """
        recent_matches = pd.read_sql_query(query, conn, params=(n,))
        conn.close()

        print("Fetched matches shape:", recent_matches.shape)
        print("First few rows of fetched matches:")
        print(recent_matches.head())

        # Filter invalid matches
        recent_matches = filter_invalid_matches(recent_matches)

        print("Filtered matches shape:", recent_matches.shape)
        return recent_matches
    except Exception as e:
        print(f"Error fetching recent matches: {e}")
        return pd.DataFrame()

def filter_invalid_matches(recent_matches):
    """
    Remove matches with invalid or extreme values.
    """
    # Drop rows with missing values in critical columns
    recent_matches = recent_matches.dropna(subset=['Winner', 'Loser', 'Surface', 'Odd_1', 'Odd_2'])

    # Remove matches with invalid or nonsensical odds
    recent_matches = recent_matches[(recent_matches['Odd_1'] > 0) & (recent_matches['Odd_2'] > 0)]

    # Optional: Remove matches with extreme odds (e.g., odds > 100)
    recent_matches = recent_matches[(recent_matches['Odd_1'] <= 100) & (recent_matches['Odd_2'] <= 100)]

    return recent_matches


def calculate_recent_win_percentage(player, recent_matches):
    """
    Calculate the win percentage of a player over their last n matches.
    """
    # Fetch matches for the player
    player_matches = recent_matches[
        ((recent_matches['Winner'] == player) | (recent_matches['Loser'] == player))
    ]
    if player_matches.empty:
        return 0  # No matches found

    # Calculate win percentage
    wins = player_matches['Winner'].value_counts().get(player, 0)
    return wins / len(player_matches)



def calculate_win_percentages(recent_matches):
    """
    Calculate win percentages for all players in the given matches.
    """
    # Count total matches and wins for each player
    total_matches = pd.concat([
        recent_matches[['Winner']].rename(columns={'Winner': 'Player'}),
        recent_matches[['Loser']].rename(columns={'Loser': 'Player'})
    ]).value_counts().reset_index(name='total_matches')

    total_wins = recent_matches['Winner'].value_counts().reset_index(name='wins')
    total_wins.columns = ['Player', 'wins']

    # Merge total matches and wins, then calculate win percentage
    win_percentages = pd.merge(total_matches, total_wins, on='Player', how='left').fillna(0)
    win_percentages['win_percentage'] = win_percentages['wins'] / win_percentages['total_matches']

    # Normalize win percentages for neural network usage
    win_percentages['win_percentage'] = win_percentages['win_percentage'].clip(0, 1)

    # Convert to dictionary
    return win_percentages.set_index('Player')['win_percentage'].to_dict()

def win_percentage_surface(player, surface, recent_matches):
    """
    Calculate the win percentage of a player on a specific surface.
    """
    # Fetch matches for the player on the given surface
    player_matches = recent_matches[
        ((recent_matches['Winner'] == player) | (recent_matches['Loser'] == player)) &
        (recent_matches['Surface'] == surface)
    ]
    if player_matches.empty:
        return 0  # No matches found on this surface

    # Calculate win percentage
    wins = player_matches['Winner'].value_counts().get(player, 0)
    return wins / len(player_matches)

def average_odds(player1, player2, recent_matches):
    """
    Calculate the average odds for two players based on Odd_1 and Odd_2.
    """
    # Fetch matches involving both players
    matches = recent_matches[
        (recent_matches['Winner'].isin([player1, player2])) |
        (recent_matches['Loser'].isin([player1, player2]))
    ]
    if matches.empty:
        return (None, None)  # No matches found

    # Calculate average odds for each player
    player1_odds = matches[matches['Winner'] == player1]['Odd_1'].mean()
    player2_odds = matches[matches['Winner'] == player2]['Odd_1'].mean()

    # Include odds when the player is the loser
    player1_odds_loser = matches[matches['Loser'] == player1]['Odd_2'].mean()
    player2_odds_loser = matches[matches['Loser'] == player2]['Odd_2'].mean()

    # Combine odds from both winner and loser perspectives
    player1_avg_odds = pd.Series([player1_odds, player1_odds_loser]).mean()
    player2_avg_odds = pd.Series([player2_odds, player2_odds_loser]).mean()

    return (player1_avg_odds, player2_avg_odds)

def head2head(player1, player2, recent_matches):
    """
    Calculate head-to-head wins between two players.
    """
    # Fetch matches between the two players
    matches = recent_matches[
        ((recent_matches['Winner'] == player1) & (recent_matches['Loser'] == player2)) |
        ((recent_matches['Winner'] == player2) & (recent_matches['Loser'] == player1))
    ]
    if matches.empty:
        return (0, 0)  # No head-to-head matches found

    # Calculate head-to-head wins
    player1_wins = matches['Winner'].value_counts().get(player1, 0)
    player2_wins = matches['Winner'].value_counts().get(player2, 0)
    return (player1_wins, player2_wins)

def main():
    n = 10000  # Number of recent matches to fetch
    recent_matches = fetch_recent_matches(n)

    if recent_matches.empty:
        print("No matches found.")
        return

    # Calculate win percentages
    win_percentages = calculate_win_percentages(recent_matches)

    # Print the results
    print("Win Percentages:")
    for player, percentage in win_percentages.items():
        print(f"{player}: {percentage:.2%}")

if __name__ == "__main__":
    main()
