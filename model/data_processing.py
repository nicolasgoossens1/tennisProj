import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# Database connection setup
DB_FILE = 'C:/Users/nicol/Downloads/tennisProj/data/tennis_database.db'

def get_connection():
    """
    Create and return a database connection.
    """
    return sqlite3.connect(DB_FILE)

# Functions for data processing
def win_percentage(player_name):
    """
    Calculate the win percentage of a given player.
    """
    conn = get_connection()
    query = """
    SELECT 
        CAST(SUM(CASE WHEN Winner = ? THEN 1 ELSE 0 END) AS FLOAT) /
        CAST(COUNT(*) AS FLOAT) AS win_percentage
    FROM atp_tennis_with_loser
    WHERE Winner = ? OR Loser = ?;
    """
    result = pd.read_sql_query(query, conn, params=(player_name, player_name, player_name))
    conn.close()
    return result.iloc[0, 0] if not result.empty else 0.0

def head2head(player1_name, player2_name):
    """
    Calculate the head-to-head record between two players.
    """
    conn = get_connection()
    query = """
    SELECT
        SUM(CASE WHEN Winner = ? AND Loser = ? THEN 1 ELSE 0 END) AS player1_wins,
        SUM(CASE WHEN Winner = ? AND Loser = ? THEN 1 ELSE 0 END) AS player2_wins
    FROM atp_tennis_with_loser
    """
    result = pd.read_sql_query(query, conn, params=(player1_name, player2_name, player2_name, player1_name))
    conn.close()
    return result.iloc[0]['player1_wins'], result.iloc[0]['player2_wins'] if not result.empty else (0, 0)

def average_odds(player1_name, player2_name):
    """
    Calculate the average odds for all matches between two players.
    """
    conn = get_connection()
    query = """
    SELECT 
        AVG(Odd_1) AS avg_odds_player1,
        AVG(Odd_2) AS avg_odds_player2
    FROM atp_tennis_with_loser
    WHERE 
        (Winner = ? AND Loser = ?) OR (Winner = ? AND Loser = ?);
    """
    result = pd.read_sql_query(query, conn, params=(player1_name, player2_name, player2_name, player1_name))
    conn.close()
    if not result.empty:
        return result.iloc[0]['avg_odds_player1'], result.iloc[0]['avg_odds_player2']
    else:
        return None, None

def average_points_difference(player1_name, player2_name):
    """
    Calculate the average points difference between two players.
    """
    conn = get_connection()
    query = """
    SELECT 
        AVG(Pts_1 - Pts_2) AS avg_points_difference
    FROM atp_tennis_with_loser
    WHERE 
        (Winner = ? AND Loser = ?) OR (Winner = ? AND Loser = ?);
    """
    result = pd.read_sql_query(query, conn, params=(player1_name, player2_name, player2_name, player1_name))
    conn.close()
    return result.iloc[0]['avg_points_difference'] if not result.empty else None

def win_percentage_surface(player_name, surface):
    """
    Calculate the win percentage of a player on a specific surface.
    """
    conn = get_connection()
    query = """
    SELECT 
        CAST(SUM(CASE WHEN Winner = ? AND Surface = ? THEN 1 ELSE 0 END) AS FLOAT) /
        CAST(COUNT(*) AS FLOAT) AS win_percentage
    FROM atp_tennis_with_loser
    WHERE 
        (Winner = ? OR Loser = ?) AND Surface = ?;
    """
    result = pd.read_sql_query(query, conn, params=(player_name, surface, player_name, player_name, surface))
    conn.close()
    return result.iloc[0]['win_percentage'] if not result.empty else None

def get_player_ranking(player_name, start_date, end_date):
    """
    Retrieve the ranking of a player over a given time period.
    """
    conn = get_connection()
    query = """
    SELECT
        Date,
        CASE 
            WHEN Player_1 = ? THEN Rank_1
            WHEN Player_2 = ? THEN Rank_2
        END AS Ranking
    FROM atp_tennis_with_loser
    WHERE 
        (Player_1 = ? OR Player_2 = ?) AND Date BETWEEN ? AND ?;
    """
    result = pd.read_sql_query(query, conn, params=(player_name, player_name, player_name, player_name, start_date, end_date))
    conn.close()
    return result if not result.empty else None

def plot_player_ranking(player_name, start_date, end_date):
    """
    Plot the ranking of a player over a given time period.
    """
    ranking_data = get_player_ranking(player_name, start_date, end_date)
    if ranking_data is not None:
        ranking_data['Date'] = pd.to_datetime(ranking_data['Date'])
        ranking_data = ranking_data.sort_values(by='Date')
        
        plt.figure(figsize=(10, 6))
        plt.plot(ranking_data['Date'], ranking_data['Ranking'], marker='o', label=player_name)
        plt.gca().invert_yaxis()  # Invert y-axis since lower rank is better
        plt.title(f"Ranking of {player_name} from {start_date} to {end_date}")
        plt.xlabel("Date")
        plt.ylabel("Ranking")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No ranking data available for {player_name} between {start_date} and {end_date}.")

def recent_win_percentage(player_name, n):
    """
    Calculate the win percentage of a player over their last n matches.
    """
    try:
        conn = get_connection()
        query = """
        SELECT 
            Winner
        FROM atp_tennis_with_loser
        WHERE Winner = ? OR Loser = ?
        ORDER BY Date DESC
        LIMIT ?;
        """
        result = pd.read_sql_query(query, conn, params=(player_name, player_name, n))
        conn.close()
        
        if not result.empty:
            wins = result['Winner'].value_counts().get(player_name, 0)
            return wins / len(result)
        return 0.0
    except Exception as e:
        print(f"Error calculating recent win percentage for {player_name}: {e}")
        return 0.0

