import os
import sqlite3
import pandas as pd

# Define the path to the data folder
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_FILE = os.path.join(DATA_FOLDER, 'tennis_database.db')  # Database file in the data folder

def create_database():
    # Ensure the 'data' folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Iterate over all CSV files in the data folder
    for file_name in os.listdir(DATA_FOLDER):
        if file_name.endswith('.csv'):
            file_path = os.path.join(DATA_FOLDER, file_name)
            table_name = os.path.splitext(file_name)[0]

        # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            if file_name == "atp_tennis.csv":
                df['Loser'] = df.apply(lambda row: row['Player_2'] if row['Winner'] == row['Player_1'] else row['Player_1'],axis=1)
                # Save the updated DataFrame back to a CSV file
                output_path = 'c:/Users/nicol/Downloads/tennisProj/data/atp_tennis_with_loser.csv'
                df.to_csv(output_path, index=False)



       

        # Write the DataFrame to the SQLite database
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Table '{table_name}' created successfully.")

    # Close the connection
    conn.close()
    print(f"Database created successfully at: {DB_FILE}")

if __name__ == "__main__":
    create_database()