import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
from model.predict import predict_match

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def home():
    """
    Home page route to serve the frontend.
    """
    return render_template('index.html')  # This will serve the HTML file for the home page.

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the winner of a tennis match.
    """
    try:
        # Get data from the request
        data = request.json
        player1 = data.get('player1')
        player2 = data.get('player2')
        surface = data.get('surface')

        # Validate input
        if not player1 or not player2 or not surface:
            return jsonify({'error': 'Missing required fields: player1, player2, surface'}), 400

        # Call the predict_match function
        winner = predict_match(player1, player2, surface)

        # Return the result
        return jsonify({'winner': winner})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)