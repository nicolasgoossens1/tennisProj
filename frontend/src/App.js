import React, { useState } from 'react';

function App() {
  const [player1, setPlayer1] = useState('');
  const [player2, setPlayer2] = useState('');
  const [surface, setSurface] = useState('');
  const [winner, setWinner] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player1, player2, surface }),
    });
    const data = await response.json();
    setWinner(data.winner || data.error);
  };

  return (
    <div>
      <h1>Tennis Match Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Player 1"
          value={player1}
          onChange={(e) => setPlayer1(e.target.value)}
        />
        <input
          type="text"
          placeholder="Player 2"
          value={player2}
          onChange={(e) => setPlayer2(e.target.value)}
        />
        <input
          type="text"
          placeholder="Surface"
          value={surface}
          onChange={(e) => setSurface(e.target.value)}
        />
        <button type="submit">Predict Winner</button>
      </form>
      {winner && <h2>Winner: {winner}</h2>}
    </div>
  );
}

export default App;
