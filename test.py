#test.py
from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import time
import chess
import torch

def state_to_tensor(state):
    return np.reshape(state, (1, -1))[0]

def action_to_move(action):
    from_square = action // 64
    to_square = action % 64
    return chess.Move(from_square, to_square)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = ChessEnv()
state_size = 64 * 13
action_size = 64 * 64
white_agent = DQNAgent(state_size, action_size, "White")
black_agent = DQNAgent(state_size, action_size, "Black")

# Load the trained models
try:
    white_agent.load("white_chess_dqn_model_final.pth")
    black_agent.load("black_chess_dqn_model_final.pth")
    print("Loaded trained models successfully.")
except FileNotFoundError:
    print("Trained models not found. Using untrained agents.")

# Test the trained agents
state = env.reset()
env.render()
done = False
step = 0
max_steps = 100

while not done and step < max_steps:
    current_player = env.get_current_player()
    agent = white_agent if current_player == chess.WHITE else black_agent
    player_name = "White" if current_player == chess.WHITE else "Black"

    state_tensor = state_to_tensor(state)
    action = agent.act(state_tensor)
    move = action_to_move(action)

    print(f"\nStep {step + 1}:")
    print(f"{player_name} agent chose action: {action}")
    print(f"Corresponding move: {move}")

    if move in env.board.legal_moves:
        state, reward, done, _ = env.step(action)
        print(f"Move is legal. Reward: {reward}")
    else:
        print(f"Illegal move! Trying to find a legal move...")
        legal_moves = list(env.board.legal_moves)
        if legal_moves:
            move = np.random.choice(legal_moves)
            action = move.from_square * 64 + move.to_square
            state, reward, done, _ = env.step(action)
            print(f"Chose random legal move: {move}. Reward: {reward}")
        else:
            print(f"No legal moves available. Game over.")
            done = True

    env.render()
    time.sleep(1)
    step += 1

    if done:
        print(f"Game over at step {step}")
        if env.board.is_checkmate():
            print(f"Checkmate! {'Black' if current_player == chess.WHITE else 'White'} wins!")
        elif env.board.is_stalemate():
            print("Stalemate!")
        elif env.board.is_insufficient_material():
            print("Draw due to insufficient material!")
        else:
            print("Game ended for other reasons.")

print("Game Over")