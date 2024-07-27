#test.py
from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import time
import chess
import torch


def visualize_board(board):
    unicode_pieces = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
    }

    print("  a b c d e f g h")
    print(" +-----------------+")
    for i in range(8):
        print(f"{8 - i}|", end="")
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7 - i))
            if piece:
                print(f"{unicode_pieces[piece.symbol()]} ", end="")
            else:
                print(". ", end="")
        print(f"|{8 - i}")
    print(" +-----------------+")
    print("  a b c d e f g h")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = ChessEnv()
white_agent = DQNAgent("White")
black_agent = DQNAgent("Black")

# Load the trained models
try:
    white_agent.load("white_chess_dqn_model_final.pth")
    black_agent.load("black_chess_dqn_model_final.pth")
    print("Loaded trained models successfully.")
except FileNotFoundError:
    print("Trained models not found. Using untrained agents.")

# Test the trained agents
state = env.reset()
visualize_board(env.board)
done = False
step = 0
max_steps = 100

while not done and step < max_steps:
    current_player = env.get_current_player()
    agent = white_agent if current_player == chess.WHITE else black_agent
    player_name = "White" if current_player == chess.WHITE else "Black"

    action = agent.act(state, env.board)

    print(f"\nStep {step + 1}:")
    print(f"{player_name} agent chose move: {action}")

    state, reward, done, _ = env.step(action.from_square * 64 + action.to_square)
    print(f"Reward: {reward}")

    visualize_board(env.board)
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