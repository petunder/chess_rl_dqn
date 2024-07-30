#validate_dataset.py
import chess
import chess.pgn
from datasets import load_dataset

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self, fen=chess.STARTING_FEN):
        self.board.set_fen(fen)
        print(f"Board reset to initial position:\n{self.board}")

    def push_san(self, move):
        self.board.push_san(move)
        print(f"Move {move} executed successfully\n{self.board}")

def validate_single_game(dataset_name, game_index=0):
    dataset = load_dataset(dataset_name, split="train[:1]")  # Limited to the first game
    env = ChessEnv()
    game = dataset[game_index]
    raw_moves = game['text']
    moves = raw_moves.replace(';', '').split()
    env.reset()

    print(f"Game {game_index}: {raw_moves}")

    for move in moves:
        if move[-1].isdigit():
            continue
        try:
            env.push_san(move)
        except chess.IllegalMoveError as e:
            print(f"Game {game_index}: Illegal move '{move}' in position {env.board.fen()} - Error: {e}")
            return  # Stop processing this game on the first illegal move

    print(f"Game {game_index}: All moves valid.")

dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_single_game(dataset_name)
