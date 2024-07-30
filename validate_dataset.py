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
        try:
            self.board.push_san(move)
            print(f"Move {move} executed successfully. Current FEN: {self.board.fen()}")
        except Exception as e:
            print(f"Error executing move {move}: {str(e)}")
            raise e


def validate_single_game(dataset_name, game_index=0):
    dataset = load_dataset(dataset_name, split="train[:1]")  # Limited to the first game
    env = ChessEnv()
    game = dataset[game_index]
    raw_moves = game['text']
    # Удаление точек после номеров и разбиение на ходы
    moves = raw_moves.replace(';', '').replace('.', ' ').split()

    env.reset()  # Reset the board to the starting position only once

    print(f"Game {game_index}: {raw_moves}")  # Log the original game text

    for move in moves:
        if move[-1].isdigit():  # Skip move numbers, ensure moves are purely algebraic
            continue
        print(f"Processing move: {move} on board:\n{env.board}")
        try:
            env.push_san(move)
        except chess.IllegalMoveError as e:
            print(f"Game {game_index}: Illegal move '{move}' - Error: {e}")
            break

    print(f"Game {game_index}: Processing complete.")


dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_single_game(dataset_name)
