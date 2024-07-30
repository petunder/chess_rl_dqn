#validate_dataset.py
import chess
import chess.pgn
from datasets import load_dataset
import re

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        print(f"Board reset to initial position:\n{self.board}")

    def push_uci(self, move):
        try:
            self.board.push_uci(move)
            print(f"Move {move} executed successfully. Current board:\n{self.board}")
        except Exception as e:
            print(f"Error executing move {move}: {str(e)}")
            raise e

def validate_single_game(moves, termination, result, game_index=0):
    env = ChessEnv()
    env.reset()

    print(f"Game {game_index}:")
    for move in moves:
        env.push_uci(move)

    # Checking game termination condition
    termination_status = {
        1: env.board.is_checkmate(),
        2: env.board.is_stalemate(),
        3: env.board.is_insufficient_material(),
        4: env.board.is_seventyfive_moves(),
        5: env.board.is_fivefold_repetition(),
        6: env.board.can_claim_fifty_moves(),
        7: env.board.can_claim_threefold_repetition(),
        8: env.board.is_variant_win(),
        9: env.board.is_variant_loss(),
        10: env.board.is_variant_draw()
    }

    print(f"Termination condition met: {termination_status.get(termination, 'Unknown termination')}")
    print(f"Result of the game: {result}")

def validate_multiple_games(dataset_name):
    # Загрузка данных в режиме потока
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.take(10)  # Берем только первые 10 игр
    for i, game in enumerate(dataset):
        validate_single_game(game['Moves'], game['Termination'], game['Result'], game_index=i)


dataset_name = "laion/strategic_game_chess"
validate_multiple_games(dataset_name)

