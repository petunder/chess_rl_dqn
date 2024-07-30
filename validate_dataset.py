#validate_dataset.py
import chess
import chess.pgn
from datasets import load_dataset
import re  # Используем для регулярных выражений


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self, fen=chess.STARTING_FEN):
        self.board.set_fen(fen)
        print(f"Board reset to initial position:\n{self.board}")

    def push_san(self, move):
        try:
            self.board.push_san(move)
            print(f"Move {move} executed successfully. Current board:\n{self.board}")
        except Exception as e:
            print(f"Error executing move {move}: {str(e)}")
            raise e


def validate_single_game(dataset_name, game_index=0):
    dataset = load_dataset(dataset_name, split="train[:1]")
    env = ChessEnv()
    game = dataset[game_index]
    raw_moves = game['text']

    # Удаление начальной точки с запятой, номеров ходов и точек
    cleaned_moves = re.sub(r'\d+\.', '', raw_moves.replace(';', '')).split()

    env.reset()  # Сброс доски в начальное положение

    print(f"Game {game_index}: {raw_moves}")
    print(f"Processed moves: {cleaned_moves}")  # Вывод обработанных ходов для проверки

    for move in cleaned_moves:
        print(f"Processing move: {move} on board:\n{env.board}")
        try:
            env.push_san(move)
        except chess.IllegalMoveError as e:
            print(f"Game {game_index}: Illegal move '{move}' - Error: {e}")
            break

    print(f"Game {game_index}: Processing complete.")


dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_single_game(dataset_name)
