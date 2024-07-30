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


def validate_single_game(game_text, game_index=0):
    env = ChessEnv()

    # Удаление начальной точки с запятой, номеров ходов и точек
    cleaned_moves = re.sub(r'\d+\.', '', game_text.replace(';', '')).split()

    # Удаление элементов, состоящих только из цифр
    filtered_moves = [move for move in cleaned_moves if not move.isdigit()]

    env.reset()  # Сброс доски в начальное положение

    print(f"Game {game_index}: {game_text}")
    print(f"Processed moves: {filtered_moves}")  # Вывод обработанных ходов для проверки

    for move in filtered_moves:
        print(f"Processing move: {move} on board:\n{env.board}")
        try:
            env.push_san(move)
        except chess.IllegalMoveError as e:
            print(f"Game {game_index}: Illegal move '{move}' - Error: {e}")
            break

    print(f"Game {game_index}: Processing complete.")


def validate_multiple_games(dataset_name):
    dataset = load_dataset(dataset_name, split="train[:10]")  # Загружаем первые 10 игр
    for i, game in enumerate(dataset):
        validate_single_game(game['text'], game_index=i)


dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_multiple_games(dataset_name)
