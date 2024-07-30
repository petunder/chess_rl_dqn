#validate_dataset.py
import os
import chess
from datasets import load_dataset
import pandas as pd

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self, fen=chess.STARTING_FEN):
        self.board.set_fen(fen)

def validate_dataset(dataset_name):
    sanitized_dataset_name = dataset_name.replace("/", "_")
    invalid_moves_file = f'{sanitized_dataset_name}_invalid_moves_log.txt'
    valid_games_file = f'{sanitized_dataset_name}_valid_games.csv'

    if os.path.exists(invalid_moves_file) and os.path.exists(valid_games_file):
        print("Validation already performed.")
        return

    dataset = load_dataset(dataset_name, split="train[:10]")  # Ограничение на 10 строк
    env = ChessEnv()
    illegal_moves = []
    valid_games = []

    for idx, game in enumerate(dataset):
        raw_moves = game['text']
        moves = raw_moves.replace(';', '').split()
        env.reset()  # Стандартная начальная позиция
        valid = True

        print(f"Game {idx}: {raw_moves}")  # Логирование исходного текста игры

        for move in moves:
            if move[-1].isdigit():  # Пропускаем номера ходов
                continue
            try:
                env.board.push_san(move)
            except ValueError as e:
                illegal_moves.append((idx, move, env.board.fen(), str(e)))
                valid = False
                print(f"Game {idx}: Illegal move '{move}' in position {env.board.fen()} - Error: {e}")
                break
            except chess.IllegalMoveError as e:
                illegal_moves.append((idx, move, env.board.fen(), str(e)))
                valid = False
                print(f"Game {idx}: Illegal move '{move}' in position {env.board.fen()} - Error: {e}")
                break

        if valid:
            valid_games.append(idx)
            print(f"Game {idx}: All moves valid.")

    with open(invalid_moves_file, 'w') as f:
        for game_idx, move, fen, error in illegal_moves:
            f.write(f"Game {game_idx}, illegal move: {move} in FEN: {fen} - Error: {error}\n")

    valid_games_df = pd.DataFrame(valid_games, columns=['valid_game_index'])
    valid_games_df.to_csv(valid_games_file, index=False)
    print(f"Validation completed for dataset {sanitized_dataset_name}:")
    print(f"Total games checked: {len(dataset)}")
    print(f"Valid games: {len(valid_games)}")
    print(f"Illegal moves found: {len(illegal_moves)}")

dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_dataset(dataset_name)

