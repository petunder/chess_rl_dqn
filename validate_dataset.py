import os
import chess
from datasets import load_dataset
import pandas as pd
from chess_common import ChessEnv


def validate_dataset(dataset_name):
    # Пути к файлам результатов проверки
    invalid_moves_file = f'{dataset_name}_invalid_moves_log.txt'
    valid_games_file = f'{dataset_name}_valid_games.csv'

    # Проверяем, существуют ли уже файлы результатов проверки
    if os.path.exists(invalid_moves_file) and os.path.exists(valid_games_file):
        # Читаем данные о валидных играх
        valid_games_df = pd.read_csv(valid_games_file)
        num_valid_games = len(valid_games_df)
        # Читаем данные о невалидных ходах
        with open(invalid_moves_file, 'r') as f:
            invalid_moves = f.readlines()
        num_invalid_moves = len(invalid_moves)

        print(f"Validation already performed:")
        print(f"Valid games: {num_valid_games}")
        print(f"Illegal moves found: {num_invalid_moves}")
        return

    # Загрузка датасета
    dataset = load_dataset(dataset_name, split="train[:10000]")

    env = ChessEnv()
    illegal_moves = []
    valid_games = []

    for idx, game in enumerate(dataset):
        game_moves = game['text'].split()[1::2]
        env.reset()
        valid = True
        for move in game_moves:
            try:
                chess_move = env.board.parse_san(move)
                env.board.push(chess_move)
            except chess.IllegalMoveError:
                illegal_moves.append((idx, move, env.board.fen()))
                valid = False
                break
        if valid:
            valid_games.append(idx)

    # Сохранение результатов в файлы
    with open(invalid_moves_file, 'w') as f:
        for game_idx, move, fen in illegal_moves:
            f.write(f"Game {game_idx}, illegal move: {move} in FEN: {fen}\n")

    valid_games_df = pd.DataFrame(valid_games, columns=['valid_game_index'])
    valid_games_df.to_csv(valid_games_file, index=False)

    # Вывод результатов в консоль
    print(f"Validation completed for dataset {dataset_name}:")
    print(f"Total games checked: {len(dataset)}")
    print(f"Valid games: {len(valid_games)}")
    print(f"Illegal moves found: {len(illegal_moves)}")


# Название датасета, который вы хотите проверить
dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_dataset(dataset_name)
