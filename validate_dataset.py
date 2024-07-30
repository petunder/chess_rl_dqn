#validate_dataset.py
import os
import chess
import chess.pgn
from datasets import load_dataset
import pandas as pd

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
    dataset = load_dataset(dataset_name, split="train[:1]")  # Ограничение на одну строку
    env = ChessEnv()

    game = dataset[game_index]
    raw_moves = game['text']
    moves = raw_moves.replace(';', '').split()
    env.reset()  # Стандартная начальная позиция
    valid = True

    print(f"Game {game_index}: {raw_moves}")  # Логирование исходного текста игры

    for move in moves:
        if move[-1].isdigit():  # Пропускаем номера ходов
            continue

        try:
            print(f"Processing move: {move} on board:\n{env.board}")
            env.push_san(move)
        except ValueError as e:
            print(f"Game {game_index}: Illegal move '{move}' in position {env.board.fen()} - Error: {e}")
            valid = False
            break
        except chess.IllegalMoveError as e:
            print(f"Game {game_index}: Illegal move '{move}' in position {env.board.fen()} - Error: {e}")
            valid = False
            break

    if valid:
        print(f"Game {game_index}: All moves valid.")

dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
validate_single_game(dataset_name)
