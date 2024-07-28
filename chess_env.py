#chess_env.py
import gym
import numpy as np
import chess
from gym import spaces

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(64 * 64)  # All possible moves (from-to)
        self.observation_space = spaces.Box(low=0, high=12, shape=(8, 8), dtype=np.uint8)
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

    def reset(self):
        self.board = chess.Board()
        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros((13, 8, 8), dtype=np.float32)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                obs[piece_type + color * 6, i // 8, i % 8] = 1
            else:
                obs[12, i // 8, i % 8] = 1
        return obs

    def _get_reward(self):
        reward = 0

        # Награда за шах
        if self.board.is_check():
            reward += 0.5

        # Награда за мат или штраф за пат
        if self.board.is_checkmate():
            return 10 if self.board.turn == chess.BLACK else -10
        elif self.board.is_stalemate():
            return -5

        # Оценка позиции
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        # Подсчет материального преимущества и контроля центра
        central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    reward += value
                else:
                    reward -= value

                # Контроль центра
                if square in central_squares:
                    reward += 0.1 if piece.color == chess.WHITE else -0.1

        # Награда за развитие фигур в начале игры
        if self.board.fullmove_number <= 10:
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                reward += 0.2 * len(self.board.pieces(piece_type, chess.WHITE))
                reward -= 0.2 * len(self.board.pieces(piece_type, chess.BLACK))

        # Штраф за неразвитые фигуры в середине игры
        if self.board.fullmove_number > 10:
            undeveloped_pieces = (
                    len(self.board.pieces(chess.KNIGHT, chess.WHITE)) +
                    len(self.board.pieces(chess.BISHOP, chess.WHITE)) -
                    len(self.board.pieces(chess.KNIGHT, chess.BLACK)) -
                    len(self.board.pieces(chess.BISHOP, chess.BLACK))
            )
            reward -= 0.1 * undeveloped_pieces

        # Награда за продвижение пешек
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE:
                    reward += 0.05 * (rank - 1)
                else:
                    reward += 0.05 * (6 - rank)

        # Нормализация награды
        return reward / 100

    def step(self, action):
        # Сохраняем предыдущее состояние доски для сравнения
        previous_board = self.board.copy()

        move = chess.Move(action // 64, action % 64)
        if move in self.board.legal_moves:
            self.board.push(move)
            done = self.board.is_game_over()
            reward = self._get_reward()

            # Штраф за повторение ходов
            if self.board.is_repetition(2):
                reward -= 0.1

            # Дополнительная награда за взятие фигуры
            if previous_board.piece_at(move.to_square) is not None:
                captured_piece_value = self.piece_values.get(previous_board.piece_at(move.to_square).piece_type, 0)
                reward += captured_piece_value / 10  # Делим на 10, чтобы не перевешивать другие награды
        else:
            done = True
            reward = -1  # Штраф за нелегальный ход

        return self._get_observation(), reward, done, {}
    def render(self):
        print(self.board)

    def get_current_player(self):
        return int(self.board.turn)