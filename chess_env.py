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

    def reset(self):
        self.board = chess.Board()
        return self._get_observation()

    def step(self, action):
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)

        if move in self.board.legal_moves:
            self.board.push(move)
            done = self.board.is_game_over()
            reward = self._get_reward()
        else:
            done = True
            reward = -1  # Penalty for illegal move

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = np.zeros((8, 8, 13), dtype=np.float32)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                obs[i // 8, i % 8, piece_type + color * 6] = 1
            else:
                obs[i // 8, i % 8, 12] = 1
        return obs

    def _get_reward(self):
        if self.board.is_checkmate():
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0
        else:
            return -0.001  # Небольшой штраф за каждый ход, чтобы стимулировать быструю игру

    def render(self):
        print(self.board)

    def get_current_player(self):
        return int(self.board.turn)