#train.py
from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import chess
import torch
import random

import matplotlib

matplotlib.use('Agg')  # Используйте не-интерактивный бэкенд


def state_to_tensor(state):
    return state.flatten()


def action_to_move(action):
    from_square = action // 64
    to_square = action % 64
    return chess.Move(from_square, to_square)


def choose_legal_move(board, q_values):
    legal_moves = list(board.legal_moves)
    legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
    legal_q_values = q_values[legal_move_indices]
    best_move_index = np.argmax(legal_q_values)
    return legal_moves[best_move_index], 0


def visualize_training(episode, white_rewards, black_rewards, white_legal_moves, black_legal_moves):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(white_rewards, label='White')
    ax1.plot(black_rewards, label='Black')
    ax1.set_title(f"Rewards over Episodes (Episode {episode})")
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()

    ax2.plot(white_legal_moves, label='White Legal Moves')
    ax2.plot(black_legal_moves, label='Black Legal Moves')
    ax2.set_title("Legal Moves per Episode")
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Legal Moves')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'training_progress_episode_{episode}.png')
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = ChessEnv()
state_size = 8 * 8 * 13  # 8x8 доска, 13 возможных состояний для каждой клетки
action_size = 64 * 64  # все возможные ходы (из-в)
white_agent = DQNAgent("White")
black_agent = DQNAgent("Black")

num_episodes = 10000
max_steps = 100
white_rewards_history = []
black_rewards_history = []
white_legal_moves_history = []
black_legal_moves_history = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        current_player = env.get_current_player()
        agent = white_agent if current_player == chess.WHITE else black_agent

        action = agent.act(state, env.board)
        next_state, reward, done, _ = env.step(action.from_square * 64 + action.to_square)

        agent.remember(state, action.from_square * 64 + action.to_square, reward, next_state, done)
        agent.replay()

        state = next_state
        step += 1

    white_agent.update_target_model()
    black_agent.update_target_model()

    white_rewards_history.append(white_reward)
    black_rewards_history.append(black_reward)
    white_legal_moves_history.append(white_legal_moves)
    black_legal_moves_history.append(black_legal_moves)

    if episode % 100 == 0:
        visualize_training(episode, white_rewards_history, black_rewards_history, white_legal_moves_history,
                           black_legal_moves_history)
        print(f"Episode: {episode}, White Reward: {white_reward}, Black Reward: {black_reward}")
        print(f"White Legal Moves: {white_legal_moves}, Black Legal Moves: {black_legal_moves}")
        print(f"White Epsilon: {white_agent.epsilon}, Black Epsilon: {black_agent.epsilon}")
        print(f"White Average Weights: {white_agent.get_average_weights()}")
        print(f"Black Average Weights: {black_agent.get_average_weights()}")
        print(f"White Average Loss: {np.mean(white_agent.losses[-100:])}")
        print(f"Black Average Loss: {np.mean(black_agent.losses[-100:])}")
        white_agent.losses = []
        black_agent.losses = []

    if episode % 1000 == 0:
        white_agent.save(f"white_chess_dqn_model_episode_{episode}.pth")
        black_agent.save(f"black_chess_dqn_model_episode_{episode}.pth")

print("Training completed.")
white_agent.save("white_chess_dqn_model_final.pth")
black_agent.save("black_chess_dqn_model_final.pth")