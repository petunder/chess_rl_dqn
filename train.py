from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import chess
import torch

import matplotlib
matplotlib.use('Agg')  # Используйте не-интерактивный бэкенд

def state_to_tensor(state):
    return state  # Теперь состояние уже в нужном формате

def action_to_move(action):
    from_square = action // 64
    to_square = action % 64
    return chess.Move(from_square, to_square)

def visualize_training(episode, white_rewards, black_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(white_rewards, label='White')
    plt.plot(black_rewards, label='Black')
    plt.title(f"Rewards over Episodes (Episode {episode})")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(f'training_progress_episode_{episode}.png')
    plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = ChessEnv()
state_size = 64 * 13  # 64 клетки, 13 возможных состояний для каждой клетки
action_size = 64 * 64  # все возможные ходы (из-в)
white_agent = DQNAgent(state_size, action_size, "White")
black_agent = DQNAgent(state_size, action_size, "Black")

num_episodes = 10000
max_steps = 100
white_rewards_history = []
black_rewards_history = []

for episode in range(num_episodes):
    state = env.reset()
    state = state_to_tensor(state)
    white_reward = 0
    black_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        current_player = env.get_current_player()
        agent = white_agent if current_player == chess.WHITE else black_agent

        action = agent.act(state)
        move = action_to_move(action)

        if move in env.board.legal_moves:
            next_state, reward, done, _ = env.step(action)
            reward += 0.1  # Дополнительная награда за корректный ход
        else:
            legal_moves = list(env.board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                action = move.from_square * 64 + move.to_square
                next_state, reward, done, _ = env.step(action)
                reward -= 0.1  # Штраф за выбор некорректного хода
            else:
                done = True
                reward = -1  # Большой штраф за отсутствие легальных ходов

        next_state = state_to_tensor(next_state)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        if current_player == chess.WHITE:
            white_reward += reward
        else:
            black_reward += reward

        state = next_state
        step += 1

    white_agent.update_target_model()
    black_agent.update_target_model()

    white_rewards_history.append(white_reward)
    black_rewards_history.append(black_reward)

    if episode % 100 == 0:
        visualize_training(episode, white_rewards_history, black_rewards_history)
        print(f"Episode: {episode}, White Reward: {white_reward}, Black Reward: {black_reward}")
        print(f"White Epsilon: {white_agent.epsilon}, Black Epsilon: {black_agent.epsilon}")

    if episode % 1000 == 0:
        white_agent.save(f"white_chess_dqn_model_episode_{episode}.pth")
        black_agent.save(f"black_chess_dqn_model_episode_{episode}.pth")

print("Training completed.")
white_agent.save("white_chess_dqn_model_final.pth")
black_agent.save("black_chess_dqn_model_final.pth")