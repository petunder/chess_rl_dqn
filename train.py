# train.py
from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import chess
import torch
import random
import matplotlib
import os
from datetime import datetime

matplotlib.use('Agg')  # Используйте не-интерактивный бэкенд


def save_game(moves, episode, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"game_{episode:07d}.txt")
    with open(filename, "w") as f:
        f.write(" ".join(moves))


def state_to_tensor(state):
    return state  # Теперь состояние уже имеет правильную форму (13, 8, 8)


def action_to_move(action):
    from_square = action // 64
    to_square = action % 64
    return chess.Move(from_square, to_square)


def choose_legal_move(board, policy):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, -1  # Нет легальных ходов, игра окончена

    legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]

    print(f"Policy shape: {policy.shape}")
    print(f"Max legal move index: {max(legal_move_indices)}")

    if max(legal_move_indices) >= policy.shape[0]:
        print("Warning: Legal move index out of bounds. Using random move.")
        return random.choice(legal_moves), 0

    legal_policy_values = policy[legal_move_indices].cpu()  # Move to CPU

    if legal_policy_values.numel() == 0:
        print("Warning: No legal moves found in policy")
        return random.choice(legal_moves), 0

    if torch.isnan(legal_policy_values).any() or torch.isinf(legal_policy_values).any():
        print("Warning: NaN or Inf values in policy")
        return random.choice(legal_moves), 0

    try:
        best_move_index = torch.argmax(legal_policy_values).item()
    except RuntimeError as e:
        print(f"Error in choose_legal_move: {e}")
        print(f"legal_policy_values: {legal_policy_values}")
        return random.choice(legal_moves), 0

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
white_agent = DQNAgent("White")
black_agent = DQNAgent("Black")

num_episodes = 10000
max_steps = 100
white_rewards_history = []
black_rewards_history = []
white_legal_moves_history = []
black_legal_moves_history = []

folder_name = datetime.now().strftime("training/%d_%m_%Y_%H_%M")

for episode in range(num_episodes):
    state = env.reset()
    state = state_to_tensor(state)
    done = False
    step = 0
    white_reward = 0
    black_reward = 0
    white_legal_moves = 0
    black_legal_moves = 0
    white_illegal_moves = 0
    black_illegal_moves = 0
    moves = []

    while not done and step < max_steps:
        current_player = env.get_current_player()
        agent = white_agent if current_player == chess.WHITE else black_agent

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, _ = agent.model(state_tensor)

        action, _ = choose_legal_move(env.board, policy.squeeze())
        print(
            f"Episode {episode}, Step {step}: {'White' if current_player == chess.WHITE else 'Black'} chooses action {action}")

        next_state, reward, done, _ = env.step(action.from_square * 64 + action.to_square)
        next_state = state_to_tensor(next_state)
        moves.append(action.uci())

        if current_player == chess.WHITE:
            white_reward += reward
            white_legal_moves += 1
        else:
            black_reward += reward
            black_legal_moves += 1

        print(f"Reward: {reward}, Done: {done}")
        print(f"Board state:\n{env.board}")

        agent.remember(state, action.from_square * 64 + action.to_square, reward, next_state, done)
        agent.replay()

        state = next_state
        step += 1

    save_game(moves, episode, folder_name)

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
        print(f"White Legal Moves: {white_legal_moves}, White Illegal Moves: {white_illegal_moves}")
        print(f"Black Legal Moves: {black_legal_moves}, Black Illegal Moves: {black_illegal_moves}")
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