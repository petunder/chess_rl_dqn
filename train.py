from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def state_to_tensor(state):
    return np.reshape(state, (1, -1))[0]


def visualize_training(episode, white_rewards, black_rewards):
    clear_output(wait=True)
    plt.figure(figsize=(12, 6))
    plt.plot(white_rewards, label='White')
    plt.plot(black_rewards, label='Black')
    plt.title(f"Rewards over Episodes (Episode {episode})")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()


env = ChessEnv()
state_size = 64 * 13
action_size = 64 * 64
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
        agent = white_agent if current_player == 0 else black_agent

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = state_to_tensor(next_state)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        if current_player == 0:
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