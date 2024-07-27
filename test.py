from chess_env import ChessEnv
from agent import DQNAgent
import numpy as np
import time


def state_to_tensor(state):
    return np.reshape(state, (1, -1))[0]


env = ChessEnv()
state_size = 64 * 13
action_size = 64 * 64
white_agent = DQNAgent(state_size, action_size, "White")
black_agent = DQNAgent(state_size, action_size, "Black")

# Load the trained models
white_agent.load("white_chess_dqn_model_final.pth")
black_agent.load("black_chess_dqn_model_final.pth")

# Test the trained agents
state = env.reset()
env.render()
done = False
step = 0
max_steps = 100

while not done and step < max_steps:
    current_player = env.get_current_player()
    agent = white_agent if current_player == 0 else black_agent

    state_tensor = state_to_tensor(state)
    action = agent.act(state_tensor)
    state, reward, done, _ = env.step(action)
    env.render()
    time.sleep(1)
    step += 1

print("Game Over")