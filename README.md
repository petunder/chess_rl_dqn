# Chess Reinforcement Learning with Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) to learn and play chess. It uses reinforcement learning techniques to train an AI agent to make chess moves based on the current board state.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project combines the game of chess with deep reinforcement learning. It uses a custom chess environment built with Python-chess and OpenAI Gym, and implements a Deep Q-Network (DQN) using PyTorch to learn chess strategies.

## Features

- Custom Chess Environment compatible with OpenAI Gym
- Deep Q-Network (DQN) implementation with PyTorch
- Experience Replay for improved learning stability
- Epsilon-greedy exploration strategy
- Visualization of training progress
- Test mode to play against the trained agent

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gym
- Python-chess

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/chess-reinforcement-learning.git
   cd chess-reinforcement-learning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the agent:

```python
python train.py
```

To test the trained agent:

```python
python test.py
```

## Project Structure

- `chess_env.py`: Contains the `ChessEnv` class, which implements the chess environment.
- `dqn.py`: Implements the Deep Q-Network architecture.
- `agent.py`: Contains the `DQNAgent` class, which implements the reinforcement learning agent.
- `train.py`: Script to train the agent.
- `test.py`: Script to test the trained agent.
- `utils.py`: Utility functions for data processing and visualization.

## How It Works

1. The chess environment (`ChessEnv`) provides an interface for the agent to interact with the chess game.
2. The DQN agent observes the current state of the chess board and chooses an action (move).
3. The environment applies the action and returns the new state, reward, and whether the game is done.
4. The agent stores this experience in its replay buffer and uses it to learn and improve its policy.
5. This process is repeated for many episodes, gradually improving the agent's chess-playing ability.

## Customization

You can customize various aspects of the project:

- Modify the neural network architecture in `dqn.py`
- Adjust hyperparameters like learning rate, discount factor, etc. in `agent.py`
- Change the reward structure in `chess_env.py`

## Limitations

- The current implementation may not achieve high-level chess play due to the complexity of the game.
- Training can be computationally intensive and time-consuming.
- The agent may struggle with long-term strategic planning.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.