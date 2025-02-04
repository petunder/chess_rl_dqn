#agent.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from dqn import ChessNetwork

class DQNAgent:
    def __init__(self, name):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ChessNetwork().to(self.device)
        self.target_model = ChessNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.memory = deque(maxlen=10000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.losses = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, board):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            with torch.no_grad():
                policy, _ = self.model(state)
                policy = policy.squeeze().cpu()

            move, penalty = choose_legal_move(board, policy)

            if move is None:
                print(f"{self.name} agent has no legal moves. Game over.")
                return None

            if move in board.legal_moves:
                return move
            else:
                attempts += 1
                print(f"{self.name} agent made an illegal move (attempt {attempts}): {move}")
                print(f"{self.name} agent is being penalized and will try again.")
                move_index = move.from_square * 64 + move.to_square
                if move_index < policy.shape[0]:
                    policy[move_index] = float('-inf')
                else:
                    print(f"Warning: Move index {move_index} out of bounds for policy shape {policy.shape}")

        print(f"{self.name} agent failed to choose a legal move after {max_attempts} attempts. Choosing randomly.")
        return random.choice(list(board.legal_moves))
    def choose_legal_move(self, board, q_values):
        legal_moves = list(board.legal_moves)
        legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
        legal_q_values = q_values[legal_move_indices]
        best_move_index = np.argmax(legal_q_values)
        return legal_moves[best_move_index]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_policy, current_value = self.model(states)
        current_q_values = current_policy.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_policy, next_value = self.target_model(next_states)
            next_q_values = next_policy.max(1)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        value_loss = nn.MSELoss()(current_value.squeeze(), target_q_values)
        policy_loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.losses.append(loss.item())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_average_weights(self):
        return {name: param.mean().item() for name, param in self.model.named_parameters()}

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())