#agent.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from dqn import DQN

class DQNAgent:
    def __init__(self, name):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
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
        if np.random.rand() <= self.epsilon:
            return random.choice(list(board.legal_moves))
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            q_values = self.model(state).squeeze().cpu().numpy()
        return self.choose_legal_move(board, q_values)

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

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

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