#pretrain.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dqn import ChessNetwork
from chess_common import ChessEnv
import pandas as pd
import os
import random

class ChessDataset(Dataset):
    def __init__(self, games, valid_games_indices):
        self.games = [games[i] for i in valid_games_indices]
        self.env = ChessEnv()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        moves = game['text'].split()[1::2]
        result = game['Result']

        self.env.reset()
        states = []
        actions = []

        for move in moves:
            chess_move = self.env.board.parse_san(move)
            action = chess_move.from_square * 64 + chess_move.to_square
            state = self.env._get_observation()
            self.env.step(action)
            states.append(state)
            actions.append(action)

        idx = random.randint(0, len(states) - 1)
        state = states[idx]
        action = actions[idx]
        value = 1 if result == "1-0" else -1 if result == "0-1" else 0

        return torch.FloatTensor(state), torch.LongTensor([action]), torch.FloatTensor([value])

def load_valid_games_indices(dataset_name, split_type):
    valid_games_file = f'{dataset_name}_{split_type}_valid_games.csv'
    if not os.path.exists(valid_games_file):
        raise FileNotFoundError(f"Valid games file for {split_type} not found. Please check the dataset validation step.")
    valid_games_df = pd.read_csv(valid_games_file)
    return valid_games_df['valid_game_index'].tolist()

def pretrain(dataset_name, split_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    dataset = load_dataset(dataset_name, split=f"{split_type}[:10000]")
    valid_games_indices = load_valid_games_indices(dataset_name, split_type)
    chess_dataset = ChessDataset(dataset, valid_games_indices)
    dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        for states, actions, values in dataloader:
            states, actions, values = states.to(device), actions.to(device), values.to(device)
            optimizer.zero_grad()
            policy, value = model(states)
            policy_loss = policy_criterion(policy, actions.squeeze())
            value_loss = value_criterion(value.squeeze(), values)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"pretrained_chess_model_{split_type}.pth")

if __name__ == "__main__":
    dataset_name = "adamkarvonen/chess_sae_individual_games_filtered"
    # Вы можете заменить 'train' на 'test' для обучения модели на тестовом наборе
    pretrain(dataset_name, 'train')
    # pretrain(dataset_name, 'test') # Раскомментируйте для тестового набора
