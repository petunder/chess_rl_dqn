import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dqn import ChessNetwork
from chess_common import ChessEnv
import chess
import chess.pgn
import io


import chess

import random
import chess

class ChessDataset(Dataset):
    def __init__(self, games):
        self.games = games
        self.env = ChessEnv()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        moves = game['text'].split()[1::2]  # Получаем только ходы, пропуская номера
        result = game['Result']

        self.env.reset()
        valid_states = []
        valid_actions = []

        for move in moves:
            try:
                chess_move = self.env.board.parse_san(move)
                action = chess_move.from_square * 64 + chess_move.to_square
                state = self.env._get_observation()
                self.env.step(action)
                valid_states.append(state)
                valid_actions.append(action)
            except (chess.IllegalMoveError, chess.InvalidMoveError):
                print(f"Invalid move: {move}")
                continue

        if not valid_states:
            # Если нет допустимых ходов, возвращаем начальное состояние и случайное действие
            return self.env._get_observation(), random.randint(0, 4095), 0

        # Выбираем случайное состояние и соответствующее действие
        idx = random.randint(0, len(valid_states) - 1)
        state = valid_states[idx]
        action = valid_actions[idx]

        # Преобразуем результат в числовое значение
        if result == "1-0":
            value = 1
        elif result == "0-1":
            value = -1
        else:  # "1/2-1/2"
            value = 0

        return torch.FloatTensor(state), action, value

def collate_fn(batch):
    states, actions, values = zip(*batch)
    return torch.stack(states), torch.LongTensor(actions), torch.FloatTensor(values)


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    dataset = load_dataset("adamkarvonen/chess_sae_individual_games_filtered",
                           split="train[:10000]")  # Загружаем первые 10000 игр для примера
    chess_dataset = ChessDataset(dataset)
    dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        for batch_idx, (states, actions, values) in enumerate(dataloader):
            states, actions, values = states.to(device), actions.to(device), values.to(device)
            optimizer.zero_grad()
            policy, value = model(states)
            policy_loss = policy_criterion(policy, actions)
            value_loss = value_criterion(value.squeeze(), values)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "pretrained_chess_model.pth")


if __name__ == "__main__":
    pretrain()