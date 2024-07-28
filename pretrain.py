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
        states = []
        actions = []

        for move in moves:
            state = self.env._get_observation()
            chess_move = chess.Move.from_uci(move)
            action = chess_move.from_square * 64 + chess_move.to_square
            self.env.step(action)
            states.append(state)
            actions.append(action)

        # Преобразуем результат в числовое значение
        if result == "1-0":
            value = 1
        elif result == "0-1":
            value = -1
        else:  # "1/2-1/2"
            value = 0

        if not states:
            # Если по какой-то причине нет состояний, вернем пустые тензоры
            return torch.empty(0, 13, 8, 8), torch.empty(0, dtype=torch.long), torch.empty(0)

        return torch.stack([torch.FloatTensor(s) for s in states]), torch.LongTensor(actions), torch.FloatTensor([value] * len(states))


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    dataset = load_dataset("adamkarvonen/chess_sae_individual_games_filtered",
                           split="train[:10000]")  # Загружаем первые 10000 игр для примера
    chess_dataset = ChessDataset(dataset)
    dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for states, actions, values in dataloader:
            if states.numel() == 0:
                continue  # Пропускаем пустые батчи

            states, actions, values = states.to(device), actions.to(device), values.to(device)
            optimizer.zero_grad()
            policy, value = model(states.view(-1, 13, 8, 8))
            policy_loss = policy_criterion(policy, actions.view(-1))
            value_loss = value_criterion(value.squeeze(), values.view(-1))
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "pretrained_chess_model.pth")


if __name__ == "__main__":
    pretrain()