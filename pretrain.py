import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dqn import ChessNetwork
from chess_env import ChessEnv


class ChessDataset(Dataset):
    def __init__(self, games):
        self.games = games
        self.env = ChessEnv()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        self.env.reset()
        for move in game['moves'].split():
            state = self.env._get_observation()
            action = self.env.board.parse_san(move)
            self.env.step(action.from_square * 64 + action.to_square)
        return torch.FloatTensor(state), torch.LongTensor([action.from_square * 64 + action.to_square])


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = load_dataset("LeelaChessZero/chess-games",
                           split="train[:1000]")  # Загружаем только первые 1000 игр для примера
    chess_dataset = ChessDataset(dataset)
    dataloader = DataLoader(chess_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            policy, _ = model(states)
            loss = criterion(policy, actions.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "pretrained_chess_model.pth")


if __name__ == "__main__":
    pretrain()