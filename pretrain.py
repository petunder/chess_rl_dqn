# pretrain.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from dqn import ChessNetwork
import random
import chess

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()

    def push_uci(self, move):
        self.board.push_uci(move)

    def get_board_state(self):
        return self.board

class ChessDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.env = ChessEnv()

    def __iter__(self):
        for game in self.dataset:
            moves = game['Moves']
            result = game['Result']

            self.env.reset()
            states = []
            actions = []

            for move in moves:
                self.env.push_uci(move)
                state = self.env.get_board_state().fen()
                action = move
                states.append(state)
                actions.append(action)

            idx = random.randint(0, len(states) - 1)
            state = states[idx]
            action = actions[idx]
            value = 1 if result == "1-0" else -1 if result == "0-1" else 0

            state_tensor = self.board_state_to_tensor(state)
            yield state_tensor, torch.LongTensor([action]), torch.FloatTensor([value])

    def board_state_to_tensor(self, state):
        # Implement the conversion from board state (FEN) to tensor
        pass

def pretrain(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    chess_dataset = ChessDataset(dataset)
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

    torch.save(model.state_dict(), f"pretrained_chess_model_train.pth")

if __name__ == "__main__":
    dataset_name = "laion/strategic_game_chess"
    pretrain(dataset_name)
    # pretrain(dataset_name, 'test') # Uncomment for the test set
