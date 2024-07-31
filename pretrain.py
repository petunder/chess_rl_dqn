# pretrain.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from dqn import ChessNetwork
import random
import chess
from loguru import logger


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        logger.info("ChessEnv initialized.")

    def reset(self):
        self.board.reset()
        logger.info("Board reset to initial position.")

    def push_uci(self, move):
        self.board.push_uci(move)
        logger.debug(f"Move {move} executed. Current board state:\n{self.board}")

    def get_board_state(self):
        return self.board


class ChessDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=100):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.env = ChessEnv()
        logger.info(f"ChessDataset initialized with buffer size {buffer_size}.")

    def __iter__(self):
        buffer = []
        for game in self.dataset:
            buffer.append(game)
            if len(buffer) >= self.buffer_size:
                logger.info("Buffer full. Shuffling and processing games.")
                random.shuffle(buffer)
                for game in buffer:
                    yield from self.process_game(game)
                buffer = []

        if buffer:
            logger.info("Processing remaining games in buffer.")
            random.shuffle(buffer)
            for game in buffer:
                yield from self.process_game(game)

    def process_game(self, game):
        moves = game['Moves']
        self.env.reset()

        for move in moves:
            state = self.env.get_board_state().fen()
            action = self.move_to_action(move)
            state_tensor = self.board_state_to_tensor(state)
            self.env.push_uci(move)
            yield state_tensor, torch.LongTensor([action])

    def move_to_action(self, move):
        from_square = chess.SQUARE_NAMES.index(move[:2])
        to_square = chess.SQUARE_NAMES.index(move[2:4])
        action = from_square * 64 + to_square
        logger.debug(f"Converted move {move} to action {action}")
        return action

    def board_state_to_tensor(self, state):
        logger.debug(f"Converting board state to tensor: {state}")
        tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
        parts = state.split()
        board_state = parts[0]

        piece_to_channel = {
            'p': 0,  'P': 1,  'r': 2,  'R': 3,
            'n': 4,  'N': 5,  'b': 6,  'B': 7,
            'q': 8,  'Q': 9,  'k': 10, 'K': 11,
        }

        rows = board_state.split('/')
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    if char in piece_to_channel:
                        channel = piece_to_channel[char]
                        tensor[channel, row_idx, col_idx] = 1
                    col_idx += 1

        tensor[12, :, :] = 0 if parts[1] == 'w' else 1
        logger.debug(f"Tensor after conversion: {tensor}")
        return tensor



def log_model_statistics(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: mean={param.data.mean().item()}, std={param.data.std().item()}")


def pretrain(dataset_name):
    logger.info(f"Starting pretraining with dataset: {dataset_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNetwork().to(device)
    logger.info(f"Model initialized on device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.CrossEntropyLoss()

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.take(100)  # Ограничиваем размер датасета 100 записями

    logger.info("First 5 entries of the dataset:")
    for i, entry in enumerate(dataset):
        logger.info(f"Entry {i + 1}: {entry}")
        if i >= 4:
            break

    chess_dataset = ChessDataset(dataset)
    dataloader = DataLoader(chess_dataset, batch_size=32)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_policy_loss = 0
        total_batches = 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started.")

        for i, (states, actions) in enumerate(dataloader):
            logger.debug(f"Batch {i + 1} loaded with {len(states)} samples.")
            states, actions = states.to(device), actions.to(device)

            optimizer.zero_grad()
            policy, _ = model(states)

            policy_loss = policy_criterion(policy, actions.squeeze())
            logger.debug(f"Policy loss: {policy_loss.item()}")

            policy_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_batches += 1
            logger.debug(f"Batch {i + 1} processed. Policy Loss: {policy_loss.item()}")

        avg_policy_loss = total_policy_loss / total_batches if total_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average Policy Loss: {avg_policy_loss:.4f}")
        log_model_statistics(model)

    model_save_path = f"pretrained_chess_model_train.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    dataset_name = "laion/strategic_game_chess"
    pretrain(dataset_name)
    # pretrain(dataset_name, 'test') # Uncomment for the test set

if __name__ == "__main__":
    dataset_name = "laion/strategic_game_chess"
    pretrain(dataset_name)
    # pretrain(dataset_name, 'test') # Uncomment for the test set
