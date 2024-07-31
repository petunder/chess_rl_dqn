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
                    yield self.process_game(game)
                buffer = []

        if buffer:
            logger.info("Processing remaining games in buffer.")
            random.shuffle(buffer)
            for game in buffer:
                yield self.process_game(game)

    def process_game(self, game):
        moves = game['Moves']
        result = game['Result']
        logger.debug(f"Processing game with moves: {moves} and result: {result}.")

        self.env.reset()
        states = []
        actions = []

        for move in moves:
            self.env.push_uci(move)
            state = self.env.get_board_state().fen()
            action = self.move_to_action(move)
            states.append(state)
            actions.append(action)

        idx = random.randint(0, len(states) - 1)
        state = states[idx]
        action = actions[idx]
        value = 1 if result == "1-0" else -1 if result == "0-1" else 0

        state_tensor = self.board_state_to_tensor(state)
        logger.debug(f"Game processed. Selected move: {action}, value: {value}")
        return state_tensor, torch.LongTensor([action]), torch.FloatTensor([value])

    def move_to_action(self, move):
        """Convert a UCI move to a unique integer."""
        from_square = chess.SQUARE_NAMES.index(move[:2])
        to_square = chess.SQUARE_NAMES.index(move[2:4])
        action = from_square * 64 + to_square
        logger.debug(f"Converted move {move} to action {action}")
        return action

    def board_state_to_tensor(self, state):
        # Преобразование состояния доски (FEN) в тензор
        logger.debug(f"Converting board state to tensor: {state}")
        # Пример преобразования: нужно адаптировать в зависимости от модели
        tensor = torch.zeros((13, 8, 8))  # Примерный размер
        # Логика преобразования состояния FEN в тензор
        # ...
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
    value_criterion = nn.MSELoss()

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.take(100)  # Ограничиваем размер датасета 100 записями
    chess_dataset = ChessDataset(dataset)
    dataloader = DataLoader(chess_dataset, batch_size=32)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started.")
        for i, (states, actions, values) in enumerate(dataloader):
            logger.debug(f"Batch {i + 1} loaded with {len(states)} samples.")
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
            logger.debug(f"Batch {i + 1} processed. Loss: {loss.item()}")

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        log_model_statistics(model)

    model_save_path = f"pretrained_chess_model_train.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    dataset_name = "laion/strategic_game_chess"
    pretrain(dataset_name)
    # pretrain(dataset_name, 'test') # Uncomment for the test set
