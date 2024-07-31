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
from sklearn.model_selection import train_test_split

def load_and_split_dataset(dataset_name, split_ratio=0.8):
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.take(30000)  # Ограничиваем размер датасета

    data_list = list(dataset)
    train_data, val_data = train_test_split(data_list, train_size=split_ratio, random_state=42)

    logger.info(f"Dataset split into {len(train_data)} training samples and {len(val_data)} validation samples.")
    return train_data, val_data

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        logger.info("ChessEnv initialized.")

    def reset(self):
        self.board.reset()
#        logger.info("Board reset to initial position.")

    def push_uci(self, move):
        self.board.push_uci(move)
#        logger.debug(f"Move {move} executed. Current board state:\n{self.board}")

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
#        logger.debug(f"Converted move {move} to action {action}")
        return action

    def board_state_to_tensor(self, state):
#        logger.debug(f"Converting board state to tensor: {state}")

        # Инициализация пустого тензора размером (13, 8, 8)
        tensor = torch.zeros((13, 8, 8), dtype=torch.float32)

        # Разбиение состояния на части (FEN разделен пробелами)
        parts = state.split()
        board_state = parts[0]  # Первое значение - состояние доски

        # Словарь соответствий фигур и каналов тензора
        piece_to_channel = {
            'p': 0, 'P': 1, 'r': 2, 'R': 3,
            'n': 4, 'N': 5, 'b': 6, 'B': 7,
            'q': 8, 'Q': 9, 'k': 10, 'K': 11,
        }

        # Преобразование состояния доски в тензор
        rows = board_state.split('/')
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)  # Пропуск пустых клеток
                else:
                    if char in piece_to_channel:
                        channel = piece_to_channel[char]
                        tensor[channel, row_idx, col_idx] = 1
#                        logger.debug(f"Placed {char} at tensor[{channel}, {row_idx}, {col_idx}]")
                    col_idx += 1

        # Дополнительный канал для текущего хода (белые или черные)
        tensor[12, :, :] = 0 if parts[1] == 'w' else 1
#        logger.debug(f"Set player turn in tensor[12, :, :] to {'0 (white)' if parts[1] == 'w' else '1 (black)'}")

#        logger.debug(f"Tensor after conversion: {tensor}")
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

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    policy_criterion = nn.CrossEntropyLoss()

    train_data, val_data = load_and_split_dataset(dataset_name)

    train_dataset = ChessDataset(train_data)
    val_dataset = ChessDataset(val_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_policy_loss = 0
        total_batches = 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started.")

        # Обучение на тренировочном наборе
        model.train()
        for i, (states, actions) in enumerate(train_dataloader):
#            logger.debug(f"Batch {i + 1} loaded with {len(states)} samples.")
            states, actions = states.to(device), actions.to(device)

            optimizer.zero_grad()
            policy, _ = model(states)

            policy_loss = policy_criterion(policy, actions.squeeze())
#            logger.debug(f"Policy loss: {policy_loss.item()}")

            policy_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_batches += 1
#            logger.debug(f"Batch {i + 1} processed. Policy Loss: {policy_loss.item()}")

        avg_policy_loss = total_policy_loss / total_batches if total_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average Training Policy Loss: {avg_policy_loss:.4f}")
        log_model_statistics(model)

        # Проверка на проверочном наборе
        model.eval()
        total_val_policy_loss = 0
        total_val_batches = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for i, (states, actions) in enumerate(val_dataloader):
                states, actions = states.to(device), actions.to(device)
                policy, _ = model(states)
                policy_loss = policy_criterion(policy, actions.squeeze())
                total_val_policy_loss += policy_loss.item()
                total_val_batches += 1

                # Подсчет правильных и неправильных предсказаний
                predicted_actions = policy.argmax(dim=1)
                correct_predictions += (predicted_actions == actions.squeeze()).sum().item()
                total_predictions += actions.size(0)

        avg_val_policy_loss = total_val_policy_loss / total_val_batches if total_val_batches > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average Validation Policy Loss: {avg_val_policy_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions} correct predictions)")

    model_save_path = f"pretrained_chess_model_train.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    dataset_name = "laion/strategic_game_chess"
    pretrain(dataset_name)
    # pretrain(dataset_name, 'test') # Uncomment for the test set
