#dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True):
        super().__init__()
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=1 if kernel_size == 3 else 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.relu = relu

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return F.relu(x, inplace=True) if self.relu else x


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3)
        self.conv2 = ConvBlock(channels, channels, 3, relu=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return F.relu(out, inplace=True)


class ChessNetwork(nn.Module):
    def __init__(self, in_channels: int = 13, board_size: int = 8, residual_channels: int = 256,
                 residual_layers: int = 19):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(*[ResBlock(residual_channels) for _ in range(residual_layers)])

        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size * board_size)

        self.value_conv = ConvBlock(residual_channels, 1, 1)
        self.value_fc_1 = nn.Linear(board_size * board_size, 256)
        self.value_fc_2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.residual_tower(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_fc(torch.flatten(policy, start_dim=1))

        # Value head
        value = self.value_conv(x)
        value = F.relu(self.value_fc_1(torch.flatten(value, start_dim=1)), inplace=True)
        value = torch.tanh(self.value_fc_2(value))

        return policy, value


class DQNAgent:
    def __init__(self, state_size, action_size, name):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ChessNetwork().to(self.device)
        self.target_model = ChessNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, nesterov=True,
                                         weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, min_lr=5e-6)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.losses = []

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.model(state)
        return policy.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

        policy, value = self.model(states)
        next_policy, next_value = self.target_model(next_states)

        q_values = policy.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_policy.max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        value_loss = F.mse_loss(value.squeeze(), expected_q_values)
        policy_loss = F.mse_loss(q_values, expected_q_values.detach())
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

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