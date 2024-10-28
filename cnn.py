import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import VISION_W, VISION_F, VISION_B, ROUND, DL_IS_TRAINING
import logging

checkpoint_dir = 'models'

GAMMA = 0.99

class Cnn(nn.Module):
    def __init__(self, model_name, replay_memory, num_actions=5, target=False):
        super(Cnn, self).__init__()
        self.main = not target
        self.model_name = model_name
        self.replay_memory = replay_memory
        self.num_actions = num_actions
        self.state_shape = (1, VISION_F + VISION_B + 1, VISION_W * 2 + 1)
        self.action_shape = (4,)
        
        # Define the layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc_action = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32 * (VISION_F + VISION_B + 1) * (VISION_W * 2 + 1) + 4, 100)
        self.fc2 = nn.Linear(100, num_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        self.count_episodes = 0
        self.count_states = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.writer = SummaryWriter(f"{checkpoint_dir}/{model_name}")
        self.logger = self._setup_logger()
        self.load_checkpoint()

    def _setup_logger(self):
        logger = logging.getLogger(f"Cnn_{self.model_name}")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"logs/{self.model_name}_cnn.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def forward(self, state, action):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        action_x = self.fc_action(action)
        x = torch.cat([x, action_x], dim=1)
        x = torch.relu(self.fc1(x))
        self.logger.debug(f"Forward pass - Input state shape: {state.shape}, Action shape: {action.shape}, Output shape: {x.shape}")
        return self.fc2(x)

    def get_q_values(self, states, actions):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        with torch.no_grad():
            return self(states, actions).cpu().numpy()

    def save_checkpoint(self, current_iteration):
        if not self.main or not DL_IS_TRAINING:
            return False
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name, f"checkpoint.pth")
        print("Checkpoint path: ", checkpoint_path)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.count_episodes,
            'iteration': current_iteration,
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint at iteration {current_iteration}")
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self):
        try:
            checkpoint_path = os.path.join(checkpoint_dir, self.model_name, "checkpoint.pth")
            print("Checkpoint path: ", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.count_episodes = checkpoint['episode']
            self.logger.info(f"Loaded checkpoint, current episode: {self.count_episodes}")
            print(f"Restored checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print("No checkpoint found. Initializing model.")

    def get_count_episodes(self):
        return self.count_episodes

    def increase_count_episodes(self):
        self.count_episodes += 1
        return self.count_episodes

    def get_count_states(self):
        return self.count_states

    def increase_count_states(self):
        self.count_states += 1
        return self.count_states

    def optimize(self, memory, batch_size=128, learning_rate=1e-3, target_network=None):
        states, targets, actions = self.get_memory_component(memory, batch_size, target_network)
        
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)

        self.optimizer.zero_grad()
        q_values = self(states, actions)
        loss = self.loss_fn(q_values, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.log_training_loss(loss.item())

    def get_memory_component(self, memory, batch_size, target_network=None):
        minibatch = random.sample(memory, batch_size)
        states = []
        actions = []
        targets = []
        for state, next_state, action, reward, end_episode, _actions, next_actions in minibatch:
            states.append(state)
            actions.append(_actions)
            target = reward
            if not end_episode:
                q_values = target_network.get_q_values(next_state, next_actions) if target_network else self.get_q_values(next_state, next_actions)
                target = reward + GAMMA * np.max(q_values)

            current = self.get_q_values(state, _actions)
            current[0][action] = target
            targets.append(current[0])
        
        states = np.array(states).reshape(-1, VISION_B + VISION_F + 1, VISION_W * 2 + 1, 1)
        targets = np.array(targets).reshape(-1, 5)
        actions = np.array(actions)
        return states, targets, actions

    def log_training_loss(self, loss):
        self.writer.add_scalar('Loss/train', loss, self.count_episodes)

    def log_q_values(self, q_values):
        self.writer.add_scalar('Q_values/sum', np.sum(q_values), self.count_states)

    def log_average_speed(self, speed):
        self.writer.add_scalar('Speed/average', speed, self.count_episodes)

    def log_testing_speed(self, speed):
        self.writer.add_scalar('Speed/test', speed, self.count_episodes)

    def log_total_frame(self, frame):
        self.writer.add_scalar('Frames/total', frame, self.count_episodes)

    def log_terminated(self, terminated):
        self.writer.add_scalar('Episode/terminated', int(terminated), self.count_episodes)

    def log_reward(self, reward):
        self.writer.add_scalar('Reward/episode', reward, self.count_episodes)

    def log_hard_brake_count(self, count):
        self.writer.add_scalar('Actions/hard_brake_count', count, self.count_states)

    def log_average_test_speed_40(self, speed):
        self.writer.add_scalar('Speed/test_average_40', speed, self.count_episodes)

    def log_average_test_speed_20(self, speed):
        self.writer.add_scalar('Speed/test_average_20', speed, self.count_episodes)

    def log_average_test_speed_60(self, speed):
        self.writer.add_scalar('Speed/test_average_60', speed, self.count_episodes)

    def log_action_frequency(self, action_stats):
        for i, freq in enumerate(action_stats):
            self.writer.add_scalar(f'Actions/frequency_{i}', freq, self.count_episodes)

    def log_histogram(self, tag, values, step, bins=1000):
        self.writer.add_histogram(tag, values, step, bins=bins)

    def close(self):
        self.writer.close()