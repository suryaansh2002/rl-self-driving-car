import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from config import VISION_W, VISION_F, VISION_B, ROUND, DL_IS_TRAINING

checkpoint_dir = 'models'

GAMMA = 0.99  # Updated to match the value used in deep_traffic_agent.py

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
        self.fc1 = nn.Linear(32 * (VISION_F + VISION_B + 1) * (VISION_W * 2 + 1) + 4, 100)
        self.fc2 = nn.Linear(100, num_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        self.count_episodes = 0
        self.count_states = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def get_q_values(self, states, actions):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        with torch.no_grad():
            return self(states, actions).cpu().numpy()

    def save_checkpoint(self, current_iteration):
        if not self.main or not DL_IS_TRAINING:
            return False
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name, f"checkpoint-{current_iteration}.pth")
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.count_episodes,
            'iteration': current_iteration,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self):
        try:
            checkpoint_path = os.path.join(checkpoint_dir, self.model_name, "checkpoint.pth")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.count_episodes = checkpoint['episode']
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

    def log_training_loss(self, loss):
        # Implement logging as needed (e.g., using tensorboard or a custom solution)
        print(f"Training loss: {loss}")

    def log_q_values(self, q_values):
        # Implement logging as needed
        print(f"Q-values sum: {np.sum(q_values)}")

    def log_histogram(self, tag, values, step, bins=1000):
        # Implement histogram logging as needed
        pass

    # Add these methods to match the usage in deep_traffic_agent.py
    def log_average_speed(self, speed):
        print(f"Average speed: {speed}")

    def log_testing_speed(self, speed):
        print(f"Testing speed: {speed}")

    def log_total_frame(self, frame):
        print(f"Total frames: {frame}")

    def log_terminated(self, terminated):
        print(f"Terminated: {terminated}")

    def log_reward(self, reward):
        print(f"Reward: {reward}")

    def log_hard_brake_count(self, count):
        print(f"Hard brake count: {count}")

    def log_average_test_speed_40(self, speed):
        print(f"Average test speed (40 cars): {speed}")

    def log_average_test_speed_20(self, speed):
        print(f"Average test speed (20 cars): {speed}")

    def log_average_test_speed_60(self, speed):
        print(f"Average test speed (60 cars): {speed}")

    def log_action_frequency(self, action_stats):
        print(f"Action frequency: {action_stats}")