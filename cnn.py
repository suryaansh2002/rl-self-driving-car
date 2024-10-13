import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from config import VISION_W, VISION_F, VISION_B, ROUND, DL_IS_TRAINING

checkpoint_dir = 'models'

GAMMA = 0.95

class Cnn(nn.Module):
    def __init__(self, model_name, replay_memory, num_actions=5, target=False):
        super(Cnn, self).__init__()
        self.main = not target
        self.model_name = model_name
        self.replay_memory = replay_memory
        self.num_actions = num_actions
        # Update the state shape to match your actual input
        self.state_shape = (1, 36, 3)  # Changed to (C, H, W) format for PyTorch
        self.action_shape = (4,)
        
        # Define the layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 36 * 3 + 4, 100)  # Adjust input size based on your state shape
        self.fc2 = nn.Linear(100, num_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        self.count_episodes = 0
        self.count_states = 0
        
        # Device configuration
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

    def optimize(self, memory, batch_size=128, learning_rate=1e-3, target_network=None):
        if len(memory) < batch_size:
            return

        states, targets, actions = self.get_memory_component(memory, batch_size, target_network)
        
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        self.optimizer.zero_grad()
        q_values = self(states, actions)
        loss = self.loss_fn(q_values, targets)
        loss.backward()
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
        
        states = np.array(states).reshape(-1, 1, VISION_B + VISION_F + 1, VISION_W * 2 + 1)  # Changed shape for PyTorch
        targets = np.array(targets).reshape(-1, 5)
        actions = np.array(actions)
        return states, targets, actions

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

    def get_count_states(self):
        return self.count_states

    def increase_count_states(self):
        self.count_states += 1

    # Logging methods
    def log_training_loss(self, loss):
        # Implement logging as needed (e.g., using tensorboard or a custom solution)
        print(f"Training loss: {loss}")

    # Other logging methods (log_average_speed, log_testing_speed, etc.) should be implemented
    # using your preferred logging solution (e.g., tensorboard, custom file logging, etc.)

    def log_q_values(self, q_values):
        # Implement logging as needed
        print(f"Q-values sum: {np.sum(q_values)}")

    def log_histogram(self, tag, values, step, bins=1000):
        # Implement histogram logging as needed
        pass

# Note: Some methods like get_weights_variable, get_variable_value, and get_tensor_value
# are not directly applicable in PyTorch and have been removed. PyTorch allows direct
# access to model parameters and tensors without needing these methods.