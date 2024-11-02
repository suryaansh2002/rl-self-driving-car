import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from config import VISION_W, VISION_F, VISION_B, ROUND, DL_IS_TRAINING
import logging
import random
from logging_config import setup_logger

checkpoint_dir = 'models'

GAMMA = 0.99

class Cnn(nn.Module):
    def __init__(self, model_name, replay_memory, num_actions=5, target=False):
        super(Cnn, self).__init__()
        self.main = not target  # used to determine if model is main network or target network(used in Q learning to stabilize training)
        self.model_name = model_name
        self.replay_memory = replay_memory  # holds buffer for experiences, useful in rl
        self.num_actions = num_actions  # agent can take 5 actions
        # Defines input dimensions for a state. Here VISION_F, VISION_B, VISION_W are configurations dtermining the perception field
        self.state_shape = (1, VISION_F + VISION_B + 1, VISION_W * 2 + 1)   # (1, 21+7+1, 7*2+1) = (1, 29, 15)        
        # Specifies input dimension for action, set as vector of length 4, possible representing features like speed or acclrn
        self.action_shape = (4,)
        
        # Define the layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)     # takes input with single ch and outputs 16 ch 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)    # takes input with single ch and outputs 16 ch
        # After the convolutional layers, the network flattens the data and combines it with the action input. The dimensions are calculated 
        # to match the output from the convolutional layers plus the action shape.
        self.fc_action = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32 * (VISION_F + VISION_B + 1) * (VISION_W * 2 + 1) + 4, 100)
        # The choice of 100 is somewhat empirical; in practice, it’s selected based on the problem's complexity, model capacity, and computational constraints
        # the output of fc2 provides the Q-values for all available actions, and the agent selects the action with the maximum Q-value as the best one. 
        # This is why fc2 has an output dimension of num_actions.
        self.fc2 = nn.Linear(100, num_actions)      # 100 is chosen to balance between model expressiveness and computational efficiency.
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3) # lr = 0.001
        
        self.count_episodes = 0
        self.count_states = 0
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)

        self.logger = setup_logger(f"CNN_{self.model_name}", 
                       ['logs/cnn_training.log'])
        self.load_checkpoint()


    def forward(self, state, action):
        #  This is the current observation (e.g., an image or sensor data) with shape (batch_size, 1, height, width)
        # This is a tensor representing additional action information (e.g., velocity, steering angle), with a shape
        #  corresponding to self.action_shape, likely (batch_size, 4)
        x = torch.relu(self.conv1(state))       # applies the first set of convolutional filters to state, producing a feature map with 16 channels..reLU introduces non linearity
        x = torch.relu(self.conv2(x))       # x has dimensions (batch_size, 32, height, width).
        x = x.view(x.size(0), -1)  # Flatten #This line flattens x from a 4D tensor to a 2D tensor, with shape (batch_size, flattened_size);flattened_size = channels * height * width. This reshaped tensor can now be fed into the fully connected layers that follow, which expect a 2D input.
        action_x = self.fc_action(action)
        x = torch.cat([x, action_x], dim=1)   # concatenates the flattened convolutional features with the action input along the last dimension (dim=1).
        # for above - This combined vector now has both state features (from convolutional layers) and action features, which the model can use to make a more informed prediction in fc1
        x = torch.relu(self.fc1(x)) # fc1 is a fully connected layer that takes in the concatenated vector and applies a linear transformation, followed by a ReLU activation.
        # self.logger.debug(f"Forward pass - Input state shape: {state.shape}, Action shape: {action.shape}, Output shape: {x.shape}")
        return self.fc2(x)      #  is the output layer that produces num_actions Q-values for the given state-action pair.

    def get_q_values(self, states, actions):
        """
        Obtain the Q-values for a given batch of states and actions without updating the model’s parameters
        """
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.FloatTensor(actions).to(self.device)
        with torch.no_grad():
            # This function takes states and actions, converts them to tensors on the correct device, performs a forward pass to get the Q-values, and then returns these Q-values as a NumPy array.
            return self(states, actions).cpu().numpy()  # self(states, actions) calls the model’s forward method to calculate the Q-values for the provided states and actions.

    def save_checkpoint(self, current_iteration):
        """
        designed to store a snapshot of the model’s state, the optimizer’s state, and the current training episode and iteration. This snapshot can be loaded later to resume training or evaluate the 
        model without starting over. The function only saves the checkpoint if the model is in training mode and is the main network.
        """
        # Ensures that checkpoints are only saved if the model is the main network (as opposed to a target network used for stabilizing training).
        if not self.main or not DL_IS_TRAINING:
            return False
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
        torch.save({
            'model_state_dict': self.state_dict(),  # Stores the model’s parameters. self.state_dict() returns a dictionary of all model parameters, which can be used to restore the model later
            'optimizer_state_dict': self.optimizer.state_dict(),    # Stores the state of the optimizer, including parameter values, gradients, and momentum, allowing the optimizer to resume from where it left off.
            'episode': self.count_episodes,
            'iteration': current_iteration,
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint at iteration {current_iteration}")
        # print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self):
        """
        The load_checkpoint function attempts to load a saved state for the model, optimizer, and episode counter from a checkpoint file. 
        If the checkpoint file is found, the model and optimizer are restored to their saved states, allowing training or evaluation to resume 
        from that point. If no checkpoint is found, it defaults to initializing the model from scratch.
        """
        try:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
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

# Deep Q-Learning (DQN), the use of a main network and a target network is a technique introduced to stabilize training.
# The main network (or policy network) is the model actively learning from experiences and making action decisions.
# The target network is a copy of the main network that is periodically updated with the main network's weights but remains frozen between updates.

# Why Use a Target Network?
# In Q-learning, the goal is to approximate the Q-values (expected future rewards) for each possible action in a given state.
#  However, directly training on Q-values introduces instability because the Q-values for each state-action pair are constantly updated, 
# which can lead to feedback loops and diverging Q-values.
# The target network helps stabilize this by providing a more consistent set of target Q-values for the main network to learn against. Instead of constantly updating the target values every step, we only update the target network occasionally. This slows down the rate at which the target values change, making learning smoother and more stable.


# The output from conv2 is a 3D tensor with dimensions [batch_size, 32, height, width]. To prepare this output for the fully connected (linear) layer, we need to flatten it into a 1D vector, which is where 32 * (VISION_F + VISION_B + 1) * (VISION_W * 2 + 1) comes from:
# 32: number of output channels
# VISION_F + VISION_B + 1: height dimension after convolution (specific to your vision configuration)
# VISION_W * 2 + 1: width dimension after convolution


# The output from conv1 has dimensions (16, VISION_F + VISION_B + 1, VISION_W * 2 + 1).
# Here, since we’re using padding=1 with a 3x3 kernel, the spatial dimensions remain the same. Therefore, conv1 will output:
# Channels: 16 (as specified).
# Height: Same as the input height, (VISION_F + VISION_B + 1).
# Width: Same as the input width, (VISION_W * 2 + 1).
# In summary, you’re right that conv1 takes in 1 channel and outputs 16 channels, but due to padding=1, it keeps the height and width the same as the input shape

# The output from conv2 also has dimensions (16, VISION_F + VISION_B + 1, VISION_W * 2 + 1).
# 