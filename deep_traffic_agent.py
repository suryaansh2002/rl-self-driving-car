import torch
import numpy as np
from random import choice, uniform
from collections import deque
from logging_config import setup_logger

from cnn import Cnn
from config import LEARNING_RATE, EPSILON_GREEDY_START_PROB, EPSILON_GREEDY_END_PROB, EPSILON_GREEDY_MAX_STATES, \
    MAX_MEM, BATCH_SIZE, VISION_W, VISION_B, VISION_F, TARGET_NETWORK_UPDATE_FREQUENCY, LEARN_START

class DeepTrafficAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.action_names = ['A', 'D', 'M', 'L', 'R']
        self.num_actions = len(self.action_names)
        # sets the maximum length of this queue; once the queue reaches this size, the oldest experience is discarded as new experiences are added.
        self.memory = deque(maxlen=MAX_MEM)     # deque (double-ended queue) that holds experiences for experience replay.
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Cnn(self.model_name, self.memory).to(self.device)
        self.target_model = Cnn(self.model_name, self.memory, target=True).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())      # This syncs the target model’s weights with the main model, ensuring they start off identically.
        self.target_model.eval()        # Sets the target model to evaluation mode, disabling certain operations like dropout (if any). This is typical because the target model is used only for inference (Q-value calculations), not training.
        self.count_states = self.model.get_count_states()   # Tracks the number of states encountered by the agent,
        self.count_episodes = self.model.get_count_episodes()
        # this is only the green observation area around the subject car
        self.previous_states = torch.zeros(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1).to(self.device)     # A tensor initialized to zeros, representing the previous state observed by the agent. It is shaped to match the model’s expected input dimensions (single-channel vision field based on the car’s view).
        self.previous_actions = torch.zeros(1, 4).to(self.device)
        self.previous_actions.fill_(2)  # Fills the previous_actions tensor with 2(to maintain)
        self.q_values = torch.zeros(5).to(self.device)  # Holds the last computed Q-values for each action, initialized to zeros.
        self.action = 2     # maintain speed

        self.delay_count = 0
        # This object gradually reduces epsilon from EPSILON_GREEDY_START_PROB to EPSILON_GREEDY_END_PROB over EPSILON_GREEDY_MAX_STATES, helping the agent shift from exploration to exploitation as it learns.
        self.epsilon_linear = LinearControlSignal(start_value=EPSILON_GREEDY_START_PROB,
                                                  end_value=EPSILON_GREEDY_END_PROB,
                                                  repeat=False)

        self.advantage = 0
        self.value = 0

        self.score = 0
        self.logger = setup_logger("DeepTrafficAgent", ["logs/agent_training.log"])
        

    def get_action_name(self, action):
        return self.action_names[action]

    def get_action_index(self, action):
        return self.action_names.index(action)
    
    def act(self, state, is_training=True):
        """
        The act method is responsible for selecting an action based on the given state. It implements an 
        epsilon-greedy policy, where the agent decides between exploration (random action) and exploitation 
        (using the model’s Q-values to select the best action).
        """
        # Reshape the state to match the expected input shape
        self.previous_states = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.previous_states = self.previous_states.view(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1)
        self.previous_actions = torch.zeros(1, 4).to(self.device)

        # Exploration
        if is_training and np.random.rand() <= self.epsilon_linear.get_value(self.count_states):
            action = np.random.randint(0, 5)
            q_values = torch.zeros(5).to(self.device)
        # Exploitation
        else:
            with torch.no_grad():
                q_values = self.model(self.previous_states, self.previous_actions)
            action = q_values.argmax().item()

        self.q_values = q_values.squeeze().cpu().numpy()
        self.action = action
        return self.q_values, self.get_action_name(action)


    def increase_count_states(self):
        self.model.increase_count_states()
        self.count_states = self.model.get_count_states()

    def increase_count_episodes(self):
        self.model.increase_count_episodes()
        self.count_episodes = self.model.get_count_episodes()

    def remember(self, reward, next_state, end_episode=False, is_training=True):
        """
        Storing experiences in memory for later replay.
        Triggering training (by calling optimize) if certain conditions are met.
        """
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)

        next_actions = self.previous_actions.clone()    # clones self.previous_actions to avoid modifying the original tensor
        next_actions = torch.roll(next_actions, -1, dims=1) # shifts actions one position to the left, making room for the current action.
        next_actions[0, -1] = self.action   # The latest action (stored in self.action) is added to the end of the next_actions tensor, preserving an action history.

        self.memory.append((self.previous_states,
                            next_state,
                            self.action,
                            reward - self.score,
                            end_episode,
                            self.previous_actions,
                            next_actions))
        
        # self.logger.info(f"Memory size: {len(self.memory)}")

        # # Check if memory has enough experiences for training
        # if len(self.memory) > BATCH_SIZE:
        #     self.logger.info("Memory has sufficient experiences for a training batch.")

        self.count_states = self.model.get_count_states()   # Synchronizes the agent’s state count with the model’s internal state counter.
        # print("states : ", self.count_states)
        # Starts training after a minimum number of states have been encountered (warm-up period). LEARN_START = 100000
        if is_training and self.count_states > LEARN_START and len(self.memory) > BATCH_SIZE:
            self.optimize()

        self.score = reward #  with the latest reward.

        if end_episode:
            self.previous_states = torch.zeros(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1).to(self.device)
            self.previous_actions = torch.zeros(1, 4).to(self.device)   # reset all values
            self.previous_actions.fill_(2)
            self.q_values = torch.zeros(5).to(self.device)
            self.action = 2
            self.score = 0
        # Do we need to log rewards, state_count and episode_counts here? 
        self.count_states = self.model.increase_count_states()

    def optimize(self):
        """
        The optimize method is responsible for training the main model by performing a single 
        step of gradient descent using a batch of experiences from memory. This is part of the 
        experience replay mechanism, where the model learns from past experiences, rather than 
        just recent ones, to improve training stability. 
        """
        batch = random.sample(self.memory, BATCH_SIZE)  #  Samples a random batch of experiences from memory
        states, next_states, actions, rewards, dones, prev_actions, next_actions = zip(*batch)  # Unpacks each experience tuple

        states = torch.cat(states).to(self.device)      #  convert each component of the batch into tensors and move them to the specified device 
        next_states = torch.cat(next_states).to(self.device)    # torch.cat(states): Concatenates the states and next states along the batch dimension for processing in a single forward pass.
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        prev_actions = torch.cat(prev_actions).to(self.device)
        next_actions = torch.cat(next_actions).to(self.device)

        current_q_values = self.model(states, prev_actions).gather(1, actions.unsqueeze(1)) # check comments below
        next_q_values = self.target_model(next_states, next_actions).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * 0.99 * next_q_values

        loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        if self.count_states % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.model.save_checkpoint(self.count_states)
            print("Target network updated")

        self.model.log_training_loss(loss.item())
        episode_count = self.model.get_count_episodes()
        state_count = self.model.get_count_states()
        # self.logger.info(
        #     f"Episode: {episode_count} , "
        #     f"Mean Q-value: {current_q_values.mean().item():.4f} ,"
        #     f"Loss: {loss.item():.4f}, "
        #     )
        self.logger.info(
            f"In Remember: - Episode: {episode_count}, State: {state_count}, Mean Q-value: {current_q_values.mean().item():.4f}, Loss: {loss.item():.4f}"
        )

class LinearControlSignal:
    def __init__(self, start_value, end_value, repeat=False):
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = EPSILON_GREEDY_MAX_STATES
        self.repeat = repeat
        self._coefficient = (end_value - start_value) / self.num_iterations

    def get_value(self, iteration):
        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value
    
# line 19 and 20 : The main and target models are identical only at initialization and at specific intervals when the target model is updated. 
# Between these updates, the main model continues to learn and change, while the target model remains static, providing stability 
# to the learning process.

# lines 55-57 
# torch.FloatTensor(state): Converts state (likely a NumPy array) to a PyTorch tensor of type float.
# .unsqueeze(0): Adds a batch dimension (since the model expects batches, even if it’s a single sample).
# .view(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1): Reshapes self.previous_states to fit the model’s expected input dimensions, which include batch size, channels, and spatial dimensions (height and width).
# self.previous_actions: Initializes the action history tensor to zeros with a shape of (1, 4), representing a batch of action placeholders.


# 60-67
# The agent uses an epsilon-greedy approach to balance exploration (trying new actions) and exploitation (selecting the best-known action).
# is_training: If True, the agent is in training mode and can explore. If False (e.g., during testing), the agent always exploits (chooses the best-known action).
# np.random.rand() <= self.epsilon_linear.get_value(self.count_states): Compares a random value with the current epsilon value, allowing exploration with a probability 
# proportional to epsilon. Over time, epsilon decreases, reducing the frequency of exploration.
# action = np.random.randint(0, 5): If exploration is chosen, the agent selects a random action (0 to 4) instead of consulting the model.
# q_values = torch.zeros(5).to(self.device): Sets Q-values to zeros for consistency, as this case uses a random action, not model predictions.


# line 104 
# Checks three conditions to decide if training (optimize) should be performed:
# is_training: Ensures training occurs only in training mode.
# self.count_states > LEARN_START: Starts training after a minimum number of states have been encountered (warm-up period).
# len(self.memory) > BATCH_SIZE: Ensures there are enough samples in memory for a full training batch.


# line 137
# self.model(states, prev_actions): Passes states and prev_actions through the main model to get the Q-values for each action.
# .gather(1, actions.unsqueeze(1)): Extracts the Q-values corresponding to the specific actions taken (from the actions tensor). actions.unsqueeze(1) adds a dimension to make the tensor shape compatible for gathering.
# current_q_values contains the Q-values predicted by the main model for the actions taken in each experience in the mini-batch.



