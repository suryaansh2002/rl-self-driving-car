import torch
import numpy as np
from random import choice, uniform
from collections import deque

from cnn import Cnn
from config import LEARNING_RATE, EPSILON_GREEDY_START_PROB, EPSILON_GREEDY_END_PROB, EPSILON_GREEDY_MAX_STATES, \
    MAX_MEM, BATCH_SIZE, VISION_W, VISION_B, VISION_F, TARGET_NETWORK_UPDATE_FREQUENCY, LEARN_START

class DeepTrafficAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.action_names = ['A', 'D', 'M', 'L', 'R']
        self.num_actions = len(self.action_names)
        self.memory = deque(maxlen=MAX_MEM)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Cnn(self.model_name, self.memory).to(self.device)
        self.target_model = Cnn(self.model_name, self.memory, target=True).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.count_states = self.model.get_count_states()
        self.count_episodes = self.model.get_count_episodes()
        self.previous_states = torch.zeros(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1).to(self.device)
        self.previous_actions = torch.zeros(1, 4).to(self.device)
        self.previous_actions.fill_(2)
        self.q_values = torch.zeros(5).to(self.device)
        self.action = 2

        self.delay_count = 0

        self.epsilon_linear = LinearControlSignal(start_value=EPSILON_GREEDY_START_PROB,
                                                  end_value=EPSILON_GREEDY_END_PROB,
                                                  repeat=False)

        self.advantage = 0
        self.value = 0

        self.score = 0

    def get_action_name(self, action):
        return self.action_names[action]

    def get_action_index(self, action):
        return self.action_names.index(action)
    
    def act(self, state, is_training=True):
        # Reshape the state to match the expected input shape
        self.previous_states = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.previous_states = self.previous_states.view(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1)
        self.previous_actions = torch.zeros(1, 4).to(self.device)

        if is_training and np.random.rand() <= self.epsilon_linear.get_value(self.count_states):
            action = np.random.randint(0, 5)
            q_values = torch.zeros(5).to(self.device)
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
        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)

        next_actions = self.previous_actions.clone()
        next_actions = torch.roll(next_actions, -1, dims=1)
        next_actions[0, -1] = self.action

        self.memory.append((self.previous_states,
                            next_state,
                            self.action,
                            reward - self.score,
                            end_episode,
                            self.previous_actions,
                            next_actions))

        self.count_states = self.model.get_count_states()

        if is_training and self.count_states > LEARN_START and len(self.memory) > BATCH_SIZE:
            self.optimize()

        self.score = reward

        if end_episode:
            self.previous_states = torch.zeros(1, 1, VISION_F + VISION_B + 1, VISION_W * 2 + 1).to(self.device)
            self.previous_actions = torch.zeros(1, 4).to(self.device)
            self.previous_actions.fill_(2)
            self.q_values = torch.zeros(5).to(self.device)
            self.action = 2
            self.score = 0

        self.count_states = self.model.increase_count_states()

    def optimize(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        states, next_states, actions, rewards, dones, prev_actions, next_actions = zip(*batch)

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        prev_actions = torch.cat(prev_actions).to(self.device)
        next_actions = torch.cat(next_actions).to(self.device)

        current_q_values = self.model(states, prev_actions).gather(1, actions.unsqueeze(1))
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