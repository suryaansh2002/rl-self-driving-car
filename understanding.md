
## `deep_trafic_agent.py`

### 1. Initialization (`__init__` method):
- Sets up the agent with a model name and initializes various parameters.
- Creates two neural networks:
  - `self.model` (main network).
  - `self.target_model` (target network for stable Q-learning).
- Initializes memory (replay buffer) as a deque.
- Sets up state and action spaces, initializes counters for states and episodes.

### 2. Action Selection (`act` method):
- Takes the current state as input.
- Implements epsilon-greedy strategy for exploration vs. exploitation:
  - With probability epsilon, chooses a random action (exploration).
  - Otherwise, uses the neural network to predict Q-values and selects the action with the highest Q-value (exploitation).
- Returns the selected action and its corresponding Q-values.

### 3. Memory Management (`remember` method):
- Stores experiences (state, action, reward, next state, done) in the replay buffer.
- Manages the size of the replay buffer, removing old experiences if it exceeds `MAX_MEM`.
- Triggers the learning process if certain conditions are met (e.g., enough samples collected).

### 4. Learning Process:
- Calls `self.model.optimize()` to update the neural network.
- Uses experience replay to sample batches from memory for training.
- Periodically updates the target network to stabilize learning.
- Saves model checkpoints at regular intervals.

### 5. Epsilon Decay:
- Implements a linear decay for the epsilon value used in the epsilon-greedy strategy.
- Starts with a high exploration rate, gradually shifting toward more exploitation.

### 6. Utility Methods:
- `get_action_name` and `get_action_index`: Convert between action indices and names.
- `increase_count_states` and `increase_count_episodes`: Update counters for states processed and episodes completed.

### 7. Reward Handling:
- Tracks cumulative score within an episode.
- Calculates reward difference between steps.

### 8. Episode Management:
- Resets various parameters at the end of an episode when `end_episode` is `True`.

**Summary**:  
The `DeepTrafficAgent` implements a Deep Q-Network (DQN) with:
- Two networks (main and target) to stabilize learning.
- Experience replay to break correlations between samples.
- Epsilon-greedy exploration with a decaying epsilon strategy.


## `cnn.py`

### 1. Class Initialization (`__init__` method):

- Inherits from TensorFlow's `Model` class.
- Sets attributes like model name, replay memory, number of actions, etc.
- Defines the shapes for state and action inputs.
- Creates a TensorFlow summary writer for logging.
- Initializes counters for episodes and states as TensorFlow variables.
- Builds the neural network model.

### 2. Model Architecture (`build_model` method):
- Creates a neural network with two inputs: `state` and `action`.
- The `state` input goes through two convolutional layers and is then flattened.
- The `action` input goes through two dense layers.
- The processed state and action are concatenated and passed through a final dense layer.
- Returns a Keras `Model` object.

### 3. Forward Pass (`call` method):
- Defines how the model processes inputs during a forward pass.

### 4. Q-Value Prediction (`get_q_values` method):
- Takes states and actions as input.
- Ensures inputs are properly formatted as numpy arrays.
- Uses the model to predict Q-values for the given states and actions.

### 5. Optimization (`optimize` method):
- Samples a batch of experiences from the replay memory.
- Computes the loss between predicted Q-values and target Q-values.
- Performs a gradient descent step to update the model's weights.

### 6. Memory Sampling (`get_memory_component` method):
- Samples a batch of experiences from the replay memory.
- Computes target Q-values using the Bellman equation.
- Prepares states, actions, and targets for training.

### 7. Checkpoint Management:
- `save_checkpoint`: Saves the model's state to a file.
- `load_checkpoint`: Loads the model's state from a file.

### 8. Logging Methods:
- Methods like `log_average_speed` and `log_training_loss` for logging metrics during training.
- Uses TensorFlow's summary writing for visualization in TensorBoard.

### 9. Utility Methods:
- Methods for incrementing episode and state counters.
- Methods for accessing model weights and variables.

### 10. Histogram Logging (`log_histogram` method):
- Logs histograms of values, useful for visualizing weight distributions.

**Summary**:  
The `CNN` class defines the neural network architecture, handles training, manages checkpoints, and provides logging capabilities. It's designed for use within a reinforcement learning framework (DQN).


## `gui.py` Execution Flow:

1. **Main GUI Initialization**:
   - Creates the main window (likely using PyQt or Tkinter).
   - Sets up the game environment.

2. **Game Environment Setup** (possibly in `game.py` or `environment.py`):
   - Initializes the road, lanes, and other game objects.
   - Sets up the initial state of the traffic.

3. **Agent Initialization** (`deep_traffic_agent.py`):
   - Creates an instance of `DeepTrafficAgent`.
     - Initializes the CNN model (`cnn.py`).
     - Sets up the replay memory.
     - Loads pre-trained weights if available.

4. **Main Game Loop**:
   - For each frame:
     1. Updates game state (`game.py`).
     2. Gets current state (`game.py` -> `deep_traffic_agent.py`).
     3. Agent chooses action (`deep_traffic_agent.py` -> `cnn.py`):
        - `DeepTrafficAgent.act()` is called.
        - CNN forward pass if not exploring.
     4. Applies action to the game state.
     5. Calculates reward (`game.py`).
     6. Agent remembers experience (`deep_traffic_agent.py`):
        - `DeepTrafficAgent.remember()` is called.
        - Adds experience to replay memory.
        - Potentially triggers learning.
     7. Updates display (`gui.py`).

5. **End of Episode Handling** (`gui.py -> deep_traffic_agent.py`):
   - Resets game state.
   - Updates episode counters.
   - Potentially saves model checkpoints (`cnn.py`).

6. **User Interaction Handling** (`gui.py`):
   - Responds to user inputs (start, stop, change parameters, etc.).


## `car.py`

### 1. Imports and Constants:
- Imports necessary modules: `math`, `pygame`, `os`, and `numpy`.
- Constants:
  - `MAX_SPEED`: 110 km/h.
  - `DEFAULT_CAR_POS`: 700.
  - `IMAGE_PATH`: './images'.

### 2. Image Loading:
- If `VISUALENABLED`, loads and scales car images using Pygame.

### 3. Probability Weights:
- `direction_weight`: Probabilities for lane change directions (Left: 1%, Middle: 98%, Right: 1%).
- `move_weight`: Probabilities for speed changes (Accelerate: 30%, Maintain: 50%, Decelerate: 20%).

### 4. Car Class:
#### Methods:
- `__init__`: Initializes car attributes like speed, position, lane, etc. Sets up player types.
- `identify`: Updates the car's position in the lane map, handles out-of-bounds conditions.
- `accelerate` / `decelerate`: Adjusts the car's speed within limits.
- `check_switch_lane`: Handles lane switching logic.
- `move`: Executes the chosen move action.
- `switch_lane`: Manages the lane-switching process.
- `identify_available_moves`: Determines available moves based on traffic.
- `random`: Chooses random actions for non-player cars.
- `relative_pos_subject`: Adjusts the car's position relative to the subject car.
- `decide`: Makes decisions for the car based on its current state.
- `draw`: Updates the car's position and renders it on the screen.
- `get_vision`: Creates a representation of the car's surrounding environment.
- `get_subjective_vision`: Returns positions of nearby cars controlled by the subject.

**Summary**:  
The `Car` class simulates car behavior, including movement, lane switching, and decision-making in traffic. It handles both player-controlled and AI-controlled cars in a simulation.



### **Cnn Class** (Detailed Breakdown):
1. **Action Shape Discrepancy**:
   - A discrepancy was noted between `num_actions` (default is 5) and `action_shape` (which is defined as `(4,)`).
   - It's important to check whether the action space truly has 5 possible actions or if the `action_shape` was mistakenly defined as 4. Consider changing `self.action_shape` to `(self.num_actions,)` for consistency.

2. **Purpose of `target_network`**:
   - The `target_network` helps stabilize learning by providing stable target Q-values.
   - It reduces overestimation bias in Q-learning algorithms, such as Double DQN.
   - The target network is updated less frequently than the main network, which provides delayed updates to improve stability.

---

### **Detailed Code Explanation**:
1. **Optimize Method**:
   - Performs the gradient descent step using TensorFlow's `GradientTape` to compute gradients.
   - Calculates loss (mean squared error) between predicted Q-values and target Q-values.
   - Uses Adam optimizer to apply gradients and log the training loss.
   - Key utility of this method: improving model performance by updating weights based on replayed experiences.

2. **get_memory_component Method**:
   - Samples a random batch from the replay memory.
   - Computes target Q-values for each experience in the batch, based on the Bellman equation.
   - Handles cases where an episode ends and adjusts Q-values accordingly.



### **Build Model Details**:
The CNN model's structure includes:
- **Conv2D layers**: To process spatial data from the state.
- **Dense layers**: To handle the action input and state-action combination.
- **Concatenation layer**: Combines processed state and action data.
- **Final output layer**: Predicts Q-values for the given state-action pairs.


### **Detailed Shapes and Explanation**:
1. **Q-Value Prediction Update**:
   - Explains how Q-values are updated using the target network and the Bellman equation.
   - Important shapes: `(36, 3, 1)` for the state, `(4,)` for the action, `(5,)` for the predicted Q-values.

2. **Reshaping States**:
   - The reshaping of states into a 4D array: `states = np.array(states).reshape(-1, VISION_B + VISION_F + 1, VISION_W * 2 + 1, 4)`.
   - This represents the environment's vertical and horizontal vision, plus 4 channels for richer state information.

---

