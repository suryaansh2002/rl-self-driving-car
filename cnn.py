import tensorflow as tf
import os
import numpy as np
import random
from config import VISION_W, VISION_F, VISION_B, ROUND, DL_IS_TRAINING
from tensorflow.keras import layers, Model

checkpoint_dir = 'models'

GAMMA = 0.95


class Cnn(Model):
    def __init__(self, model_name, replay_memory, num_actions=5, target=False):
        super(Cnn, self).__init__()
        self.main = not target
        self.model_name = model_name
        self.replay_memory = replay_memory
        self.num_actions = num_actions
        # Update the state shape to match your actual input
        self.state_shape = (36, 3, 1)
        self.action_shape = (4,)
        self.summary_writer = tf.summary.create_file_writer(f"{checkpoint_dir}/{model_name}")

        self.count_episodes = tf.Variable(0, trainable=False, dtype=tf.int64, name='count_episodes')
        self.count_states = tf.Variable(0, trainable=False, dtype=tf.int64, name='count_states')

        # Build the model
        self.model = self.build_model()


    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.count_episodes.numpy()

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        self.count_episodes.assign_add(1)

    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        """
        return self.count_states.numpy()

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        self.count_states.assign_add(1)


    def build_model(self):
        state_input = layers.Input(shape=self.state_shape)
        action_input = layers.Input(shape=self.action_shape)

        # CNN layers for state processing
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(state_input)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)

        # Process action input
        action_x = layers.Dense(4, activation='relu')(action_input)
        action_x = layers.Dense(4, activation='relu')(action_x)

        # Combine state and action processing
        combined = layers.Concatenate()([x, action_x])
        x = layers.Dense(100, activation='relu')(combined)
        output = layers.Dense(self.num_actions)(x)

        return Model(inputs=[state_input, action_input], outputs=output)

    def call(self, inputs):
        return self.model(inputs)

    def get_q_values(self, states, actions):
        # Ensure inputs are numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        
        # Reshape if necessary
        if states.ndim == 3:
            states = np.expand_dims(states, axis=0)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)
        
        # Get Q-values
        return self.model.predict([states, actions])



    # def optimize(self, memory, batch_size=128, learning_rate=1e-3, target_network=None):
    #     states, targets, actions = self.get_memory_component(memory, batch_size, target_network)
        
    #     with tf.GradientTape() as tape:
    #         q_values = self([states, actions])
    #         loss = tf.reduce_mean(tf.square(q_values - targets))

    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     tf.keras.optimizers.Adam(learning_rate=learning_rate).apply_gradients(zip(gradients, self.trainable_variables))

        self.log_training_loss(loss)
    def close(self):
        """Close the TensorFlow session."""
        self.session.close()

    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """
        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            ckpt = tf.train.latest_checkpoint(checkpoint_dir='{}/{}'.format(checkpoint_dir, self.model_name))

            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=ckpt)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", ckpt)

        except Exception as e:
            if not DL_IS_TRAINING:
                print(e)
                raise Exception('Failed to load checkpoint', '{}/{}'.format(checkpoint_dir, self.model_name))

            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", checkpoint_dir)
            print("Initializing variables instead.")
            # self.session.run(tf.global_variables_initializer())
            self.session.run(self.init_graph)

    def save_checkpoint(self, current_iteration):
        if not self.main or not DL_IS_TRAINING:
            return False
        """Save all variables of the TensorFlow graph to a checkpoint."""
        print("Save checkpoint")
        self.saver.save(self.session,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)

        try:
            temp_checkpoint_path = os.path.join('{}/{}'.format(checkpoint_dir, self.model_name),
                                                "checkpoint-episode-{}".format(self.get_count_episodes()))

            self.saver.save(self.session,
                            save_path=temp_checkpoint_path,
                            global_step=self.get_count_episodes())
        except Exception as e:
            print(e)

        print("Saved checkpoint.")

    # def get_q_values(self, states, actions):
    #     """
    #     Calculate and return the estimated Q-values for the given states.
    #     A single state contains two images (or channels): The most recent
    #     image-frame from the game-environment, and a motion-tracing image.
    #     See the MotionTracer-class for details.
    #     The input to this function is an array of such states which allows
    #     for batch-processing of the states. So the input is a 4-dim
    #     array with shape: [batch, height, width, state_channels].

    #     The output of this function is an array of Q-value-arrays.
    #     There is a Q-value for each possible action in the game-environment.
    #     So the output is a 2-dim array with shape: [batch, num_actions]
    #     """

    #     # Create a feed-dict for inputting the states to the Neural Network.
    #     feed_dict = {self.x: states, self.actions: [actions]}

    #     # Use TensorFlow to calculate the estimated Q-values for these states.
    #     values = self.session.run([self.q_values], feed_dict=feed_dict)

    #     return values

    def optimize(self, memory, batch_size=128, learning_rate=1e-3, target_network=None):
        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

        states, targets, actions = self.get_memory_component(memory, batch_size, target_network=target_network)

        feed_dict = {self.x: states,
                     self.actions: actions,
                     self.q_values_new: targets,
                     self.learning_rate: learning_rate}

        # Perform one optimization step and get the loss-value.
        with tf.GradientTape() as tape:
            q_values = self([states, actions])
            loss = tf.reduce_mean(tf.square(q_values - targets))

        gradients = tape.gradient(loss, self.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=learning_rate).apply_gradients(zip(gradients, self.trainable_variables))

        self.log_training_loss(loss)
    
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
            current = current.numpy()
            current[0][action] = target
            targets.append(current[0])
        
        states = np.array(states).reshape(-1, VISION_B + VISION_F + 1, VISION_W * 2 + 1, 4)
        targets = np.array(targets).reshape(-1, 5)
        actions = np.array(actions)
        return states, targets, actions

    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.
        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """
        # The tf.layers API uses this name for the weights in a conv-layer.
        variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    def get_variable_value(self, variable):
        """Return the value of a variable inside the TensorFlow graph."""

        weights = self.session.run(variable)

        return weights

    def get_tensor_value(self, tensor, state):
        """Get the value of a tensor in the Neural Network."""

        # Create a feed-dict for inputting the state to the Neural Network.
        feed_dict = {self.x: [state]}

        # Run the TensorFlow session to calculate the value of the tensor.
        output = self.session.run(tensor, feed_dict=feed_dict)

        return output

    # def get_count_episodes(self):
    #     """
    #     Get the number of episodes that has been processed in the game-environment.
    #     """
    #     return self.session.run(self.count_episodes)

    # def increase_count_episodes(self):
    #     """
    #     Increase the number of episodes that has been processed
    #     in the game-environment.
    #     """
    #     return self.session.run(self.count_episodes_increase)

    # def get_count_states(self):
    #     """
    #     Get the number of states that has been processed in the game-environment.
    #     This is not used by the TensorFlow graph. It is just a hack to save and
    #     reload the counter along with the checkpoint-file.
    #     """
    #     return self.session.run(self.count_states)

    # def increase_count_states(self):
    #     """
    #     Increase the number of states that has been processed
    #     in the game-environment.
    #     """
    #     return self.session.run(self.count_states_increase)

    def log_average_speed(self, speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='avg_speed', simple_value=speed)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_testing_speed(self, speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_speed', simple_value=speed)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_training_loss(self, loss):
        summary = tf.Summary(value=[tf.Summary.Value(tag='losses', simple_value=loss)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_total_frame(self, frame):
        summary = tf.Summary(value=[tf.Summary.Value(tag='frames', simple_value=frame)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_terminated(self, terminated):
        summary = tf.Summary(value=[tf.Summary.Value(tag='terminated', simple_value=1 if terminated else 0)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_reward(self, reward):
        summary = tf.Summary(value=[tf.Summary.Value(tag='rewards', simple_value=reward)])
        self.writer.add_summary(summary, self.get_count_episodes())

    # def log_average_test_speed(self, test_speed):
    #     summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed', simple_value=test_speed)])
    #     self.writer.add_summary(summary, 0)
    #     self.writer.add_summary(summary, 1)

    def log_average_test_speed_20(self, test_speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed_20', simple_value=test_speed)])
        self.writer.add_summary(summary, 0)
        self.writer.add_summary(summary, 1)

    def log_average_test_speed_40(self, test_speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed_40', simple_value=test_speed)])
        self.writer.add_summary(summary, 0)
        self.writer.add_summary(summary, 1)

    def log_average_test_speed_60(self, test_speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed_60', simple_value=test_speed)])
        self.writer.add_summary(summary, 0)
        self.writer.add_summary(summary, 1)

    def log_target_network_update(self):
        summary = tf.Summary(value=[tf.Summary.Value(tag='target_update', simple_value=1)])
        self.writer.add_summary(summary, self.get_count_states())


    def log_q_values(self, q_values):
        with self.summary_writer.as_default():
            tf.summary.scalar('sum_q_values', tf.reduce_sum(q_values), step=self.count_states)
        self.summary_writer.flush()

    def log_hard_brake_count(self, count):
        summary = tf.Summary(value=[tf.Summary.Value(tag='hard_brake_count', simple_value=count)])
        self.writer.add_summary(summary, self.get_count_states())

    def log_action_frequency(self, stats):
        sum = float(np.sum(stats))
        s = stats.tolist()
        for index, value in enumerate(s):
            summary = tf.Summary(value=[tf.Summary.Value(tag='action_frequency', simple_value=value/sum)])
            self.writer.add_summary(summary, index)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
