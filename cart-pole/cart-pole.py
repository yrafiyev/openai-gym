import gym
import numpy as np
import random
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

class GymLoader:

    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def render(self):
        self.env.render()

    def reset(self):
        return self.env.reset()

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        return self.env.step(action)


class QAgent:

    gamma = .95
    epsilon = 1
    epsilon_decay = 0.995
    epsilon_min = 0.1
    num_episodes = 1000
    batch_size = 32
    memory_size = 50000
    layer_dimension = 12
    rewards = []

    def __init__(self, render):
        self.experience = collections.deque(maxlen=self.memory_size)
        self.updateModel = None
        self.session = None
        self.Q_out = None
        self.predict = None
        self.target_Q = None
        self.input = None
        self.render = render
        self.gym = GymLoader()
        self.intialize_NN()

    def intialize_NN(self):

        # These lines establish the feed-forward part of the network used to choose actions
        self.input = tf.placeholder(tf.float32, [1, 4])

        # First Layer weights
        W = tf.Variable(tf.random_uniform([4, self.layer_dimension], 0.001, 0.002))
        W_hidden = tf.Variable(tf.random_uniform([self.layer_dimension, self.layer_dimension], 0.001, 0.002))
        W_hidden_second = tf.Variable(tf.random_uniform([self.layer_dimension, self.layer_dimension], 0.001, 0.002))
        W_output = tf.Variable(tf.random_uniform([self.layer_dimension, 2], 0.001, 0.002))

        # Two bias vectors
        B = tf.Variable(tf.constant(0.0001, shape=[self.layer_dimension]))
        B_hidden = tf.Variable(tf.constant(0.0001, shape=[self.layer_dimension]))
        B_hidden_second = tf.Variable(tf.constant(0.0001, shape=[self.layer_dimension]))

        # Two Hidden Layer Model
        layer = tf.nn.relu_layer(self.input, W, B)
        hidden_layer = tf.nn.relu_layer(layer, W_hidden, B_hidden)
        hidden_layer_second = tf.nn.relu_layer(hidden_layer, W_hidden_second, B_hidden_second)

        self.Q_out = tf.matmul(hidden_layer_second, W_output)
        self.predict = tf.argmax(self.Q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_Q = tf.placeholder(shape=[1, 2], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.target_Q - self.Q_out))
        trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.updateModel = trainer.minimize(loss)

        self.session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def store(self, state, action, next_state, reward, done):
        self.experience.append([state, action, next_state, reward, done])

    def sample_batch(self):
        if len(self.experience) < self.batch_size:
            return []
        return random.sample(self.experience, self.batch_size)

    def predict(self, state):
        return self.session.run([self.predict, self.Q_out], feed_dict={input: np.reshape(state, [1, 4])})

    def train(self, state, target_Q):
        self.session.run([self.updateModel], feed_dict={input: np.reshape(state, [1, 4]), self.target_Q: target_Q})

    def get_model(self):
        return [self.predict, self.Q_out]

    def get_epsilon(self):
        return self.epsilon

    def update_epsilon(self):
        self.epsilon = self.epsilon_min if self.epsilon <= self.epsilon_min else self.epsilon * self.epsilon_decay

    def get_gamma(self):
        return self.gamma

    def act(self, state):
        action = self.session.run(self.predict, feed_dict={self.input: np.reshape(state, [1, 4])})
        if np.random.rand(1) < self.epsilon:
            action[0] = self.gym.sample_action()
        return action

    def replay(self):
        # replay what you learned
        randomList = self.sample_batch()

        for _state, _action, _next_state, _reward, _done in randomList:

            # In ideal, separate in two type of predictions
            Q_hat = self.session.run(self.Q_out, feed_dict={self.input: np.reshape(_next_state, [1, 4])})
            target_Q = self.session.run(self.Q_out, feed_dict={self.input: np.reshape(_state, [1, 4])})

            if not _done:
                target_Q[0, _action[0]] = _reward + self.gamma * np.max(Q_hat)
            else:
                target_Q[0, _action[0]] = _reward

            # Train our network using target and predicted Q values
            self.session.run(self.updateModel,
                             feed_dict={self.input: np.reshape(_state, [1, 4]), self.target_Q: target_Q})

    def train(self, number_of_episodes):

        for i in range(number_of_episodes):

            # Reset environment and get first new observation
            state = self.gym.reset()
            step = 0

            while True:
                if self.render:
                    self.gym.render()

                # Choose an action (with epsilon chance of random action) from the Q-network
                action = self.act(state)

                # Get new state and reward from environment
                next_state, reward, done, _ = self.gym.step(action[0])

                self.store(state, action, next_state, reward, done)

                self.replay()

                state = next_state
                step = step + 1

                if done:
                    print("Episode #: ", i, " Total Reward: ", step)
                    self.update_epsilon()
                    self.rewards.append(step)
                    step = 0
                    break

    def plot(self):
        plt.plot(self.rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episode #')
        plt.show()

agent = QAgent(render=False)
agent.train(number_of_episodes=1000)
agent.plot()
