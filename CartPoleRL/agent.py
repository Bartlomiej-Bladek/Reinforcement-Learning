import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self, action_size, state_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.policy_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()


    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            action = self.policy_network.predict(state.reshape(1, self.state_size))
        return np.argmax(action[0])

    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batches = np.random.choice(len(self.memory), batch_size, replace=False)

        for i in batches:
            state, action, next_state, reward, done = self.memory[i]
            target = reward

            target = reward + self.gamma * np.amax(self.policy_network.predict(next_state.reshape(1, self.state_size))[0])

            target_f = self.target_network.predict(state.reshape(1, self.state_size))
            target_f[0][action] = target

            self.policy_network.fit(state.reshape(1, self.state_size), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())





    # def learn(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
    #
    #     batches = np.random.choice(len(self.memory), batch_size, replace=False)
    #
    #     for i in batches:
    #         state, action, next_state, reward, done = self.memory[i]
    #         target = reward
    #
    #         target = reward + self.gamma * np.amax(self.policy_network.predict(next_state.reshape(1, self.state_size))[0])
    #
    #         target_f = self.target_network.predict(state.reshape(1, self.state_size))
    #         target_f[0][action] = target
    #
    #         self.policy_network.fit(state.reshape(1, self.state_size), target_f, epochs=1, verbose=0)
    #
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay



    # def learn(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
    #
    #     batches = np.random.choice(len(self.memory), batch_size, replace=False)
    #
    #     for i in batches:
    #         state, action, next_state, reward, done = self.memory[i]
    #         target = self.policy_network.predict(state.reshape(1, self.state_size))
    #         if done:
    #             target[0][action] = reward
    #         else:
    #             target[0][action] = reward + self.gamma * np.amax(self.target_network.predict(next_state.reshape(1, self.state_size))[0])
    #
    #
    #         self.policy_network.fit(state.reshape(1, self.state_size), target, epochs=1, verbose=0)
    #
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay