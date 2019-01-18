from tictactoe_generic import TicTacToe
import numpy as np
import gym
from random import sample

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import os
import sys
from collections import deque

WEIGHTS_FILE_MODEL = "dqn-against-random-weights.h5"
WEIGHTS_FILE_TARGET = "dqn-against-random-target-weights.h5"
MY_ID = 1

class GymLikeEnv:
    def __init__(self, ttt):
        self.ttt = ttt
        self.done = False
        self.n_input = 18
        self.n_output = 9
        if ttt.turn != 1:
            ttt.play_random_move()

    def get_random_action(self):
        return sample([0,1,2,3,4,5,6,8], 1)[0]

    def reset(self):
        self.ttt.reset()
        self.done = False
        if self.ttt.turn != 1:
            self.ttt.play_random_move()
        return self.get_state()

    def get_state(self):
        return np.concatenate([self.ttt.board_p1.flatten().astype(np.int), self.ttt.board_p2.flatten().astype(np.int)])

    def step(self, action):
        move = ((0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2))[action]
        r = self.ttt.play_move(move)
        if r == False:
            reward = -5
        elif self.ttt.winner == MY_ID:
            reward = 10
        else:
            reward = 0
            self.ttt.play_random_move()
        next_state = self.get_state()
        done = self.ttt.finished
        return (next_state, reward, done, {})


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(6, input_dim=self.env.n_input, activation="relu"))
        # model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.n_output))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        # if os.path.isfile(self.weight_backup):
        #     model.load_weights(self.weight_backup)
        #     self.epsilon = self.epsilon_min

        return model

    def act(self, state, explore=True):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if explore & (np.random.random() < self.epsilon):
            return self.env.get_random_action()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        print('Saving weights')
        self.model.save(WEIGHTS_FILE_MODEL)
        self.target_model.save(WEIGHTS_FILE_TARGET)

    def load_weights(self):
        print('Loading weights')
        self.epsilon = self.epsilon_min
        self.model.load_weights(WEIGHTS_FILE_MODEL)
        self.target_model.load_weights(WEIGHTS_FILE_TARGET)

    def train_model(self, trials = 1000, trial_len=10, autosave = False):
        #Model raining
        for trial in range(trials):
            cur_state = self.env.reset()
            cur_state = np.reshape(cur_state, [1, self.env.n_input])
            reward_sum = 0
            for step in range(trial_len):
                # env.render()
                action = self.act(cur_state)
                new_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                # reward = reward if not done else -20
                new_state = np.reshape(new_state, [1, self.env.n_input])
                dqn_agent.remember(cur_state, action, reward, new_state, done)

                cur_state = new_state

                if done: break

            self.replay()  # internally iterates default (prediction) model
            self.target_train()  # iterates target model

            print("Trial {}, game length {}, rewards {}, e: {:0.2f}, lr: {:0.4f}".format(trial, step, reward_sum, dqn_agent.epsilon, dqn_agent.learning_rate))
            if (step % 100 == 0) & autosave:
                self.save_model()

    def play_against_human(self):
        self.env.ttt.reset()
        while self.env.ttt.finished == False:
            if self.env.ttt.turn == 1:
                state = self.env.get_state()
                state = np.reshape(state, [1, self.env.n_input])
                action = self.act(state, explore=False)
                move = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))[action]
                self.env.ttt.play_move(move)
            else:
                self.env.ttt.print_state()
                ip = int(input())
                move = ((0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2))[ip-1]
                self.env.ttt.play_move(move)
        self.env.ttt.print_state()


if __name__ == "__main__":
    g = gym.make('CartPole-v1')

    env = GymLikeEnv(TicTacToe())

    dqn_agent = DQN(env=env)

    # dqn_agent.load_weights()
    dqn_agent.train_model(2000)
    # dqn_agent.save_model()
    # dqn_agent.play_against_human()
