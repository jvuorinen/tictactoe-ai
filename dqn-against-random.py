from collections import deque
import random
import pickle

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Concatenate, Flatten
from keras.optimizers import Adam

from tictactoe import TicTacToe


class DQNAgent:
    def __init__(self, size, name='anonymous'):
        self.size = size
        self.memory = deque(maxlen=2000)
        self.name = name

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.01
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        s = self.size
        input_me = Input(name="input-me", shape=(s, s, 1))
        input_op = Input(name="input-opponent", shape=(s, s, 1))
        convolution = Conv2D(13, kernel_size=2, strides=2, padding='valid', use_bias=False)

        conv_me = Flatten()(convolution(input_me))
        conv_op = Flatten()(convolution(input_op))

        concat = Concatenate()([conv_me, conv_op])
        dense_1 = Dense(100)(concat)
        out = Dense(s ** 2)(dense_1)

        model = Model([input_me, input_op], out)
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def get_action(self, state, explore=True):
        # Gets best q action or explores
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if explore & (np.random.random() < self.epsilon):
            return np.random.randint(self.size ** 2)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def train_batch(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for s in samples:
            state, action, reward, new_state, done = s
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        print('Saving weights')
        self.model.save("weights/{}.h5".format(self.name))
        self.target_model.save("weights/{}-target.h5".format(self.name))

    def load_weights(self):
        print('Loading weights')
        self.epsilon = self.epsilon_min
        self.model.load_weights("weights/{}.h5".format(self.name))
        self.target_model.load_weights("weights/{}-target.h5".format(self.name))


class Arena:
    def __init__(self, ttt):
        self.ttt = ttt
        self.dqn_agent = DQNAgent(size=ttt.size)
        self.dqn_memory_buffer = None

        self.reward_victory = 20
        self.reward_defeat = 0
        self.reward_illegal = -5

        self.learning_history = pd.DataFrame(columns=['Game', 'Winner', 'Length', 'Errors'], dtype=int)

        self.possible_actions = [(i, j) for i in range(self.ttt.size) for j in range(self.ttt.size)]

    def get_state(self):
        s = self.ttt.size
        return [self.ttt.board_p1.astype(np.int).reshape(1, s, s, 1),
                self.ttt.board_p2.astype(np.int).reshape(1, s, s, 1)]

    def _step(self, action_idx, my_player_id=1):
        # Gets best move and plays it
        move = self.possible_actions[action_idx]
        r = self.ttt.play_move(move)
        if not r:
            reward = self.reward_illegal
            # self.ttt.play_random_move() # Plays a random move if dqn wants to play an illegal move
        elif self.ttt.winner == my_player_id:
            reward = self.reward_victory
        else:
            reward = 0
        next_state = self.get_state()
        done = self.ttt.finished
        return next_state, reward, done

    def train_dqn_against_random(self, n_games=10, autosave=True):
        my_id, opp_id = 1, 2

        for i in range(n_games):
            self.ttt.reset()
            if self.ttt.turn != my_id:
                self.ttt.play_random_move()
            cur_state = self.get_state()
            error_counter = 0
            for step in range(self.ttt.n_cells):
                action_idx = self.dqn_agent.get_action(cur_state)
                new_state, reward, done = self._step(action_idx)

                if reward == self.reward_illegal:  # Potential bug if reward_illegal == reward_defeat
                    error_counter += 1

                if self.ttt.winner == opp_id:
                    self.dqn_memory_buffer[2] = self.reward_defeat
                    # print(self.dqn_memory_buffer[2])

                if self.dqn_memory_buffer:
                    self.dqn_agent.remember(*self.dqn_memory_buffer)

                self.dqn_memory_buffer = [cur_state, action_idx, reward, new_state, done]

                if done: break

                self.ttt.play_random_move()
                cur_state = new_state

            self.dqn_agent.train_batch()  # internally iterates default (prediction) model
            self.dqn_agent.target_train()  # iterates target model

            print("Game {}, winner {}, game length {}, errors {}, e: {:0.2f}".format(
                i,
                ttt.winner,
                step + 1,
                error_counter,
                self.dqn_agent.epsilon))

            self.learning_history = self.learning_history.append(
                {'Game': i, 'Winner': ttt.winner, 'Length': step + 1, 'Errors': error_counter}, ignore_index=True)

            if ((i + 1) % 100 == 0) & autosave:
                self.dqn_agent.save_model()

    def play_against_human(self):
        self.ttt.reset()
        while self.ttt.finished == False:
            if self.ttt.turn == 1:
                state = self.env.get_state()
                # state = np.reshape(state, [1, self.env.n_input])
                action = self.get_action(state, explore=False)
                move = self.env.action_space[action]
                self.env.ttt.play_move(move)
            else:
                print(self.env.ttt)
                ip = int(input())
                move = self.env.action_space[ip - 1]
                self.env.ttt.play_move(move)
        print(self.env.ttt)


if __name__ == "__main__":
    ttt = TicTacToe(size=6, win_length=4)
    arena = Arena(ttt)

    arena.train_dqn_against_random(n_games=2000)
    h = arena.learning_history.set_index('Game')
    h.to_pickle('reports/learning_history.pkl')
    h.rolling(10).mean().plot()

