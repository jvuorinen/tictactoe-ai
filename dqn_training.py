from collections import deque
import random
import pickle

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from tictactoe import TicTacToe
from keras_models import create_model

class DQNAgent:
    def __init__(self, size, name='anonymous'):
        self.size = size
        self.memory = deque(maxlen=2000)
        self.name = name

        self.gamma = 0.8
        self.epsilon = 0.15
        # self.epsilon_min = 0.15
        # self.epsilon_decay = 0.99995
        self.learning_rate = 0.01
        self.tau = .125

        self.model = create_model(self.size, lr=self.learning_rate)
        self.target_model = create_model(self.size, lr=self.learning_rate)

    def get_action(self, state, explore=True):
        # Gets best q action or explores
        # self.epsilon *= self.epsilon_decay
        # self.epsilon = max(self.epsilon_min, self.epsilon)
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
        # self.epsilon = self.epsilon_min
        self.model.load_weights("weights/{}.h5".format(self.name))
        self.target_model.load_weights("weights/{}-target.h5".format(self.name))


class Arena:
    def __init__(self, ttt, dqn_agent):
        self.ttt = ttt
        self.dqn_agent = dqn_agent
        try:
            self.dqn_agent.load_weights()
        except OSError:
            print("No weights found")

        self.dqn_memory_buffer = None

        self.reward_victory = 20
        self.reward_defeat = -15
        self.reward_illegal = -100

        self.learning_history = pd.DataFrame(columns=['Game', 'Victory', 'Length', 'Errors'], dtype=int)

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

    def _play_2nd_player_move(self, use_dqn=True):
        if use_dqn:
            #Play 2nd player move using dqn
            state = self.get_state()
            state[0], state[1] = state[1], state[0]
            action = self.dqn_agent.get_action(state)
            move = self.possible_actions[action]
            r = self.ttt.play_move(move)
            if not r:
                # print("Illegal move, playing randomly...")
                self.ttt.play_random_move()
        else:
            self.ttt.play_random_move()

    def train_dqn(self, n_games=10, autosave=True, use_dqn=True, warm_start=False):
        my_id, opp_id = 1, 2

        for i in range(n_games):
            self.ttt.reset()

            if warm_start:
                times = np.random.randint(4, 14)
                for n in range(times):
                    self.ttt.play_random_move()
                    if self.ttt.finished:
                        self.ttt.reset()
                        break

            if self.ttt.turn == opp_id:
                self._play_2nd_player_move(use_dqn)

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

                if done:
                    break

                self._play_2nd_player_move(use_dqn)
                cur_state = new_state

            self.dqn_agent.train_batch()  # internally iterates default (prediction) model
            self.dqn_agent.target_train()  # iterates target model

            print("Game {}, Victory {}, game length {}, errors {}, e: {:0.2f}".format(
                i,
                self.ttt.winner == 1,
                step + 1,
                error_counter,
                self.dqn_agent.epsilon))

            self.learning_history = self.learning_history.append(
                {'Game': i, 'Victory': self.ttt.winner == 1, 'Length': step + 1, 'Errors': error_counter}, ignore_index=True)

            if ((i + 1) % 500 == 0) & autosave:
                self.dqn_agent.save_model()


    def play_against_human(self):
        self.ttt.reset()
        while not self.ttt.finished:
            if self.ttt.turn == 1:
                print(self.ttt)
                ip = int(input())
                move = self.possible_actions[ip - 1]
                self.ttt.play_move(move)
            else:
                self._play_2nd_player_move()
        print(self.ttt)


if __name__ == "__main__":

    ttt = TicTacToe(size=6, win_length=4)
    dqn_agent = DQNAgent(size=ttt.size, name="board-6-4-big-with-skipped")
    arena = Arena(ttt, dqn_agent)

    arena.train_dqn(n_games=100, autosave=False, warm_start=True)
    # arena.play_against_human()
    # arena.train_dqn_against_random(n_games=50000)
    # h = arena.learning_history.set_index('Game')
    # h.to_pickle('reports/learning_history-with-skipped-5.pkl')
    #
    # h.Errors.rolling(1000).mean().plot()
    # h.Length.rolling(1000).mean().plot()
    # h.Victory.rolling(1000).mean().plot()
    #
