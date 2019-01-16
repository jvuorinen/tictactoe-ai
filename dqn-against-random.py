from tictactoe import TicTacToe
import numpy as np
import gym

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import os
from collections import deque

MODEL_BACKUP = "dqn-against-random-weights.h5"
MY_ID = 1

class GymLikeEnv:
    def __init__(self, ttt):
        self.ttt = ttt
        self.done = False
        self.n_input = 16
        self.n_output = 8
        if ttt.turn != 1:
            ttt.play_random_move()


    def reset(self):
        self.ttt.reset()
        self.done = False
        if self.ttt.turn != 1:
            self.ttt.play_random_move()

    def get_state(self):
        return np.concatenate([self.ttt.board_p1.flatten().astype(np.int), self.ttt.board_p2.flatten().astype(np.int)])

    def step(self, input):
        move = ((0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2))[input]
        r = self.ttt.play_move(move)
        if r == False:
            reward = -5
        elif self.ttt.winner == MY_ID:
            reward = 10
        else:
            reward = 0
        next_state = self.get_state()
        done = self.ttt.finished
        return (next_state, reward, done, {})


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.weight_backup = MODEL_BACKUP

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(6, input_dim=state_shape[0], activation="relu"))
        # model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        # if os.path.isfile(self.weight_backup):
        #     model.load_weights(self.weight_backup)
        #     self.epsilon = self.epsilon_min

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
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

    def save_model(self, fn):
        self.model.save(fn)


if __name__ == "__main__":
    g = gym.make('CartPole-v1')

    env = GymLikeEnv(TicTacToe())
    self = env


    env.ttt.print_state()
    env.step(4)

    #
    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    #
    # trials = 10
    # trial_len = 500
    #
    # # updateTargetNetwork = 1000
    # dqn_agent = DQN(env=env)
    # steps = []
    #
    # win_counter = 0
    #
    # for trial in range(trials):
    #     cur_state = env.reset()
    #     cur_state = np.reshape(cur_state, [1, state_size])
    #     max_reward = 0
    #     for step in range(trial_len):
    #         # env.render()
    #         action = dqn_agent.act(cur_state)
    #         new_state, reward, done, _ = env.step(action)
    #
    #         reward = reward if not done else -20
    #         new_state = np.reshape(new_state, [1, state_size])
    #         dqn_agent.remember(cur_state, action, reward, new_state, done)
    #
    #         cur_state = new_state
    #
    #         if done: break
    #
    #     if step > 400:
    #         win_counter += 1
    #     else:
    #         win_counter = 0
    #
    #     if win_counter == 10:
    #         break
    #
    #     dqn_agent.replay()  # internally iterates default (prediction) model
    #     dqn_agent.target_train()  # iterates target model
    #
    #
    #     print("Trial {}, length {}, e: {:0.2f}, lr: {:0.4f}".format(trial, step, dqn_agent.epsilon, dqn_agent.learning_rate))
    #     if step % 10 == 0:
    #         dqn_agent.save_model(MODEL_BACKUP)
    #
    #
    # for trial in range(10):
    #     cur_state = env.reset()
    #     cur_state = np.reshape(cur_state, [1, state_size])
    #     for step in range(trial_len):
    #         env.render()
    #         action = dqn_agent.act(cur_state)
    #         new_state, reward, done, _ = env.step(action)
    #
    #         # reward = reward if not done else -20
    #         new_state = np.reshape(new_state, [1, state_size])
    #
    #         cur_state = new_state
    #         if done:
    #             break
    #

