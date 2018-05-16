# coding=utf-8
import matplotlib
matplotlib.use('Agg')
from schedenv import SchedEnv
import os
import numpy as np
import random
import heft
import heft_new
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
import copy
import pylab


class REINFORCE:

    def __init__(self, q, n1, n2, discount_factor, lr):
        """"""
        self.n1 = n1
        self.n2 = n2
        self.load_model = True
        """actions which agent can do"""
        self.action_space = [a+1 for a in range(q)]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 1 + 2 * q
        self.discount_factor = discount_factor
        self.learning_rate = lr

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []
        self.makespans = []

        # if self.load_model:
        #     self.model.load_weights('./save_model/REINFORCE_trained_v=' + str(V) + "q=" + str(q) + "_"
        #                             + str(self.n1) + "_" + str(self.n2) + ".h5")

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.n1, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.n2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)
        return train

    # get action from policy network
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        # print(np.random.choice(self.action_size, 1, p=policy)[0])  # 0, 1, 2
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action - 1] = 1
        self.actions.append(act)

    # update policy neural network
    def train_model(self):
        discounted_rewards = np.float64(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


def read_parameter(q, t, v):
    """q is the number of processor, t is which line"""
    filepath = 'graph_parameter\\'
    filename = filepath + 'graph_parameter_v=' + str(v) + 'q=' + str(q) + '.txt'
    f = open(filename, 'r')
    line = f.readlines()

    line_need = line[t - 1]
    line_list = line_need.split()
    v = line_list[0]
    ccr = line_list[1]
    alpha = line_list[2]
    beta = line_list[3]
    q = line_list[4]
    return v, ccr, alpha, beta, q


def cyclic(N, Q, V, N1, N2, discount_factor, lr):
    """N is the number of DAG, Q is the number of processors"""
    t = 1
    while t <= N:
        heft = heft_new.Heft(Q, t, V)
        heft_make_span = heft.heft()
        num_task = heft.v
        env = SchedEnv(num_task, Q, t)
        agent = REINFORCE(Q, N1, N2, discount_factor, lr)
        # print(env.dag, heft_make_span)

        global_step = 0
        scores, episodes = [], []
        make_spans = []

        for e in range(EPISODES):
            done = False
            score = 0
            # fresh env
            state = env.reset(num_task, Q, t)
            state = np.reshape(state, [1, 1 + 2 * Q])

            while not done:
                global_step += 1
                # get action for the current state and go one step in environment
                action = agent.get_action(state) + 1

                next_state, reward, done = env.step(action)
                if len(next_state) != 1:
                    next_state = np.reshape(next_state, [1, 1 + 2 * Q])

                agent.append_sample(state, action, reward)
                score += reward
                state = copy.deepcopy(next_state)

                if done:
                    # update policy neural network for each episode

                    agent.train_model()
                    scores.append(score)
                    make_spans.append(env.makespan)
                    episodes.append(e)

                    score = round(score, 2)
                    # print("episode:", e, "  score:", score, "  time_step:", global_step)
                    if e % 10 == 0:
                        print("episode:", e, "  makespan:", score, "  time_step:", global_step)

            if e % 10 == 0:

                pylab.plot(episodes, scores, 'b')

                pylab.xlabel("Episode")
                pylab.ylabel("- Makespan")
                # pylab.plot(episodes, make_spans, 'b')
                # pylab.axhline(-heft.make_span, linewidth=0.5, color='r')
                pylab.axhline(-heft_make_span, linewidth=0.5, color='r')
                """"""
                n, ccr, alpha, beta, q = read_parameter(Q, t, V)

                info = str(N1) + "*" + str(N2) + " ccr=" + str(ccr) + " alpha=" + str(alpha) + " beta=" + str(beta)
                pylab.title(info)
                info1 = "_ccr=" + str(ccr) + " alpha=" + str(alpha) + " beta=" + str(beta)
                save_path = "./save_graph/v=" + str(V) + "q=" + str(Q) + "/"
                save_name = save_path + "_" + str(t) + info1 + ".png"
                file_dir = os.path.split(save_name)[0]
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                pylab.savefig(save_name)
                pylab.close()
                if N >= 50:
                    agent.model.save_weights("./save_model/REINFORCE_trained_v=" + str(V) + "q=" + str(q) + "_"
                                             + str(N1) + "_" + str(N2) + ".h5")

        t += 1


if __name__ == "__main__":
    EPISODES = 100
    N = 2
    Q = 3
    V = 100
    N1 = 10
    N2 = 10
    discount_factor = 1
    lr = 0.001

    cyclic(N, Q, V, N1, N2, discount_factor, lr)

