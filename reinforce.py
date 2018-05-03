# coding=utf-8
from schedenv import SchedEnv
import numpy as np
import random
import heft
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
import copy
import pylab


EPISODES = 1000


class REINFORCE:

    def __init__(self, q):
        """"""
        # self.load_model = True
        """actions which agent can do"""
        self.action_space = [a+1 for a in range(q)]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 7
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []
        self.makespans = []

        """
        if self.load_model:
            # self.model.load_weights('./save_model/reinforce_trained.h5')
            self.model.load_weights('./save_model/REINFORCE.h5')
        """

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def optimizer(self):
        action = K.placeholder(shape=[None, 3])
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
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


"""
num_task = heft.v
env = SchedEnv(num_task)
agent = REINFORCE(env.q)

"""

if __name__ == "__main__":
    num_task = heft.v
    env = SchedEnv(num_task)
    agent = REINFORCE(env.q)

    global_step = 0
    scores, episodes = [], []
    make_spans = []

    for e in range(EPISODES):
        done = False
        # score = 0
        # fresh env
        env.reset(num_task)
        state = [0, 0, 0, 0, 0, 0, 0]
        state = np.reshape(state, [1, 7])

        while not done:
            global_step += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state) + 1

            next_state, reward, done = env.step(action)
            if len(next_state) != 1:
                next_state = np.reshape(next_state, [1, 7])

            agent.append_sample(state, action, reward)
            # score += reward
            state = copy.deepcopy(next_state)

            if done:
                # update policy neural network for each episode

                agent.train_model()
                # scores.append(score)
                make_spans.append(env.makespan)
                episodes.append(e)

                # score = round(score, 2)
                # print("episode:", e, "  score:", score, "  time_step:",
                #       global_step)
                if e % 10 == 0:
                    print("episode:", e, "  makespan:", env.makespan, "  time_step:", global_step)

        if e % 10 == 0:
            # pylab.plot(episodes, scores, 'b')
            pylab.xlabel("Episode")
            pylab.ylabel("Makespan")
            pylab.plot(episodes, make_spans, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/REINFORCE.h5")


"""
num_task = heft.v
env = SchedEnv(num_task)
num_pi = len(heft.computation_costs[0])
done = False

for n in range(10):
    action = random.randrange(1, num_pi + 1)    # random action is select Pj: 1-n
    next_state, reward, done = env.step(action)
    if n > 0:
        print("next_state =", next_state, "reward =", reward, "done =", done)
        print("--------------------------------------------------------------------------")

print("env.scheduler =", env.scheduler)

reinforce = REINFORCE(env.q)
"""