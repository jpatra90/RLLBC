import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {
            (-1, -1): {'exit': 0}
        }
        # self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    # TODO 3.3.02
    def addNewKeyValInQ(self, state, action):
        newState = state not in self.Q
        newAction = False if newState else True if action not in self.Q[state] else False
        if newState:
            self.Q[state] = {action: 0.0}
        if (not newState) and newAction:
            self.Q[state][action] = 0.0

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        return self.Q[state][max(self.Q[state], key=self.Q[state].get)]
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        self.addNewKeyValInQ(state, action)
        # qVal = self.Q[state][action]
        return self.Q[state][action]
        # return qVal
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        return self.getRandomAction(state) if state not in self.Q \
            else max(self.Q[state], key=self.Q[state].get)
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)

        if len(all_actions) > 0:
            # *********
            # TODO 3.3.01
            action = np.random.choice(all_actions)
            for action01 in all_actions:
                self.addNewKeyValInQ(state, action01)

            # self.addNewKeyValInQ(state, action)
            return action
            # *********
        else:
            # TODO 3.3.03
            self.addNewKeyValInQ(state, "exit")
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        return self.getRandomAction(state) if np.random.rand() < self.epsilon \
            else self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # ********
        # TODO 3.5.
        self.addNewKeyValInQ(state, action)

        nextAction = self.getPolicy(nextState)
        self.addNewKeyValInQ(nextState, nextAction)
        self.Q[state][action] = self.Q[state][action] + self.learningRate * \
                                (reward + self.discount * self.Q[nextState][nextAction] -
                                 self.Q[state][action])
        # if 'exit' in self.Q[state]:
        #     print(state, self.Q[state])
        #     print(nextState, self.Q[nextState])
        #     self.Q[state][action] = self.Q[state][action] + self.learningRate * \
        #                             (reward - self.Q[state][action])
        # else:
        #     nextAction = self.getPolicy(nextState)
        #     # self.addNewKeyValInQ(nextState, nextAction)
        #     self.Q[state][action] = self.Q[state][action] + self.learningRate * \
        #                             (reward + self.discount * self.Q[nextState][nextAction] -
        #                              self.Q[state][action])
        # *********
