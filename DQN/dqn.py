import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from visual  import *

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

SARS = namedtuple('SARS', ['state', 'action', 'reward', 'next_state'])

# Replay Memory


class ReplayMemory(object):
    """Repaly Memory"""

    def __init__(self, capacity, batch_size):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque()

    def __len__(self):
        return len(self.memory)

    def push_back(self, sars):
        self.memory.append(sars)
        while len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self):
        return random.choices(self.memory, k=self.batch_size)

    @property
    def trainable(self):
        return len(self.memory) > self.batch_size


# Simple DQN


class SimpleDQN(nn.Module):
    """Simple Deep Q-network for vanilla state input"""

    def __init__(self, num_state, num_action, num_hidden_units, num_hidden_layers=1, activation=nn.ReLU()):
        super(SimpleDQN, self).__init__()
        self.input_layer = nn.Linear(num_state, num_hidden_units)
        self.fc = nn.Linear(num_hidden_units, num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, num_action)
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

    def forward(self, state):
        # state: batch_size * num_state,
        # output: batch_size * num_action
        output = self.activation(self.input_layer(state))
        for _ in range(self.num_hidden_layers):
            output = self.activation(self.fc(output))
        output = self.output_layer(output)
        return output


class Agent(object):
    """Agent"""

    def __init__(self, num_state, actions, initial_eps, end_eps, eps_decay):
        super(Agent, self).__init__()
        self.num_state = num_state
        self.num_action = len(actions)
        self.actions = actions

        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay
        self.eps_count = 0

        self.dqn = SimpleDQN(self.num_state, self.num_action, 20)

    def greedy_act(self, state):
        # input: state, numpy array, size = (num_state,)
        # output: action, numpy array, size = (1,)
        state = Variable((torch.from_numpy(state)).unsqueeze(0).type(dtype))
        output = self.dqn(state)
        _, action = torch.max(output, 1)
        action = action.squeeze(0)
        return action.data.numpy()

    @flatten_single_value
    def boltzmann_act(self, state):
        # input: state, numpy array, size = (num_state,)
        # output: action, list, size = (1,)
        state = Variable((torch.from_numpy(state)).unsqueeze(0).type(dtype))
        output = self.dqn(state)
        output = F.softmax(output)
        output = output.squeeze(0)
        output = output.data.numpy()
        assert self.actions.shape == output.shape, 'action shape: {}, output shape: {}'.format(
            self.actions.shape, output.shape)
        return random.choices(self.actions, weights=output)

    @flatten_single_value
    def e_greedy_act(self, state):
        # input: state, numpy array, size = (num_state,)
        # output: action, list, size = (1,)
        eps = self.end_eps + \
            (self.initial_eps - self.end_eps) * \
            np.exp(-self.eps_count/self.eps_decay)
        self.eps_count += 1
        if np.random.random() > eps:
            return self.greedy_act(state)
        else:
            return random.choices(self.actions)


class DQN_optimizer(object):
    """An optimizer of training DQN"""

    def __init__(self, agent, gamma, capacity, batch_size, initial_lr):
        super(DQN_optimizer, self).__init__()
        self.agent = agent
        self.gamma = gamma
        self.memory = ReplayMemory(capacity, batch_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.agent.dqn.parameters(), lr=initial_lr)

    def step(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states = get_attr(
            samples, ['state', 'action', 'reward', 'next_state'])

        self.optimizer.zero_grad()

        outputs = self.agent.dqn(
            Variable(torch.from_numpy(states)).type(dtype))
        mask = Variable(make_one_hot(actions, self.agent.num_action))
        outputs = torch.masked_select(outputs, mask).view(-1,1)

        no_final_states = np.array([ns for ns in next_states if ns is not None])
        no_final_targets = self.agent.dqn(
            Variable(torch.from_numpy(no_final_states).type(dtype), volatile=True))
        no_final_targets, _ = torch.max(no_final_targets, 1)
        targets = Variable(torch.zeros(self.memory.batch_size, 1))
        mask = Variable(torch.ByteTensor([ns is not None for ns in next_states]).view(-1,1))
        targets.masked_copy_(mask, no_final_targets)

        targets = self.gamma * targets + Variable((torch.from_numpy(rewards)
                                                   ).unsqueeze(1).type(dtype))
        targets = targets.detach()

        loss = self.criterion(outputs, targets)

        # dqn_visualizer = pytorch_net_visualizer(loss)
        # dqn_visualizer.view()
        # input("Visualizing DQN...")

        loss.backward()
        
        self.optimizer.step()

        return loss.data[0]