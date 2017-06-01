import copy
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from visual  import *

SARS = namedtuple('SARS', ['state', 'action', 'reward', 'next_state'])

# Replay Memory

class ReplayMemory(object):
    """Replay Memory"""

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


class SimpleAgent(object):
    """SimpleAgent"""

    def __init__(self, state_space, action_space, initial_eps, end_eps, eps_decay):
        super(SimpleAgent, self).__init__()
        self.num_state = state_space.shape[0]
        self.num_action = action_space.n
        self.action_space = action_space

        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay
        self.eps_count = 0

        self.dqn = SimpleDQN(self.num_state, self.num_action, 128)
        self.target_dqn = copy.deepcopy(self.dqn)
        if USE_CUDA:
            self.dqn = self.dqn.cuda()
            self.target_dqn = self.target_dqn.cuda()

    @flatten_single_value
    def greedy_act(self, state):
        # input: state, numpy array, size = (num_state,)
        # output: action, numpy array, size = (1,)
        state = Variable((torch.from_numpy(state)).unsqueeze(0).type(dtype))
        output = self.dqn(state)
        _, action = torch.max(output, 1)
        action = action.squeeze(0)
        return action.data.cpu().numpy()

    @flatten_single_value
    def boltzmann_act(self, state):
        # input: state, numpy array, size = (num_state,)
        # output: action, list, size = (1,)
        state = Variable((torch.from_numpy(state)).unsqueeze(0).type(dtype))
        output = self.dqn(state)
        output = F.softmax(output)
        output = output.squeeze(0)
        output = output.data.cpu().numpy()
        assert self.action_space.n == output.shape[0], 'action shape: {}, output shape: {}'.format(
            self.action_space.n, output.shape)
        actions = np.arange(self.num_action)
        return random.choices(actions, weights=output)

    def e_greedy_act(self, state):
        # input: state, numpy array, size = (num_state,)
        # output: action, scalar value
        self.eps = self.end_eps + \
            (self.initial_eps - self.end_eps) * \
            np.exp(-self.eps_count/self.eps_decay)
        self.eps_count += 1
        if np.random.random() > self.eps:
            return self.greedy_act(state)
        else:
            return self.action_space.sample()


class DQN_Optimizer(object):
    """An optimizer of training DQN"""

    def __init__(self, agent, gamma, capacity, batch_size, initial_lr):
        super(DQN_Optimizer, self).__init__()
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
        if USE_CUDA:
            mask = mask.cuda()
        outputs = torch.masked_select(outputs, mask).view(-1,1)

        no_final_states = np.array([ns for ns in next_states if ns is not None])
        no_final_targets = self.agent.target_dqn(
            Variable(torch.from_numpy(no_final_states).type(dtype), volatile=True))
        no_final_targets, _ = torch.max(no_final_targets, 1)
        targets = Variable(torch.zeros(self.memory.batch_size, 1))
        mask = Variable(torch.ByteTensor([ns is not None for ns in next_states]).view(-1,1))
        if USE_CUDA:
            mask = mask.cuda()
            targets = targets.cuda()
        targets.masked_copy_(mask, no_final_targets)
        targets = self.gamma * targets + Variable((torch.from_numpy(rewards)
                                                   ).unsqueeze(1).type(dtype))
        targets = targets.detach()

        loss = self.criterion(outputs, targets)

        # dqn_visualizer = pytorch_net_visualizer(loss)
        # dqn_visualizer.view()
        # input("Visualizing DQN...")

        loss.backward()

        # gradient clamping in case of gradient explosion
        for param in self.agent.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        if self.agent.eps_count % 128 == 0:
            self.agent.target_dqn = copy.deepcopy(self.agent.dqn)
            if USE_CUDA:
                self.agent.target_dqn = self.agent.target_dqn.cuda()

        return loss.data[0]
