import copy
import random
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pycrayon import CrayonClient

from utils import Ornstein_Uhlenbeck_Process, Clip_Action_Values, Variable, USE_CUDA #, dtype
from visual import pytorch_net_visualizer

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


class Actor(nn.Module):
    """docstring for actor"""

    def __init__(self, num_states, num_actions, actions_bound, num_hidden_units, num_hidden_layers=1, activation=nn.ReLU()):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(num_states, num_hidden_units)
        self.fc = nn.Linear(num_hidden_units, num_hidden_units)
        self.bn = nn.BatchNorm1d(num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, num_actions)
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.actions_bound = actions_bound

    def forward(self, state):
        output = self.activation(self.bn(self.input_layer(state)))
        for _ in range(self.num_hidden_layers):
            output = self.activation(self.bn(self.fc(output)))
        output = F.tanh(self.output_layer(output))
        output = torch.mul(output, self.actions_bound)
        return output


class Critic(nn.Module):
    """docstring for Critic"""

    def __init__(self, num_states, num_actions, num_hidden_units, num_hidden_layers=1, activation=nn.ReLU()):
        super(Critic, self).__init__()
        self.state_input_layer = nn.Linear(num_states, num_hidden_units)
        self.cat_fc = nn.Linear(num_hidden_units+num_actions, num_hidden_units)
        self.fc = nn.Linear(num_hidden_units, num_hidden_units)
        self.bn = nn.BatchNorm1d(num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, 1)
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

    def forward(self, state, action):
        # input: state + action (both are continuous)
        state_input = self.activation(self.bn(self.state_input_layer(state)))

        output = torch.cat((state_input, action), 1)
        output = self.activation(self.bn(self.cat_fc(output)))
        for _ in range(self.num_hidden_layers):
            output = self.activation(self.bn(self.fc(output)))
        output = self.output_layer(output)
        return output


class DDPGAgent(object):
    """docstring for DDPGAgent"""

    def __init__(self, num_states, num_actions, actions_bound):
        super(DDPGAgent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.actor = Actor(num_states, num_actions, actions_bound, 128)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(num_states, num_actions, 128, 2)
        self.critic_target = copy.deepcopy(self.critic)

        self.ou_noise = Ornstein_Uhlenbeck_Process(num_actions)

        if USE_CUDA:
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

    @Clip_Action_Values
    def noisy_act(self, state):
        action = self.act(state)
        action = action + self.ou_noise.next()
        return action

    def act(self, state):
        # No need to clip action values here since tanh will do the job
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        output = self.actor(state)
        action = output.squeeze(0).data.cpu().numpy()
        return action


class DDPGOptimizer(object):
    """docstring for DDPGOptimizer"""

    def __init__(self, agent, capacity, batch_size, gamma, tau, init_lr, weight_decay, crayon_vis):
        super(DDPGOptimizer, self).__init__()
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayMemory(capacity, batch_size)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.agent.critic.parameters(), lr=init_lr[
                                           'critic'], weight_decay=weight_decay)
        self.actor_optimizer = optim.Adam(
            self.agent.actor.parameters(), lr=init_lr['actor'])

        self.crayon_vis = crayon_vis
        if self.crayon_vis:
            self.cc = CrayonClient()
            try:
                self.stats = self.cc.create_experiment('stats')
            except ValueError:
                self.cc.remove_experiment('stats')
                self.stats = self.cc.create_experiment('stats')

    def step(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states = map(
            lambda x: np.asarray(x), zip(*samples))

        # Update critic network
        self.critic_optimizer.zero_grad()

        outputs = self.agent.critic(Variable(torch.from_numpy(
            states)), Variable(torch.from_numpy(actions)))

        no_final_states = np.array(
            [ns for ns in next_states if ns is not None])
        no_final_targets = self.agent.critic_target(Variable(torch.from_numpy(no_final_states), volatile=True),
                                                    self.agent.actor_target(Variable(torch.from_numpy(no_final_states), volatile=True)))
        targets = Variable(torch.zeros(self.memory.batch_size, 1))
        mask = Variable(torch.ByteTensor(
            [ns is not None for ns in next_states]).view(-1, 1))
        targets.masked_copy_(mask, no_final_targets)
        targets = self.gamma * targets + \
            Variable(torch.from_numpy(rewards).unsqueeze(1))
        targets = targets.detach()

        loss = self.critic_criterion(outputs, targets)

        if self.crayon_vis:
            self.stats.add_scalar_value('critic loss', loss.data[0])

        loss.backward()

        # critic_visualizer = pytorch_net_visualizer(loss)
        # critic_visualizer.view()
        # input("Visualizing critic networks...")

        '''
        # gradient clamping in case of gradient explosion
        for param in self.agent.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        '''

        self.critic_optimizer.step()

        # Update actor network
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        outputs = self.agent.critic(Variable(torch.from_numpy(states), requires_grad=False),
                                    self.agent.actor(Variable(torch.from_numpy(states), requires_grad=True)))
        outputs = -torch.mean(outputs) # negation since we want the police increase the likelihood of good reward trajectory
        outputs.backward()

        # actor_visualizer = pytorch_net_visualizer(outputs)
        # actor_visualizer.view()
        # input("Visualizing actor networks...")

        self.actor_optimizer.step()

        # Update target network
        for param, param_target in zip(self.agent.critic.parameters(), self.agent.critic_target.parameters()):
            param_target = self.tau * param + (1 - self.tau) * param_target

        for param, param_target in zip(self.agent.actor.parameters(), self.agent.actor_target.parameters()):
            param_target = self.tau * param + (1 - self.tau) * param_target

        return loss.data[0]
