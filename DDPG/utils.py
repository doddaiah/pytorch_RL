import numpy as np
import torch
from gym import ActionWrapper
from gym.spaces.box import Box

OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2
OU_DT = 1e-2

USE_CUDA = torch.cuda.is_available()
# dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def Normalize_Env(env):

    class Normalized_Env(ActionWrapper):
        """docstring for Normalized_Env"""
        def __init__(self, env):
            super(Normalized_Env, self).__init__(env)
            assert(isinstance(env.action_space, Box))
            self.actions_lb = env.action_space.low
            self.actions_up = env.action_space.high
            self.actions_range = self.actions_up - self.actions_lb

            print("Normalize {} action space to [-1,1]".format(str(env)))

        def _action(self, action):
            # action are [-1, 1]
            action = (self.actions_lb + self.actions_up + self.actions_range * action) / 2
            return action
            
        def _reverse_action(self, action):
            # action are [-a, a]
            action = 2 * np.divide((action - self.actions_lb), self.actions_range) - 1
            return action
            
    normalized_env = Normalized_Env(env)
            
    return normalized_env

class Ornstein_Uhlenbeck_Process(object):
    """docstring for Ornstein_Uhlenbeck_Process"""

    def __init__(self, num_dim, theta=OU_THETA, mu=OU_MU, sigma=OU_SIGMA, dt=OU_DT):
        super(Ornstein_Uhlenbeck_Process, self).__init__()
        self.num_dim = num_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x = np.zeros(shape=(self.num_dim))

    def next(self):
        dx = self.theta*(self.mu-self.x) * self.dt + self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.num_dim)
        self.x = self.x + dx
        return self.x

    def reset(self):
        self.x = np.zeros(shape=(self.num_dim))

class Variable(torch.autograd.Variable):
    """docstring for Variable"""

    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)
        if USE_CUDA:
            self.data = self.data.cuda()


if __name__ == '__main__':
    ou = Ornstein_Uhlenbeck_Process(1)

    states = []
    for _ in range(100):
        states.append(ou.next())
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print(states)
    plt.plot(np.asarray(states))
    plt.show()
