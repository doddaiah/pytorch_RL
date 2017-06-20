import numpy as np

OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2


class Ornstein_Uhlenbeck_Process(object):
    """docstring for Ornstein_Uhlenbeck_Process"""

    def __init__(self, num_dim, theta=OU_THETA, mu=OU_MU, sigma=OU_SIGMA):
        super(Ornstein_Uhlenbeck_Process, self).__init__()
        self.num_dim = num_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x = np.zeros(shape=(self.num_dim))

    def next(self):
        dx = self.theta*(self.mu-self.x) + self.sigma * \
            np.random.normal(size=self.num_dim)
        self.x = self.x + dx
        return self.x

    def reset(self):
        self.x = np.zeros(shape=(self.num_dim))


def Clip_Action_Values(func):
    # Clip action values so that they are within reasonbale range
    def wrapper(*args, **kwargs):
        value = np.clip(func(*args, **kwargs), -1, 1)
        return value
    return wrapper

if __name__ == '__main__':
    ou = Ornstein_Uhlenbeck_Process(1)

    states = []
    for _ in range(100):
        states.append(ou.next)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print(states)
    plt.plot(np.asarray(states))
    plt.show()
