import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def flatten_single_value(func):
        # flatter a single value list/np.array to single value
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        return value[0]
    return wrapper


def get_attr(samples, names):
        # return numpy array of samples' name field
    return [np.array([getattr(sample, name) for sample in samples]) for name in names]


def make_one_hot(indices, depth):
    tensor = torch.ByteTensor(len(indices), depth).zero_()
    for ii, index in enumerate(indices):
        tensor[ii, index] = 1
    return tensor


class Variable(torch.autograd.Variable):
    """A simple wrapper that will automatically send data to GPU every time
    we construct a variable"""

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, ** kwargs)
