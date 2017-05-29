import numpy as np
import torch


def action_space_wrapper(gym_action_space):
    return np.array([x for x in range(gym_action_space.n)], dtype=np.int)


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
