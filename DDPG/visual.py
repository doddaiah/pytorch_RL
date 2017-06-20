import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
else:
    matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
import numpy as np

import torch
from graphviz import Digraph
from torch.autograd import Variable


class Visualizer(object):
    """Visualizer for statistics, e.g. loss"""

    def __init__(self):
        super(Visualizer, self).__init__()
        plt.ion()
        self.bundle = {}
        self.fig = plt.figure(figsize=(6, 12))

    def __del__(self):
        plt.ioff()

    def register(self, title, xlabel, ylabel):
        self.bundle[title] = {'data': np.zeros(
            shape=(0,)), 'title': title, 'xlabel': xlabel, 'ylabel': ylabel}

    def draw(self):
        self.fig.clf()
        num_ax = len(self.bundle)
        for ax, name in enumerate(self.bundle):
            plt.subplot(num_ax, 1, ax+1)
            plt.plot(self.bundle[name]['data'])
            plt.title(self.bundle[name]['title'])
            plt.xlabel(self.bundle[name]['xlabel'])
            plt.ylabel(self.bundle[name]['ylabel'])
        plt.tight_layout()
        self.fig.canvas.draw()
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def push_back(self, **kwargs):
        for name, data in kwargs.items():
            self.bundle[name]['data'] = np.append(
                self.bundle[name]['data'], data)
        self.draw()

    def save(self):
        self.fig.savefig('Visualizer.png')


def pytorch_net_visualizer(var):
    '''
    outputs = model(Variable(inputs))
    g = pytorch_net_visualizer(outputs)
    g.view()
    '''
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d' % v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot
