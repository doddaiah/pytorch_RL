import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
import numpy as np

from graphviz import Digraph
from torch.autograd import Variable

class Visualizer(object):
    """Visualizer for statistics, e.g. loss"""

    def __init__(self):
        super(Visualizer, self).__init__()
        self.fig = plt.figure(figsize=(6, 6))
        self.data = np.zeros(shape=(0,))

    def draw(self):
        self.fig.clf()
        plt.title("Training...")
        plt.plot(self.data)
        plt.draw()

    def push_back(self, x):
        self.data = np.append(self.data, x)
        self.draw()

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
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
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