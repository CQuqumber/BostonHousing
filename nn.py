"""
This script builds and runs a graph with miniflow.

(x + y) + y
"""

from miniflow import *
import numpy as np

inputs, weights, bias  = Input(), Input(), Input()

#  f = Add(x,y)
#  feed_dict = {x: 10, y: 5}
#  sorted_nodes = topological_sort(feed_dict)
#  output = forward_pass(f, sorted_nodes)

f = Linear(inputs, weights, bias)
g = Sigmoid(f)

ip = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: ip, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

#print(graph)
print(output)



