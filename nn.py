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

X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1) # First layer L-combination
s = Sigmoid(l1) # Filter

l2 = Linear(inputs, weights, bias)	# Sec Layer combination
cost = MSE(l2, y)


feed_dict = {inputs: ip, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

#print(graph)
print(output)



