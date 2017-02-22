"""
This script builds and runs a graph with miniflow.

(x + y) + y
"""

from miniflow import *

x, y = Input(), Input()

f = Add(x,y)

free_dict = {x: 10, y: 5}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)


# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (via miniflow)".format(feed_dict[x], feed_dict[y], output))