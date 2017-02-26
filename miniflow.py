import numpy as np

class Node:
	"""docstring for Node"""
	def __init__(self, inbound_nodes=[]):
		#  Nodes from which this node receives values

		self.inbound_nodes = inbound_nodes
		#  Nodes to which this node passes value

		self.outbound_nodes = []
		#  for each inbound node here, add this node as an outbound to _that_

		self.value = None
		#  Add this node as an outbound node on it's inputs.

		self.gradients = {}

		for n in inbound_nodes:
			n.outbound_nodes.append(self)

	def forward(self):
        #  Forward propagation.
        #  Compute the output value based on `inbound_nodes` and
        #  store the result in self.value.

		raise NotImplemented


	def backward(self):
		raise NotImplemented


class Input(Node):
	def __init__(self):
		#  Input node has no inbound nodes,
		#  so no need to pass anythig to the Node instantiator
		Node.__init__(self)
	#  NOTE: Input node is the only node that may receive its 
	#  values as an argument to forward()
	#  ALL other node implementations should calculate their
	#  values from the value of previous nodes, using
	#  self.inbound_nodes
	#  val = self.inbound_nodes[0].value

	def forward(self):
		#  Overwrite the value if one is passed in.
		pass


	def backward(self):
		#  An input node has no inputs so the gradient is 0
		#  The key, 'self', is reference to this object.
		self.gradients = {self: 0}

		#  Weights and bias may be inputs, so need to sum
		#  the gradient from output gradients.
		for n in self.outbound_nodes:
			self.gradients[self] += n.gradients[self]



class Linear(Node):	#  Perform a caculation
	def __init__(self, inputs, weights, bias):
		Node.__init__(self, [inputs, weights, bias])
		#  You should access 'x' and 'y' in forward with
		#  self.inbound_nodes[0]('x') and self.inbound_nodes[1] ('y')

	def forward(self):
		inputs = self.inbound_nodes[0].value
		weights = self.inbound_nodes[1].value
		bias = self.inbound_nodes[2].value
		self.value = np.dot(inputs, weights) + bias

	def backward(self):
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
		#  Cycle through the outputs. The gradient will change depending
		#  on each output, so the gradients are summed over all outputs.
		for n in self.outbound_nodes:
			grad_cost = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)

			self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)

			self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims = False)

class Sigmoid(Node):
	"""docstring for Sigmoid"""
	def __init__(self, node):
		Node.__init__(self, [node])


	def _sigmoid(self, x):
		return 1. / (1. + np.exp(-x))


	def forward(self):
		input_value = self.inbound_nodes[0].value
		self.value = self._sigmoid(input_value)


	def backward(self):
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

		for n in self.outbound_nodes:
			grad_cost = n.gradients[self]
			sigmoid = self.value
			self.gradients[self.inbound_nodes[0]] += sigmoid* (1 - sigmoid) * grad_cost



class MSE(Node):
	"""docstring for MSE"""
	def __init__(self, y, a):
		Node.__init__(self,[y, a])

	def forward(self):
		#  Calculate the mean squared error
		y = self.inbound_nodes[0].value.reshape(-1, 1)
		a = self.inbound_nodes[1].value.reshape(-1, 1)

		self.m = self.inbound_nodes[0].value.shape[0]
		self.diff = y - a
		self.value = np.mean(self.diff**2)

	def backward(self):
		self.gradients[self.inbound_nodes[0]] = (2/self.m) * self.diff
		self.gradients[self.inbound_nodes[1]] = (-2/self.m) * self.diff


def topological_sort(feed_dict):
	'''
	Sore generic nodes in topological order using Kahn's algorithm.

	feed_dict: A dictionary where the key is a input node and value is the
	respective value feed to that node.

	Return a list of sorted nodes.
	'''

	input_nodes = [n for n in feed_dict.keys()]

	G = {}
	nodes = [ n for n in input_nodes]
	while len(nodes) > 0 :
		n = nodes.pop(0)
		if n not in G:
			G[n] = {'in': set(), 'out': set()}

		for m in n.outbound_nodes:
			if m not in G:
				G[m] = {'in': set(), 'out': set()}
			G[n]['out'].add(m)
			G[m]['in'].add(n)
			nodes.append(m)

	L =[]
	S = set(input_nodes)
	while len(S) > 0:
		n = S.pop()

		if isinstance(n, Input):
			n.value = feed_dict[n]

		L.append(n)
		for m in n.outbound_nodes:
			G[n]['out'].remove(m)
			G[m]['in'].remove(n)
			#  if no other incoming edges add to S
			if len(G[m]['in']) == 0:
				S.add(m)
	return L


def forward_pass(output_node, sorted_nodes):
	'''
	Performance a forward pass through a list of sorted nodes.
	Arguments:
		'output_node': A node in the graph, should be the output node
											(have no outgoing edges).
		'sorted_nodes': A topologically sorted list of nodes.
	Return the output Nodes value
	'''
	for n in sorted_nodes:
		n.forward()

	return output_node.value



def forward_and_backward(graph):
	
	#  Forward pass
	for n in graph:
		n.forward()

	#  Backward pass
	for n in graph[::-1]:
		n.backward()


def sgd_update(trainables, learning_rate = 0.01):
	for t in trainables:

		partial = t.gradients[t]
		t.value -= learning_rate * partial












































		