import numpy as np

class Node(object):
	"""docstring for Node"""
	def __init__(self, inbound_nodes=[]):
		#  Nodes from which this node receives values

		self.inbound_nodes = inbound_nodes
		#  Nodes to which this node passes value

		self.outbound_nodes = []
		#  for each inbound node here, add this node as an outbound to _that_

		self.value = None
		#  Add this node as an outbound node on it's inputs.

		for n in self.inbound_nodes:
			n.outbound_nodes.append(self)
		
	def forward(self):
        #  Forward propagation.
        #  Compute the output value based on `inbound_nodes` and
        #  store the result in self.value.

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

	def forward(self, value = None):
		#  Overwrite the value if one is passed in.
		if value is not None:
			self.value = value

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
		'''
		self.value = bias
		for x, w, in zip(inputs, weights):
			self.value += x * w
		'''



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















































		