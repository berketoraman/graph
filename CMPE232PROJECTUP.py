import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import matplotlib.pyplot as plt
class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self
class Operation:
    """Represents a graph node that performs a computation.

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=[]):
        """Construct Operation
        """
        self.input_nodes = input_nodes

        self.consumers = []

        for input_node in input_nodes:
            input_node.consumers.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass        
class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
       
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        return x_value + y_value
class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        return a_value.dot(b_value)
class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """
        return x_value * y_value    
class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
       
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))    
class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
       
        self.consumers = []
      

        _default_graph.placeholders.append(self)
class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        
        self.value = initial_value
        self.consumers = []

        _default_graph.variables.append(self)

import numpy as np
class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):
        """Computes the output of an operation

        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if type(node) == placeholder:
               
                node.output = feed_dict[node]
            elif type(node) == Variable:
              
                node.output = node.value
            else: 
                node.inputs = [input_node.output for input_node in node.input_nodes]

                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)
 
        return operation.output


def traverse_postorder(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
class node():
    def __init__(self, name: str, value=0):
        self.name = name
        self.neighbors = [] 
        self.value=value
        
    def neighbors_name(self):
        """
        info about neighbors names
        """
        return [node_s.name for node_s in self.neighbors]
class digraph():
    def __init__(self, elist):
        """
            self.nodes is a dictionary
                key   : node name
                value : node class
        """
        self.elist = elist
        self.node_names = list(set([s for s,t in elist] + [t for s,t in elist]))
        self.nodes = {s:node(s) for s in self.node_names}
        
        self.create_graph()
      
    def add_edge(self, s,t):
        """directed Edge"""
        self.nodes[s].neighbors.append(self.nodes[t])
    
    def create_graph(self):
        for s,t in self.elist:
             self.add_edge(s,t)
                
    def info(self):
        return {s:node_s.neighbors_name() for s,node_s in self.nodes.items()}
    def draw(self, color = 'lightblue'):
        G = nx.DiGraph()
        G.add_edges_from(self.elist)
        plt.figure(figsize=(10,5))
        nx.draw(G, node_size=4500, node_color=color, with_labels=True)
    def reverse(self):
        reversed_elist = [(t,s) for s, t in  self.elist]
        return digraph(reversed_elist)
    
    def add_values(self,list_of_values):
        for node_name,node_value in list_of_values:
            self.nodes[node_name].value=node_value

Graph().as_default()
"""creating placeholders"""
a = placeholder() 
b = placeholder() 
c = placeholder() 
d = Variable(3)
""" creating the equation
j=d(a+(b*c))"""
u = multiply(b,c)
v = add(a,u)
j = multiply(v,d)

session = Session()
output = session.run(j, {
    a: (5),b: (3),c: (2)
})
print("The equation is equal to = ",output)
elist = [('a','a+(bc)'), ('b', 'b*c'),('c', 'b*c'),('b*c', 'a+(bc)'),('a+(bc)', '(d(a+(bc)))')]
G = digraph(elist)
G.draw()
G.reverse()
print("Information = ",G.info())
 
