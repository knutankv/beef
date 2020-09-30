import functools
import numpy as np

@functools.total_ordering
class Node:
    def __init__(self, label, coordinates, ndofs=None, global_dofs=None):
        self.label = int(label)
        self.coordinates = np.array(coordinates)
        self.ndofs = ndofs                #number of dofs, normally defined later
        self.global_dofs = global_dofs    #global dofs, normally defined later

        # Defined during element initialization
        self.x0 = None
        self.x = None
        self.rots = None
        self.u = None 

    # CORE METHODS
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.label == other.label
        elif isinstance(other, int):
            return self.label == other
            
    def __lt__(self, other):
        if isinstance(other, Node):
            return self.label < other.label
        elif isinstance(other, int):
            return self.label < other

    def __repr__(self):
        return f'Node {self.label}'

    def __str__(self):
        return f'Node {self.label}'

    def __hash__(self):
        return hash(self.label)