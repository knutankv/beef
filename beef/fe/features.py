from beef import basic_coupled
import numpy as np

class Feature:
    def __init__(self, matrix_type, node_labels, dofs, value, local=False, name=None):
        
        if len(node_labels) == 1 or node_labels[1]==None:
            matrix = basic_coupled()[0:1, 0:1]*value
            node_labels = [node_labels[0]]
            dofs = [dofs[0]]
        else:
            matrix = basic_coupled()*value
        
        self.type = matrix_type
        self.node_labels = node_labels
        self.dofs = dofs
        self.matrix = matrix
        self.local = local
        self.name = name

    # CORE METHODS
    def __str__(self):
        return f'BEEF Feature: {self.name}'

    def __repr__(self):
        return f'BEEF Feature: {self.name}'

class Spring(Feature):
    def __init__(self, node_labels, dofs, k, **kwargs):
        super().__init__('k', node_labels, dofs, k, **kwargs)
        
class Dashpot(Feature):
    def __init__(self, node_labels, dofs, c, **kwargs):
        super().__init__('c', node_labels, dofs, c, **kwargs)
        
class PointMass(Feature):
    def __init__(self, node_label, dofs, m, **kwargs):
        super().__init__('m', node_label, dofs, m, **kwargs)
