'''
FE objects submodule: constraint definitions
'''

import numpy as np
from ..general import convert_dofs_list

#%% Constraint class definition
class Constraint:
    '''
    Constraint definition class.

    Arguments
    -----------------
    master_nodes : int
        list of master node labels
    slave_nodes : None, int
        list of slave node labels (if None, master nodes are constrained to ground)
    name : string
        name of constraint
    dofs : 'all'
        either specify strings 'all', 'rot' or 'trans' or a list with dof indices (0 is first)
    node_type : {'beam2d', 'beam3d'}
        type of nodes (defining number of DOFs per node)
    relative_to : 'global'
        only 'global' supported currently

    '''

    def __init__(self, master_nodes, slave_nodes=None, name='Untitled constraint', 
                 dofs='all', node_type='beam3d', relative_to='global'):
        self.name = name       
        dofs = convert_dofs_list(dofs, len(master_nodes), node_type=node_type)

        if slave_nodes is None:
            self.type = 'node-to-ground'
        else:
            self.type = 'node-to-node'
            
        self.node_constraints = [None]*len(master_nodes)
        self.relative_to = relative_to
        
        if self.relative_to != 'global':
            raise ValueError("Only 'global' constraints supported currently, specified with variable 'relative_to'")
        
        for ix, master_node in enumerate(master_nodes):
            if self.type == 'node-to-ground':
                self.node_constraints[ix] = NodeConstraint(master_node, dofs[ix], None, relative_to)
            else:
                self.node_constraints[ix] = NodeConstraint(master_node, dofs[ix], slave_nodes[ix], relative_to)

    # CORE METHODS
    def __str__(self):
        return f'BEEF Constraint: {self.name}'

    def __repr__(self):
        return f'BEEF Constraint: {self.name}'


class NodeConstraint:
    '''
    Core class for a single DOF-to-DOF constraint, used to setup Constraint.

    Arguments
    -----------------
    master_node : int
        master node label
    dof_ix : int
        list of indices of the two dofs [master, slave] to constrain
    slave_node : int
        slave node label
    relative_to : str
        only 'global' supported so far 
    '''    

    def __init__(self, master_node, dof_ix, slave_node, relative_to):
        self.slave_node = slave_node
        self.master_node = master_node
        self.dof_ix = dof_ix
        self.relative_to = relative_to
  