import numpy as np

#%% Constraint class definition
class Constraint:
    def __init__(self, master_nodes, slave_nodes=None, name='constraint-0', dofs='all', relative_to='global'):
        self.name = name       
        dofs = convert_dofs_list(dofs, len(master_nodes))

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
    def __init__(self, master_node, dof_ix, slave_node, relative_to):
        self.slave_node = slave_node
        self.master_node = master_node
        self.dof_ix = dof_ix
        self.relative_to = relative_to
  