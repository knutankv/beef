from . import *

class EngineeringFeature:
    def __init__(self, master_node, dof_ix, feature_type, slave_node=None, name='engfeature-0'):
        self.master_node = master_node
        self.slave_node = slave_node
        self.dof_ix = dof_ix
        self.feature_type = feature_type
        self.name = name

    # CORE METHODS
    def __str__(self):
        return f'BEEF Feature: {self.name}'

    def __repr__(self):
        return f'BEEF Feature: {self.name}'

        
class Spring(EngineeringFeature):
    def __init__(self, k, master_node, dof_ix, slave_node=None, name='spring-0'):
        super().__init__(master_node, dof_ix, 'spring', slave_node=slave_node, name=name)
        self.k = k
        
class Dashpot(EngineeringFeature):
    def __init__(self, c, damping_constants, master_node, dof_ix, slave_node=None, name='dashpot-0'):
        super().__init__(master_node, dof_ix, 'dashpot', slave_node=slave_node, name=name)
        self.c = c
        
class PointMass(EngineeringFeature):
    def __init__(self, m, node, dof_ix='trans', name='mass-0'):
        super().__init__(node, dof_ix, 'dashpot', name=name)
        self.m = m

class AddedMatrix(EngineeringFeature):
    def __init__(self, node, matrix_type='K', name='matrix-0'):
        if matrix_type not in ['K', 'M', 'C']:
            raise ValueError('matrix_type should be "K", "M", or "C".')
            
        super().__init__(node, 'all', 'added_{}'.format(matrix_type), name=name)

class Pontoon(EngineeringFeature):  #group of added matrices
    def __init__(self, node, pontoon_type, orientation=np.eye(3), name='pontoon-0'):
        super().__init__(node, 'all', 'pontoon', name=name)
        self.orientation = orientation
        
    @classmethod
    def assign_multiple(cls, nodes, pontoon_types, orientations=np.eye(3), prefix_label='pontoon-'):
        pontoon_types = ensure_list(pontoon_types)
        orientations = ensure_list(orientations)
        
        if len(pontoon_types) == 1:
            pontoon_types = pontoon_types*len(nodes)
            
        if len(orientations) == 1:
            orientations = orientations*len(nodes)
        
        pontoons = [None]*len(nodes)
        for ix, node in enumerate(nodes):
            label = prefix_label+str(ix+1)
            pontoons[ix] = cls(node, pontoon_types[ix], orientation=orientations[ix], label=label)            
    
        return pontoons
    
class PontoonType:
    def __init__(self, K=None, C=None, M=None, omega=None):
        self.K = K
        self.C = C
        self.M = M
        self.omega = omega