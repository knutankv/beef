from beef import basic_coupled

class Feature:
    def __init__(self, matrix_type, node_labels, dof_ixs, value, local=False, name=None):
        
        if len(node_labels) == 1 or node_labels[1]==None:
            matrix = basic_coupled()[0:1, 0:1]*value
            node_labels = [node_labels[0]]
            dof_ixs = [dof_ixs[0]]
        else:
            matrix = basic_coupled()*value
        
        self.type = matrix_type
        self.node_labels = node_labels
        self.dof_ixs = dof_ixs
        self.matrix = matrix
        self.local = local
        self.name = name

    # CORE METHODS
    def __str__(self):
        return f'BEEF Feature {self.name}'

    def __repr__(self):
        return f'BEEF Feature {self.name}'

class Spring(Feature):
    def __init__(self, node_labels, dof_ixs, k):
        super().__init__('k', node_labels, dof_ixs, k)
        
class Dashpot(Feature):
    def __init__(self, node_labels, dof_ixs, c):
        super().__init__('c', node_labels, dof_ixs, c)
        
class PointMass(Feature):
    def __init__(self, node_label, dof_ixs, m):
        super().__init__('m', node_label, dof_ixs, m)


# class Pontoon(Feature):  #group of added matrices
#     def __init__(self, node, pontoon_type, orientation=np.eye(3), name='pontoon-0'):
#         super().__init__(node, 'all', 'pontoon', name=name)
#         self.orientation = orientation
        
#     @classmethod
#     def assign_multiple(cls, nodes, pontoon_types, orientations=np.eye(3), prefix_label='pontoon-'):
#         pontoon_types = ensure_list(pontoon_types)
#         orientations = ensure_list(orientations)
        
#         if len(pontoon_types) == 1:
#             pontoon_types = pontoon_types*len(nodes)
            
#         if len(orientations) == 1:
#             orientations = orientations*len(nodes)
        
#         pontoons = [None]*len(nodes)
#         for ix, node in enumerate(nodes):
#             label = prefix_label+str(ix+1)
#             pontoons[ix] = cls(node, pontoon_types[ix], orientation=orientations[ix], label=label)            
    
#         return pontoons
    
# class PontoonType:
#     def __init__(self, K=None, C=None, M=None, omega=None):
#         self.K = K
#         self.C = C
#         self.M = M
#         self.omega = omega