'''
FE objects submodule: engineering features definitions
'''

from beef import basic_coupled
import numpy as np
 

class Feature:
    '''
    Feature definition class.

    Arguments
    ------------
    node_labels : int
        list of node labels to connect with feature (if length is 1, the value is added directly in the specified
        global dof of the relevant matrix)
    dofs : int
        list of DOFs to connect with feature
    value : float
        strength/amplitude of feature (spring stiffness, mass, damper constant)
    local : False
        whether the feature is applied wrt. a local csys or global (only global supported currently)
    name : None, optional
        name of feature
    tmat : None
        transformation matrix used prior to insertion in global matrices

    Notes
    -----------
    The submatrix of [K] that describes the two DOFs connected by the spring, denoted i and j
    is established as follows:
    $$[K]_{i,j} = k\cdot\\begin{bmatrix}1 & -1 \\\ -1 & 1\\end{bmatrix}$$
    
    where k is the spring stiffness. The same procedure is valid for damping and mass. If a single node/DOF is input,
    the resulting addition to the global system matrix will be scalar:
    $$[K]_{i,i} = k $$
    
    '''
    def __init__(self, feature_type, node_labels, dofs, value, local=False, name=None, tmat=None):

        if len(node_labels) == 1 or node_labels[1]==None:
            matrix = basic_coupled()[0:1, 0:1]*value
            node_labels = [node_labels[0]]
            dofs = [dofs[0]]
        else:
            matrix = basic_coupled()*value

        if local==True:
            raise ValueError('Only global CSYS features supported currently.')

        self.node_labels = node_labels
        self.dofs = dofs
        self.matrix = matrix
        self.local = local
        self.name = name
        self.type = feature_type
        self.tmat = tmat


    # CORE METHODS
    def __str__(self):
        return f'BEEF Feature: {self.name}'

    def __repr__(self):
        return f'BEEF Feature: {self.name}'
    
    @property
    def tmat(self):
        if self._tmat is None:
            return np.eye(self.matrix.shape[0])
        else:
            return self._tmat
        
    @tmat.setter
    def tmat(self, val):
        self._tmat = val


    @property
    def global_matrix(self):
        return self.tmat.T @ self.matrix @ self.tmat

class CustomMatrix(Feature):
    def __init__(self, feature_type, node_labels, matrix, dofs=None, name=None, tmat=None):
        if np.ndim(node_labels)==0:
            node_labels = [node_labels]

        if dofs is None:
            self.dofs = np.arange(0, int(matrix.shape[0]/len(node_labels)))
        else:
            self.dofs = np.array(dofs)

        if len(self.dofs) != int(matrix.shape[0]/len(node_labels)):
            raise ValueError('Size of dofs is not compatible with matrix size and specified node labels')

        self.node_labels = node_labels
        self.matrix = matrix
        self.name = name
        self.type = feature_type  
        self.tmat = tmat


class Spring(Feature):
    def __init__(self, node_labels, dofs, k, **kwargs):
        super().__init__('k', node_labels, dofs, k, **kwargs)
        
class Dashpot(Feature):
    def __init__(self, node_labels, dofs, c, **kwargs):
        super().__init__('c', node_labels, dofs, c, **kwargs)
        
class PointMass(Feature):
    def __init__(self, node_label, dofs, m, **kwargs):
        super().__init__('m', node_label, dofs, m, **kwargs)
