'''
FE objects submodule: nodes
'''

import functools
import numpy as np
from .. import rotation

@functools.total_ordering

class Node:
    '''
    Node core class. Basic functionality (common for all children objects)
    will inherit these methods.

    Arguments
    ---------
    label : int
        label of node object
    coordinates : float
        coordinates (2d or 3d) of node
    ndofs : int, optional
        number of DOFs, noramlly defined later (after stacked in `ElDef`)
    global_dofs : int, optional
        global DOFs, normally defined later (after stacked in `ElDef`.
    '''

    def __init__(self, label, coordinates, ndofs=None, global_dofs=None):
        self.label = int(label)
        self.coordinates = np.array(coordinates)
        self.ndofs = ndofs                #number of dofs, normally defined later
        self.global_dofs = global_dofs    #global dofs, normally defined later
        self.global_ix = None

        # Defined during element initialization
        self.x0 = None
        self.x = None
        self.u = None 
        self.vel = None     # velocity
        self.du = None
        self.dim = len(coordinates[1:])
        
        # Initialize rotation tensor of node
        self.R = np.eye(3)

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


    def increment_rotation_tensor(self):
        '''
        Increments the rotation tensor describing the rotations of the node, based on defined
        increment of rotational DOFs.

        Notes
        ----------
        First, establishes the rotation tensor $[R_{inc}]$ from the incremental rotation ${\Delta \theta}$ from Rodrigues
        formula. Then, conducts the multiplication by the previous total rotational tensor $[R_{prev}]$ to yield new total 
        rotations $[R] = [R_{inc}] [R_{prev}]$.
        '''
        if len(self.coordinates)==3:
            self.R = rotation.R_from_drot(self.du[3:]) @ self.R     # global CSYS


    def get_deformation_rotations(self, R0n):
        '''
        Establishes deformation part of rotations based on input corotational configuration (rigid body motion from reference) 
        transformation matrix (defined at element level).

        Arguments
        ----------
        R0n : float
            3x3 matrix describing the rigid-body rotation to the corotated configuration from the base configuration ($C_0$ --> $C_{0n}$).
            
        Returns
        ----------
        def_rots : float
            numpy array with three components (pseudo-vector) of rotation from deformation contribution for node
            
        Notes
        ----------
        See Equation 4.41 and Section 2.4.3 in [[4]](../#4) Bruheim. Computes deformation tensor of total rotation from
        $[R_d] = [R] [R_{0n}]^T$, then converts to pseudo-vector with rotations (as they are small due to assumption of
        small strains).

        '''
        Rd = self.R @ R0n.T                 # Equation 4.41 in Bruheim [4]
        def_rots = rotation.rot_from_R(Rd)  # Small angles --> convert tensor to rotation vector, as in Section 2.4.3 in Bruheim

        return def_rots