'''
General purpose functions.
'''

import numpy as np
from scipy.linalg import null_space as null

#%% General functions
def n2d_ix(node_ixs, n_dofs=6):
    '''
    Convert node indices to dof indices.

    Arguments
    ----------
    node_ixs : int
        list/array of indices of nodes
    n_dofs : 6, optional    
        number of dofs per node

    Returns
    ----------
    dof_ixs : int
        array of DOF indices corresponding to input node indices
    '''
    return np.hstack([node_ix*n_dofs+np.arange(n_dofs) for node_ix in node_ixs])


def extract_dofs(mat, dof_ix=[0,1,2], n_dofs=6, axis=0):
    '''
    Extract selected dofs from matrix.

    Arguments
    ------------
    mat : float
        numpy array (matrix) to extract DOFs from
    dof_ix : int, optional
        list of DOF indices (local numbering of DOFs in node) to extract
    n_dofs : 6, optional
        number of DOFs per node
    axis : 0, optional
        axis to pick along

    Returns
    ----------
    mat_sub : float
        subselection of matrix
    '''
    get_dofs = np.vstack([np.arange(dof, mat.shape[axis],n_dofs) for dof in dof_ix]).T.flatten()
    return mat.take(get_dofs, axis=axis)

def convert_dofs(dofs_in, node_type='beam3d', sort_output=True):
    '''
    Convert string DOFs to indices.

    Arguments
    ----------
    dofs_in : {'all', 'trans', 'rot'} or int
        string or list of indices
    node_type : {'beam2d', 'beam3d'}, optional
        type of nodes (defining number of DOFs per node)
    sort_output : True, optional
        whether or not to sort the DOF indices output
    
    Returns
    --------
    dofs_out : int
        array of DOF indices 
    '''
    dof_dict = dict(beam2d=dict(all=np.arange(0,3), trans=np.arange(0,2), rot=np.arange(2,3)), 
                        beam3d=dict(all=np.arange(0,6), trans=np.arange(0,3), rot=np.arange(3,6)))

    if dofs_in in ['all', 'trans', 'rot']:
        dofs_out = dof_dict[node_type][dofs_in]
    else:
        dofs_out = dofs_in
        
    if sort_output:
        dofs_out = np.sort(dofs_out)
        
    return dofs_out
        

def convert_dofs_list(dofs_list_in, n_nodes, node_type='beam3d', sort_output=True):
    '''
    Convert DOFs list to correct format.

    Arguments
    -----------
    dofs_list_in : str,int
        list of strings and indices corresponding to DOFs to
    n_nodes : int
        number of nodes the DOFs are related to
    node_type : {'beam3d', 'beam2d'}
        type of nodes (defining number of DOFs per node)
    sort_output : True
        whether or not to sort output

    Returns
    -----------
    dofs_list_out : int
        array of DOF indices
    '''
    
    contains_strings = np.any([type(d) is str for d in dofs_list_in])
    
    if type(dofs_list_in) is not list:  # single input (all, rot or trans)
        dofs_list_out = [dofs_list_in]*n_nodes      
    elif ~contains_strings and (len(dofs_list_in)<=6) and (np.max(dofs_list_in)<6):   #
        dofs_list_out = [dofs_list_in]*n_nodes
    elif len(dofs_list_in)!=n_nodes:
        raise TypeError('Wrong input format of "dofs"')
    else:
        dofs_list_out = dofs_list_in
        
    for ix, dofs_list_out_i in enumerate(dofs_list_out):
        dofs_list_out[ix] = convert_dofs(dofs_list_out_i, node_type=node_type, sort_output=sort_output)
    
    return dofs_list_out


def transform_unit(e1, e2p):
    '''
    Establish transformation matrix from e1 and temporary e2 vectors.

    Arguments
    -----------
    e1 : float
        unit vector describing element longitudinal direction
    e2p : float
        unit vector describing a chosen vector that's perpendicular to the longitudinal direction

    Returns
    -----------
    T : float
        transformation matrix


    Notes
    -----------
    Input vectors \(\{e_1\}\) and \(\{e_{2p}\}\) are used to establish a third unit vector, as the cross product of the two, as follows:

    $$
    \{e_3\} = \{e_1\} \\times \{e_{2p}\}
    $$

    Then, a final vector is created based on a second cross product:
    $$
    \{e_2\} = \{e_3\} \\times \{e_1\}
    $$

    Finally, the transformation matrix is established as follows:

    $$
    [T] = \\begin{bmatrix}
        \{e_1\}^T \\\\
        \{e_2\}^T \\\\
        \{e_3\}^T
    \\end{bmatrix}
    $$

    where the unit vectors established are the rows of the transformation matrix.

    '''

    e1 = np.array(e1).flatten()
    e2p = np.array(e2p).flatten()
    
    e3 = np.cross(e1, e2p)         # Direction of the third unit vector
    e2 = np.cross(e3, e1)          # Direction of the second unit vector

    e1 = e1/np.linalg.norm(e1)     # Normalize the direction vectors to become unit vectors
    e2 = e2/np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    
    T = np.vstack([e1,e2,e3])
    
    return T


def gdof_ix_from_nodelabels(all_node_labels, node_labels, dof_ix=[0,1,2]):     # from nlfe - not debugged
    '''
    Get global DOF indices from node labels.

    Arguments
    ------------
    all_node_labels : int
        list or array of all node labels
    node_labels : int
        list or array of the node labels to get the indices of
    dof_ix : int
        local DOF indices to use

    Returns
    ---------
    gdof_ix : int
        global indices of DOFs of requested nodes
    '''
    if type(node_labels) is not list:
        node_labels = [node_labels]
    
    node_ix = [np.where(nl==all_node_labels)[0] for nl in node_labels]
    gdof_ix = gdof_from_nodedof(node_ix, dof_ix)
    
    return gdof_ix


def gdof_from_nodedof(node_ix, dof_ixs, n_dofs=3, merge=True):
    '''
    Get global DOF from node DOF.

    Arguments
    ----------
    node_ix : int
        node indices to establish global DOF indices of
    dof_ixs : int
        local DOF indices to include
    n_dofs : 3, optional
        number of DOFs (all)
    merge : True, optional
        whether or not to merge the global DOF indices from all nodes, to a single list

    Returns
    ---------
    gdof_ix : int
        global indices of DOFs of requested nodes

    '''
    gdof_ix = []
    
    if type(node_ix) is not list:
        node_ix = [node_ix]
    
    if type(dof_ixs) is not list:
        dof_ixs = [dof_ixs]*len(node_ix)
        
    elif len(dof_ixs) != len(node_ix):
        dof_ixs = [dof_ixs]*len(node_ix)

    for ix, n in enumerate(node_ix):
        gdof_ix.append(n*n_dofs + np.array(dof_ixs[ix]))
    
    if merge:
        gdof_ix = np.array(gdof_ix).flatten()
        
    return gdof_ix

def B_to_dofpairs(B, master_val=1):
    '''
    Establish pairs of indices of DOFs to couple from compatibility matrix B.

    Arguments
    ----------
    B : float   
        Lagrange compatiblity \([B]\) matrix
    master_val : float
        value used to identify master DOF

    Returns
    ----------
    dof_pairs : int
        indices of DOFs paired through input \([B]\) matrix

    '''
    n_constr, n_dofs = B.shape
    dof_pairs = [None]*n_constr
    
    for icon in range(0, n_constr):
        master = np.where(B[icon,:] == master_val)[0][0]
        slave = np.where(B[icon,:] == -master_val)[0]
        if len(slave) != 0:
            slave = slave[0]
        else:
            slave = None
        
        dof_pairs[icon] = [master, slave]
        dof_pairs = np.array(dof_pairs).T

    return dof_pairs


def dof_pairs_to_Linv(dof_pairs, n_dofs):    
    '''
    Establish quasi-inverse of L from dof pairs describing constraints.

    Arguments
    ----------
    dof_pairs : int   
        list of lists of indices of DOFs paired
    n_dofs : int
        number of DOFs

    Returns
    ----------
    Linv : int

    Notes
    ----------
    \([L_{inv}]\) is a constructed quantity, conventient as it can construct the constrainted u (unique, free DOFs) from the full as follows:

    $$
    \{u_c\} = [L_{inv}] \{u\}
    $$

    It is worth noting that it only works for fully hard-constrained systems (no weak connections, or similar).

    '''

    slave_nodes = dof_pairs[dof_pairs[:,1]!=None, 1]
    grounded_nodes = dof_pairs[dof_pairs[:,1]==None, 0]
    
    remove = np.unique(np.hstack([grounded_nodes, slave_nodes]))   
    all_dofs = np.arange(0,n_dofs)
    primal_dofs = np.setdiff1d(all_dofs, remove)
    
    Linv = np.zeros([len(primal_dofs), n_dofs])

    for primal_dof_ix, primal_dof in enumerate(primal_dofs):
        Linv[primal_dof_ix, primal_dof] = 1
    
    return Linv


def compatibility_matrix(dof_pairs, n_dofs):
    '''
    Establish compatibility matrix from specified pairs of DOFs.

    Arguments
    ----------
    dof_pairs : int   
        list of lists of indices of DOFs paired
    n_dofs : int
        number of DOFs in full system; defines the number of rows of \([B]\)

    Returns
    ----------
    B : int
        numpy array describing compatibility matrix \([B]\)
    '''
    n_constraints = dof_pairs.shape[0]
    B = np.zeros([n_constraints, n_dofs])

    for constraint_ix, dof_pair in enumerate(dof_pairs):
        if dof_pair[1] is None:    # connected dof is None --> fixed to ground
            B[constraint_ix, dof_pair[0]] = 1
        else:
            B[constraint_ix, dof_pair[0]] = 1
            B[constraint_ix, dof_pair[1]] = -1
    return B


def lagrange_constrain(mat, dof_pairs, null=False):
    '''
    Lagrange constrain matrix from specified DOF pairs.

    Arguments
    -----------------
    mat : float
        matrix to constrain
    dof_pairs : int
        list of lists of indices of DOFs paired
    null : False, optional
        to create a 0 matrix of the correct size, this can be set to True

    Returns
    ----------
    mat_fixed : float
        constrained matrix
    '''
    # Lagrange multiplier method - expand matrix
    n_constraints = len(dof_pairs)
    n_dofs = mat.shape[0]
    
    if not null:
        B = compatibility_matrix(dof_pairs, n_dofs)
    else:
        B = np.zeros([n_constraints, n_dofs])
        
    O = np.zeros([n_constraints, n_constraints])
    mat_fixed = np.vstack([np.hstack([mat, B.T]), np.hstack([B, O])])
   
    return mat_fixed

def lagrange_constrain_from_B(mat, B, null=False):
    '''
    Lagrange constrain matrix from specified compatibility matrix.

    Arguments
    -----------------
    mat : float
        matrix to constrain
    B : int
        numpy array describing compatibility matrix \([B]\)
    null : False, optional
        to create a 0 matrix of the correct size, this can be set to True

    Returns
    ----------
    mat_fixed : float
        constrained matrix
    '''
    # Lagrange multiplier method - expand matrix
    n_constraints = B.shape[0]        
    O = np.zeros([n_constraints, n_constraints])
    mat_fixed = np.vstack([np.hstack([mat, B.T]), np.hstack([B, O])])
   
    return mat_fixed


def expand_vector_with_zeros(vec, n_constraints):
    '''
    Append vector with zeros based on number of constraints (n_constraints).
    '''
    vec_expanded = np.hstack([vec.flatten(), np.zeros(n_constraints)])[np.newaxis, :].T
    return vec_expanded


def blkdiag(mat, n):
    return np.kron(np.eye(n), mat)


def nodes_to_beam_element_matrix(node_labels, first_element_label=1):
    '''
    Establish element matrix definition assuming ordered node labels.

    Arguments
    -----------
    node_labels : int
        list of integer node labels
    first_element_label : int
        first integer used in element matrix

    Returns
    -----------
    element_matrix : int
        element definition where each row represents an element as [element_label_i, node_label_1, node_label_2]

    '''
    n_nodes = len(node_labels)
    n_elements = n_nodes-1
    
    element_matrix = np.empty([n_elements, 3])
    element_matrix[:, 0] = np.arange(first_element_label,first_element_label+n_elements)
    element_matrix[:, 1] = node_labels[0:-1]
    element_matrix[:, 2] = node_labels[1:]

    return element_matrix


def get_phys_modes(phi, B, lambd=None, return_as_ix=False):
    '''
    Get physical modes.

    Arguments
    -----------
    phi : float
    B : int
        numpy array describing compatibility matrix \([B]\)
    lambd : float, optional
        standard value None does not 
    return_as_ix : False, optional
        whether or not to output as indices instead of eigenvectors and eigenvalues

    Returns
    ----------
    lambd_phys : float
        physical eigenvalue array
    phi_phys : float
        physical eigenvector array (modal transformation matrix)
    phys_ix : int
        indices of selected DOFs (if `return_as_ix=True` is passed)
    '''
    n_lagr = B.shape[0]
    u_ = phi[0:-n_lagr, :]     #physical dofs
    l_ = phi[-n_lagr:,:]       #lagr dofs
    
    # Compatibility at interface
    mask1 = np.all((B @ u_)==0, axis=0)
    
    # Equilibrium at interface
    L = null(B)
    g = -B.T @ l_
    mask2 = np.all((L.T @ g)==0, axis=0)
    
    phys_ix = np.logical_and(mask1, mask1)
    
    if return_as_ix:
        return phys_ix
    else:  
        if lambd is None:
            lambd_phys = None
        else:
            lambd_phys = lambd[phys_ix]
            
        phi_phys = u_[:, phys_ix]
        
        return lambd_phys, phi_phys    
    
    
def ensure_list(list_in, levels=1, increase_only=True):
    '''
    Ensure input variable is list.

    Arguments
    ----------
    list_in : float, int
        list or scalar
    levels : int
        number of levels of list (nest-level)
    increase_only : True, optional
        if True, only increase amount of list-wrapping

    Returns
    ----------
    list_out : float, int
        list
    '''

    dimensions = np.array(list_in).ndim 

    if type(list_in) is not list:
        list_out = [list_in]
        dimensions = dimensions+1
    else:
        list_out = list_in * 1
    
    
    if not increase_only:
        while dimensions>levels:
            list_out = list_out[0]
            dimensions = dimensions - 1

    while dimensions<levels:            
        list_out = [list_out]
        dimensions = dimensions + 1

    return list_out


def feature_matrix(master_dofs, values, slave_dofs=None, ndofs=None, return_small=False):
    '''
    Arguments
    ---------
    master_dofs : int
        list of indices of master DOFs
    values : float
        list of amplitudes/values for features
    slave_dofs : int, optional
        list of indices of slave DOFs (standard value None, makes grounded connections from master_dofs)
    ndofs : int, optional
        number of modes used to construct full matrix (not relevant if `return_small=True`)
    return_small : False, optional
        whether or not to return small or full matrix - if `return_small=True`, returns small ndofs-by-ndofs matrix and corresponding indices in global matrix; if `return_small=False` returns big ndofs_global-by-ndofs_global matrix
    
    Returns
    ---------
    mat : float
        feature matrix
    dof_ixs : int
        list of indices of DOFs for relevant feature connectivity (only returned if `return_small=True`)

    '''
    
    ndofs_small = len(master_dofs)
    if type(values) is float or type(values) is int:
        values = [float(values)]*ndofs_small
    elif len(values) != len(master_dofs):
        raise ValueError('Length of master_dofs and values must match')

    if slave_dofs is None:
        small = np.diag(values)
        slave_dofs = []
    else:
        if len(slave_dofs) != len(master_dofs):
            raise ValueError('Length of master_dofs and slave_dofs must match')
        small = np.diag(np.hstack([values,values]))
        for ix in range(ndofs_small):
            small[ix, ix+ndofs_small] = small[ix+ndofs_small, ix] = -values[ix]

    dof_ixs = np.hstack([master_dofs, slave_dofs]).astype(int)

    if return_small:
        return small, dof_ixs
    else:
        if len(set(dof_ixs)) != len(dof_ixs):
            raise ValueError('Non-unique dof indices are provided')
        if ndofs is None:
            ndofs = np.max(dof_ixs)+1

        large = np.zeros([ndofs, ndofs])
        large[np.ix_(dof_ixs, dof_ixs)] = small
        
        return large

def basic_coupled():
    return np.array([[1, -1], [-1, 1]])

def generic_beam_mat(L, yy=0, yz=0, yt=0, zy=0, zz=0, zt=0, ty=0, tz=0, tt=0):
    mat = np.zeros([12,12])

    mat[0:6, 0:6] = np.array([
        [0,         0,          0,          0,          0,              0           ],
        [0,         156*yy,    156*yz,    147*yt,    -22*L*yz,      22*L*yy    ],
        [0,         156*zy,    156*zz,    147*zt,    -22*L*zz,      22*L*zy    ],
        [0,         147*ty,    147*tz,    140*tt,     -21*L*tz,      21*L*ty   ],
        [0,         -22*L*zy,  -22*L*zz,  -21*L*zt,  4*L**2*zz,     -4*L**2*zy ],
        [0,         22*L*yy,   22*L*yz,   21*L*yt,   -4*L**2*yz,    4*L**2*yy  ],
    ])

    mat[0:6, 6:12] = np.array([
        [0,         0,          0,          0,          0,              0            ],
        [0,         54*yy,    54*yz,      63*yt,     13*L*yz,       -13*L*yy    ],
        [0,         54*zy,    54*zz,      63*zt,     13*L*zz,       -13*L*zy    ],
        [0,         63*ty,    63*tz,      70*tt,     14*L*tz,       -14*L*ty    ],
        [0,         -13*L*zy,  -13*L*zz,  -14*L*zt,  -3*L**2*zz,     3*L**2*zy  ],
        [0,         13*L*yy,   13*L*yz,   14*L*yt,   3*L**2*yz,     -3*L**2*yy  ],
    ])

    mat[6:12, 0:6] = np.array([
        [0,         0,          0,          0,          0,              0            ],
        [0,         54*yy,    54*yz,      63*yt,     -13*L*yz,       13*L*yy    ],
        [0,         54*zy,    54*zz,      63*zt,     -13*L*zz,       13*L*zy    ],
        [0,         63*ty,    63*tz,      70*tt,     -14*L*tz,       14*L*ty    ],
        [0,         13*L*zy,  13*L*zz,    14*L*zt,   -3*L**2*zz,     3*L**2*zy  ],
        [0,         -13*L*yy, -13*L*yz,   -14*L*yt,   3*L**2*yz,     -3*L**2*yy ],
    ])

    mat[6:12,6:12] = np.array([
        [0,         0,          0,          0,          0,              0               ],
        [0,         156*yy,    156*yz,    147*yt,    22*L*yz,      -22*L*yy        ],
        [0,         156*zy,    156*zz,    147*zt,    22*L*zz,      -22*L*zy        ],
        [0,         147*ty,    147*tz,    140*tt,    21*L*tz,      -21*L*ty        ],
        [0,         22*L*zy,   22*L*zz,   21*L*zt,    4*L**2*zz,   -4*L**2*zy      ],
        [0,         -22*L*yy,   -22*L*yz,   -21*L*yt,   -4*L**2*yz,    4*L**2*yy   ],
    ])

    return mat * L / 420


def bar_foundation_stiffness(L, kx, ky, kz):    #only z and y currently, will be extended!
    mat = np.vstack([ 
        [kx*1/4, 0, 0, 0, 0, 0,     kx*1/4, 0, 0, 0, 0, 0],     #ux1
        [0, ky*1/3, 0, 0, 0, 0,     0, ky*1/6, 0, 0, 0, 0],     #uy1
        [0, 0, kz*1/3, 0, 0, 0,     0, 0, kz*1/6, 0, 0, 0],     #uz1

        [0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0],          #rx1
        [0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0],          #ry1
        [0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0],          #rz1

        [kx*1/4, 0, 0, 0, 0, 0,     kx*1/4, 0, 0, 0, 0, 0],     #ux2
        [0, ky*1/6, 0, 0, 0, 0,     0, ky*1/3, 0, 0, 0, 0],     #uy2
        [0, 0, kz*1/6, 0, 0, 0,     0, 0, kz*1/3, 0, 0, 0],     #uz2

        [0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0],          #rx2
        [0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0],          #ry2
        [0, 0, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0]]) * L      #rz2
    return mat