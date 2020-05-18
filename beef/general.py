import numpy as np
from scipy.linalg import null_space as null

#%% General functions
def extract_dofs(mat, dof_ix=[0,1,2], n_dofs=6, axis=0):
    get_dofs = np.vstack([np.arange(dof, mat.shape[axis],n_dofs) for dof in dof_ix]).T.flatten()
    return mat.take(get_dofs, axis=axis)

def convert_dofs(dofs_in, sort_output=True):
    if dofs_in == 'all':
        dofs_out = np.arange(0,6)
    elif dofs_in == 'trans' or dofs_in == 'translation':
        dofs_out = np.arange(0,3)
    elif dofs_in == 'rot' or dofs_in == 'rotation':
        dofs_out = np.arange(3,6)
    else:
        dofs_out = dofs_in
        
    if sort_output:
        dofs_out = np.sort(dofs_out)
        
    return dofs_out
        

def convert_dofs_list(dofs_list_in, n_nodes, sort_output=True):
    contains_strings = np.any([type(d) is str for d in dofs_list_in])
    
    if type(dofs_list_in) is not list:  # single input (all, rot or trans)
        dofs_list_out = [dofs_list_in]*n_nodes      
    elif ~contains_strings and (len(dofs_list_in)!=n_nodes) and (len(dofs_list_in)<=6) and (np.max(dofs_list_in)<6):   #
        dofs_list_out = [dofs_list_in]*n_nodes
    elif len(dofs_list_in)!=n_nodes:
        raise TypeError('Wrong input format of "dofs"')
    else:
        dofs_list_out = dofs_list_in
        
    for ix, dofs_list_out_i in enumerate(dofs_list_out):
        dofs_list_out[ix] = convert_dofs(dofs_list_out_i, sort_output=sort_output)
    
    return dofs_list_out


def transform_unit(e1, e2p):
    # Copy from wawi
    e1 = np.array(e1).flatten()
    e2p = np.array(e2p).flatten()
    
    e3 = np.cross(e1, e2p)         # Direction of the third unit vector
    e2 = np.cross(e3, e1)          # Direction of the second unit vector

    e1 = e1/np.linalg.norm(e1)     # Normalize the direction vectors to become unit vectors
    e2 = e2/np.linalg.norm(e2)
    e3 = np.cross(e1,e2)
    
    T = np.vstack([e1,e2,e3])
    
    return T


def compatibility_matrix(dof_pairs, n_dofs):
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
    # Lagrange multiplier method - expand matrix
    n_constraints = B.shape[0]        
    O = np.zeros([n_constraints, n_constraints])
    mat_fixed = np.vstack([np.hstack([mat, B.T]), np.hstack([B, O])])
   
    return mat_fixed


def expand_vector_with_zeros(vec, n_constraints):
    vec_expanded = np.hstack([vec.flatten(), np.zeros(n_constraints)])[np.newaxis, :].T
    return vec_expanded


def blkdiag(mat, n):
    return np.kron(np.eye(n), mat)


def nodes_to_beam_element_matrix(node_labels, first_element_label=1):
    n_nodes = len(node_labels)
    n_elements = n_nodes-1
    
    element_matrix = np.empty([n_elements, 3])
    element_matrix[:, 0] = np.arange(first_element_label,first_element_label+n_elements)
    element_matrix[:, 1] = node_labels[0:-1]
    element_matrix[:, 2] = node_labels[1:]

    return element_matrix


def get_phys_modes(phi, B, lambd=None, return_as_ix=False):
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
    

def frame_creator(frames=30, repeats=1, swing=False, full_cycle=False):
    
    if full_cycle:
        d = 2/frames
        start = -1
    else:
        d = 1/frames
        start = 0
    
    if swing:
        base_scaling = np.hstack([np.linspace(start,1-d,frames), np.linspace(1,start+d,frames)])
    else:
        base_scaling = np.linspace(start,1-d,frames) 

    return np.tile(base_scaling, repeats)


def feature_matrix(master_dofs, values, slave_dofs=None, ndofs=None, return_small=False):
    # return_small defined as True => returns small ndofs-by-ndofs matrix and corresponding indices in global matrix
    # return_small defined as False => returns big ndofs_global-by-ndofs_global matrix
    
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



