import numpy as np
from scipy.linalg import null_space as null

#%% General functions
def extract_dofs(mat, dof_ix=[0,1,2], n_dofs=6, axis=0):
    get_dofs = np.vstack([np.arange(dof, mat.shape[axis],n_dofs) for dof in dof_ix]).T.flatten()
    return mat.take(get_dofs, axis=axis)

def convert_dofs(dofs_in, node_type='beam3d', sort_output=True):
    dof_translate = dict(beam2d=dict(all=np.arange(0,3), trans=np.arange(0,2), rot=np.arange(2,3)), 
                        beam3d=dict(all=np.arange(0,6), trans=np.arange(0,3), rot=np.arange(3,6)))

    if dofs_in in ['all', 'trans', 'rot']:
        dofs_out = dof_translate[node_type][dofs_in]
    else:
        dofs_out = dofs_in
        
    if sort_output:
        dofs_out = np.sort(dofs_out)
        
    return dofs_out
        

def convert_dofs_list(dofs_list_in, n_nodes, node_type='beam3d', sort_output=True):
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
        dofs_list_out[ix] = convert_dofs(dofs_list_out_i, node_type=node_type, sort_output=sort_output)
    
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


def gdof_ix_from_nodelabels(all_node_labels, node_labels, dof_ix=[0,1,2]):     # from nlfe - not debugged
    
    if type(node_labels) is not list:
        node_labels = [node_labels]
        
    node_ix = [np.where(nl==all_node_labels)[0] for nl in node_labels]
    gdof_ix = gdof_from_nodedof(node_ix, dof_ix)
    
    return gdof_ix


def gdof_from_nodedof(node_ix, dof_ixs, n_dofs=3, merge=True):
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
    # u_constr = Linv @ u_full (u_full: all DOFs, u_constr are unique,free DOFs)
    
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



