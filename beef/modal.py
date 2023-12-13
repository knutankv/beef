'''
Modal calculations.
'''

import numpy as np

def statespace(K, C, M):
    '''
    Establish state matrix based on stiffness, damping and mass matrices.

    Arguments
    ----------
    K : float
        global stiffness matrix (numpy array)
    C : float
        global damping matrix (numpy array)
    M : float
        global mass matrix (numpy array)

    Returns
    --------
    A : float
        state matrix
    '''

    ndofs = np.shape(K)[0]
    A = np.zeros([2*ndofs, 2*ndofs])
    A[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
    A[ndofs:2*ndofs, 0:ndofs] = -np.dot(np.linalg.inv(M), K)
    A[ndofs:2*ndofs, ndofs:2*ndofs] = -np.dot(np.linalg.inv(M), C)

    return A

def normalize_phi(phi, include_dofs=[0, 1, 2, 3, 4, 5], n_dofs=6, return_scaling=False):
    '''
    Normalize input phi matrix (modal transformation with modes stacked as columns).

    Arguments
    ----------
    phi : float
        modal transformation matrix with modes stacked column-wise
    include_dofs : [0,1,2,3,4,5], optional
        list of DOF integer indices to include in normalization
    n_dofs : 6, optional
        integer number of DOFs per node
    return_scaling : False, optional
        whether or not to return a second variable with the applied scalings

    Returns
    --------
    phi : float
        modal transformation matrix with modes stacked column-wise, 
        where each mode is normalized to the largest component
    mode_scaling : float
        array of scaling factors applied (only returned if requested by passing `return_scaling=True`)
    '''
    phi_n = phi*0

    phi_for_scaling = np.vstack([phi[dof::n_dofs, :] for dof in include_dofs])
    mode_scaling = np.max(np.abs(phi_for_scaling), axis=0)
    ix_max = np.argmax(np.abs(phi_for_scaling), axis=0)

    signs = np.sign(phi_for_scaling[ix_max, range(0, len(ix_max))])
    signs[signs==0] = 1

    mode_scaling[mode_scaling==0] = 1
    phi_n = phi/np.tile(mode_scaling[np.newaxis,:]*signs[np.newaxis,:], [phi.shape[0], 1])

    if return_scaling:
        return phi_n, mode_scaling
    else:
        return phi_n

    return phi_n, mode_scaling


def maxreal(phi):
    """
    Rotate complex vectors (stacked column-wise) such that the absolute value of the largest real part is maximized.

    Arguments
    ---------------------------
    phi : double
        complex-valued modal transformation matrix (column-wise stacked mode shapes)

    Returns
    ---------------------------
    phi_max_real : boolean
        complex-valued modal transformation matrix, with vectors rotated to have maximum real parts
    """   

    angles = np.expand_dims(np.arange(0,np.pi/2, 0.01), axis=0)
    
    if phi.ndim==1:
        phi = np.array([phi]).T

    phi_max_real = np.zeros(np.shape(phi)).astype('complex')        

    for mode in range(np.shape(phi)[1]):
        rot_mode = np.dot(np.expand_dims(phi[:, mode], axis=1), np.exp(angles*1j))
        max_angle_ix = np.argmax(np.real(rot_mode)**2, axis=1)

        phi_max_real[:, mode] = phi[:, mode] * np.exp(angles[0, max_angle_ix]*1j)*np.sign(sum(np.real(phi[:, mode])))
  
    return phi_max_real