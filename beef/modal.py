import numpy as np

def statespace(K, C, M):
    ndofs = np.shape(K)[0]
    A = np.zeros([2*ndofs, 2*ndofs])
    A[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
    A[ndofs:2*ndofs, 0:ndofs] = -np.dot(np.linalg.inv(M), K)
    A[ndofs:2*ndofs, ndofs:2*ndofs] = -np.dot(np.linalg.inv(M), C)

    return A

def normalize_phi(phi, include_dofs=[0,1,2,3,4,5], n_dofs=6, return_scaling=False):
    phi_n = phi*0

    phi_for_scaling = np.vstack([phi[dof::n_dofs, :] for dof in include_dofs])
    mode_scaling = np.max(np.abs(phi_for_scaling), axis=0)
    ix_max = np.argmax(np.abs(phi_for_scaling), axis=0)
    signs = np.sign(phi_for_scaling[ix_max, range(0, len(ix_max))])
    signs[signs==0] = 1
    mode_scaling[mode_scaling==0] = 1

    phi_n = phi/np.tile(mode_scaling[np.newaxis,:]/signs[np.newaxis,:], [phi.shape[0], 1])

    if return_scaling:
        return phi_n, mode_scaling
    else:
        return phi_n
