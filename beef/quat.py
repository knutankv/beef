'''
Quaternion operation functions.
'''

import numpy as np

def quat_from_R(R):
    '''
    Calculate quaternions from matrix [R]

    Arguments
    ----------
    R : float
        numpy 3x3 matrix describing to rotation transformation
        matrix equivalent to the input quaternion representation
        
    Returns 
    ---------
     r0 : float
         scalar defining the trace of the rotation tensor
     r : float
         numpy array with three quaternions r1,r2,r3
    '''
    

    e = lambda i,j,k:(i-j)*(j-k)*(k-i)/2

    r0 = (np.trace(R)+1)/4
    r = np.zeros(3)
    for l in range(3):
        e_mat = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                e_mat[i,j] = e(l, i, j)
        
        r[l] = -np.sum(e_mat*R)/4/r0
    
    return r0, r


def R(r0, r, row_wise=True): 
    '''
    Calculate transformation matrix [R] according to
    Eq 3.50 in Krenk [1].

    Arguments
    ----------
    r0 : float
        scalar defining the trace of the rotation tensor
    r : float
        numpy array with three quaternions r1,r2,r3
    row_wise : {True, False}
        if converting the result such that basis vectors are
        stacked column-wise (instead of the column-wise used in Krenk)

    Returns 
    ---------
    R : float
        numpy 3x3 matrix describing to rotation transformation
        matrix equivalent to the input quaternion representation
    '''

    r_hat = np.array([[0, -r[2], r[1]], 
                     [r[2], 0, -r[0]], 
                     [-r[1], r[0], 0]]) # Equation 3.7 in Krenk [1]

    R = (r0**2 - np.dot(r,r)) * np.eye(3) + 2*r0*r_hat + 2 * np.outer(r,r)

    # Convert such that unit vectors are stacked row-wise
    if row_wise:    
        R = R.T

    return R

def increment_from_drot(drot):
    '''
    Establish linearized quaternion increments from 
    given rotation increments.

    Arguments
    -----------
    drot : float
        3-by-1 or 1-by-3 numpy array with three rotation increments of a node

    Returns
    -----------
    dr0 : float
        scalar defining the trace of the rotation tensor
        of the incremental rotation
    dr : float
        numpy array with three quaternions r1,r2,r3, corresponding
        to the incremental rotation
    '''

    dr = 0.5 * drot
    dr0 = np.sqrt(1 - np.dot(dr,dr))

    return dr0, dr

def add_increment_from_quat(r0, r, dr0, dr):
    '''
    Add incremental rotation quaternions to initial rotation using Eq. 5.124
    in Krenk [1].

    Arguments
    ----------
    r0 : float
        scalar defining the trace of the initial rotation tensor
    r : float
        numpy array with three quaternions r1,r2,r3 representing
        the inital rotation
    dr0 : float
        scalar defining the trace of the rotation tensor
        of the incremental rotation
    dr : float
        numpy array with three quaternions r1,r2,r3, corresponding
        to the incremental rotation

    Returns 
    ---------
    r0 : float
        scalar defining the trace of the final rotation tensor
    r : float
        numpy array with three quaternions r1,r2,r3 representing
        the final rotation
    '''

    r0 = dr0*r0 - np.dot(dr, r)
    r = dr0*r + r0*dr + np.cross(dr, r)

    return r0, r

def add_increment_from_rot(r0, r, drot):
    '''
    Add incremental rotation to initial rotation using Eq. 5.124
    in Krenk [1]. Combining functions inc_from_rot and increment_from_quat.
    
    Arguments
    -----------
    r0 : float
        scalar defining the trace of the initial rotation tensor
    r : float
        numpy array with three quaternions r1,r2,r3 representing
        the inital rotation
    drot : float
        3-by-1 or 1-by-3 numpy array with three rotation increments of a node

    Returns 
    ---------
    r0 : float
        scalar defining the trace of the final rotation tensor
    r : float
        numpy array with three quaternions r1,r2,r3 representing
        the final rotation
    '''

    dr0, dr = increment_from_drot(drot)
    r0, r = add_increment_from_quat(r0, r, dr0, dr)

    return r0,r


def mean(ra0, ra, rb0, rb):
    '''
    Calculate the mean and difference quaternions from two sets of quaternions
    {ra0, ra} and {rb0, rb}.

    Arguments
    ----------
    ra0 : float
        scalar defining the trace of the rotation tensor
        of the first point
    ra : float
        numpy array with three quaternions r1,r2,r3 representing
        the rotation of the first point
    rb0 : float
        scalar defining the trace of the rotation tensor
        of the second point
    rb : float
        numpy array with three quaternions r1,r2,r3 representing
        the rotation of the second point

    Returns 
    ---------
    r0 : float
        scalar defining the trace of the final rotation tensor
    r : float
        numpy array with three quaternions r1,r2,r3 representing
        the final rotation
    s0 : float
        scalar part of difference quaternion
    s : float
        numpy array with three quaternions s1,s2,s3 representing
        the difference in rotation
    '''    

    s0 = 0.5 * np.sqrt((ra0+rb0)**2 + np.linalg.norm(ra+rb)**2)
    r0 = 0.5 * (ra0+rb0)/s0
    r = 0.5 * (ra+rb)/s0
    s = 0.5 * (ra0*rb - rb0*ra + np.cross(ra,rb))/s0

    return r0, r, s0, s
