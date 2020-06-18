"""
##############################################
Newmark time simulation module
##############################################
All functions related to Newmark solutions.
"""

import numpy as np
from scipy.linalg import solve


def is_converged(values, tols, scaling=None):
    """
    Check whether multiple values are below specified tolerances.  (value/scaling)
    
    Parameters
    -----------------
    values : double
        list of values to check
    tols : double
        corresponding list of tolerances to compare values to
    scaling : double, optional
        corresponding list of scaling of values


    Returns
    -----------------
    ok : boolean
        converged or not?

    Notes
    --------------------
    If entry in tols is None, the corresponding value is assumed to pass tolerance criterion. 
    If entry in tols is different than None, the value pass if value <= tol * scaling.
            

    """

    if scaling is None:
        scaling = np.ones([len(tols)])
    
    for ix, value in enumerate(values):
        if (tols[ix] is not None) and (value>(tols[ix]*scaling[ix])):
            return False
        
    return True


def residual(f, f_int, C, M, udot, uddot):
    return f - (M @ uddot + C @ udot + f_int)


def residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt):
    return (1+alpha)*f - alpha*f_prev - ((1+alpha)*f_int - alpha*f_int_prev + C @ ((1+alpha)*udot - alpha*udot_prev) + M @ uddot)


def effective_mass(M, C, K, dt, gamma, beta, alpha=0.0):
    return M + C*gamma*dt*(1+alpha) + K*beta*dt**2*(1+alpha)


def acc_estimate(K, C, M, f, udot, u=None, f_int=None, dt=None, beta=None, gamma=None):
    """
    Predict acceleration for time integration, based on internal forces, 
        damping matrix, mass matrix, current velocity and external force.

    Parameters:
    -----------
    K : double
        Next-step (tangent) stiffness matrix (time step k+1), 
        ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Current external forces (time step k), ndofs-by-1 Numpy array.
    udot : double
        Velocity at chosen time instance, ndofs-by-1 Numpy array.  
    u : optional, double
        Displacement at chosen time instance, ndofs-by-1 Numpy array. 
    f_int : optional, double
        Current internal forces (time step k). Equal K @ u if not given, 
        ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : double
        Scalar value specifying the beta parameter. 
    gamma : double
        Scalar value specifying the gamma parameter.

    Returns:
    -----------
    uddot : double
        Resulting acceleration array, ndofs-by-1 Numpy array.

    """
    
    if f_int is None:   # assume linear system/solution
        if u is not None:
            f_int = K @ u
        else:
            raise ValueError('Input either f_int or u!')

    acc = solve(M, f - C @ udot - f_int)

    return acc


def pred(u, udot, uddot, dt):
    """
    Predictor step in non-linear Newmark algorithm.

    Parameters:
    -----------
    u : double
        Current displacement (time step k), ndofs-by-1 Numpy array.
    udot : double
        Current velocity (time step k), ndofs-by-1 Numpy array.
    uddot : double
        Current acceleration (time step k), ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.


    Returns:
    -----------
    u : double
        Predicted next-step displacement (time step k+1), ndofs-by-1 Numpy array.
    udot : double
        Predicted next-step velocity (time step k+1), ndofs-by-1 Numpy array.
    uddot : double
        Predicted next-step acceleration (time step k+1), ndofs-by-1 Numpy array.
        (Input uddot is output without modifications)

    """
    du = dt*udot + 0.5*dt**2*uddot
    u = u + du
    udot = udot + dt*uddot

    return u, udot, uddot, du


def corr(r, K, C, M, u, udot, uddot, dt, beta, gamma):
    """
    Corrector step in non-linear Newmark algorithm.

    Parameters:
    -----------
    r : double
        Residual forces.
    K : double
        Next-step (tangent) stiffness matrix (time step k+1), 
        ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    u : double
        Next-step displacement, time step k+1, iteration i, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, iteration i, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, iteration i, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : double
        Scalar value specifying the beta parameter. 
    gamma : double
        Scalar value specifying the gamma parameter.

    Returns:
    -----------
    u : double
        Predicted next-step displacement, time step k+1, iteration i+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted next-step velocity, time step k+1, iteration i+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted next-step acceleration, time step k+1), iteration i+1, ndofs-by-1 Numpy array.
    norm_r : double
        Frobenius norm of residual force vector
    norm_u : double
        Frobenius norm of added displacement

    """

    K_eff = K + (gamma*dt)/(beta*dt**2)*C + 1/(beta*dt**2)*M
    du = solve(K_eff, r)
    
    u = u + du
    udot = udot + gamma*dt/(beta*dt**2)*du
    uddot = uddot + 1/(beta*dt**2)*du    

    return u, udot, uddot, du
                

def corr_alt(r, K, C, M, u, udot, uddot, dt, beta, gamma, alpha=0.0):
    """
    Corrector step in non-linear Newmark algorithm. Alternative version - uses Meff rather than Keff and allows for alpha damping.

    Parameters:
    -----------
    r : double
        Residual forces.
    K : double
        Next-step (tangent) stiffness matrix (time step k+1), 
        ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    u : double
        Next-step displacement, time step k+1, iteration i, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, iteration i, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, iteration i, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : double
        Scalar value specifying the beta parameter. 
    gamma : double
        Scalar value specifying the gamma parameter.

    Returns:
    -----------
    u : double
        Predicted next-step displacement, time step k+1, iteration i+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted next-step velocity, time step k+1, iteration i+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted next-step acceleration, time step k+1), iteration i+1, ndofs-by-1 Numpy array.
    norm_r : double
        Frobenius norm of residual force vector
    norm_u : double
        Frobenius norm of added displacement

    """
    Meff = effective_mass(M, C, K, dt, gamma, beta, alpha)
    duddot = solve(Meff, r)
    
    uddot = uddot + duddot
    udot = udot + duddot*gamma*dt
    
    du = duddot * beta*dt**2
    u = u + du
   
    return u, udot, uddot, du


def dnewmark(K, C, M, f, u, udot, uddot, dt, f_int=None, beta=1.0/4.0, gamma=0.5, 
             tol_u=1e-5, tol_r=1e-5, itmax=10):
    """
    Combined stepwise non-linear Newmark (predictor-corrector), 
        based on Algorithm 9.2 in Krenk, 2009. Because f_int is not updated each iteration
        this is equivalent to modified NR iteration.

    Parameters:
    -----------
    K : double
        Next-step (tangent) stiffness matrix (time step k+1), 
        ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Next-step external forces (time step k+1), ndofs-by-1 Numpy array.
    u : double
        Next-step displacement, time step k+1, iteration i, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, iteration i, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, iteration i, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    f_int : optional, double
        Current internal forces (time step k). Equal K @ u if not given, 
        ndofs-by-1 Numpy array.
    beta : 1/4, double
        Scalar value specifying the beta parameter. 
    gamma : 0.5, double
        Scalar value specifying the gamma parameter.
    tol_u : 1e-5, double
        Convergence satisfied when |du_{k+1}| < tol_u.
    tol_r : 1e-5, double
        Convergence satisfied when |dr_{k+1}| < tol_r.
    itmax : 10, int
        Maximum number of iterations allowed per time step / increment.
        

    Returns:
    -----------
    u : double
        Predicted displacement at time step k+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted velocity at time step k+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted acceleration at time step k+1, ndofs-by-1 Numpy array.
        
        
    References:
    --------------------
    :cite:`Krenk2009` 
    
    Other:
    --------------
    Because predictor and corrector steps are both included, 
    only modified Newton-Raphson is possible (can't update tangent stiffness 
    each iteration), because that would require updating model and reassembly of system. 
    Use newmark.pred and newmark.corr separately for full non-linear Newton-Raphson.

    """

    # If no internal stiffness force is provided, assume linear system => f_int = K u
    if f_int is None:
        f_int = K @ u

    # Predictor step and initial residual calc
    u, udot, uddot, du = pred(u, udot, uddot, dt)   
    f_int += K @ du
    r = residual(f, f_int, C, M, udot, uddot)

    # Loop through iterations until convergence is met
    for it in range(itmax):
        # Corrector step
        u, udot, uddot, du = corr(r, K, C, M, u, udot, uddot, dt, beta, gamma)
        
        # Update internal forces and calculate residual
        f_int += K @ du
        r = residual(f, f_int, C, M, udot, uddot)

        # Check convergence and break if convergence is met
        converged = is_converged([np.linalg.norm(du), np.linalg.norm(r)], 
                            [tol_u, tol_r])
        if converged:
            break
    
    return u, udot, uddot
           


def dnewmark_hht(K, C, M, f, u, udot, uddot, dt, f_prev, f_int=None, beta=1.0/4.0, gamma=0.5, alpha=0.0, 
             tol_u=1e-5, tol_r=1e-5, itmax=10):
    """
    Incremental formulation of Newmark allowing for alpha-damping.

    Parameters:
    -----------
    K : double
        Next-step (tangent) stiffness matrix (time step k+1), 
        ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Next-step external forces (time step k+1), ndofs-by-1 Numpy array.
    u : double
        Next-step displacement, time step k+1, iteration i, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, iteration i, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, iteration i, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    f_int : optional, double
        Current internal forces (time step k). Equal K @ u if not given, 
        ndofs-by-1 Numpy array.
    beta : 1/4, double
        Scalar value specifying the beta parameter. 
    gamma : 0.5, double
        Scalar value specifying the gamma parameter.
    tol_u : 1e-5, double
        Convergence satisfied when |du_{k+1}| < tol_u.
    tol_r : 1e-5, double
        Convergence satisfied when |dr_{k+1}| < tol_r.
    itmax : 10, int
        Maximum number of iterations allowed per time step / increment.
        

    Returns:
    -----------
    u : double
        Predicted displacement at time step k+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted velocity at time step k+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted acceleration at time step k+1, ndofs-by-1 Numpy array.
        
        
    References:
    --------------------
    Elena Chatzi, presentation.
    
    Other:
    --------------
    Because predictor and corrector steps are both included, 
    only modified Newton-Raphson is possible (can't update tangent stiffness 
    each iteration), because that would require updating model and reassembly of system. 
    Use newmark.pred and newmark.corr separately for full non-linear Newton-Raphson.

    """

    # If no internal stiffness force is provided, assume linear system => f_int = K u
    if f_int is None:
        f_int = K @ u
    
    # Save current status as previous
    u_prev = 1.0*u
    udot_prev = 1.0*udot
    f_int_prev = 1.0*f_int

    # Predictor step and initial residual calc
    u, udot, uddot, du = pred(u, udot, uddot, dt)   
    f_int += K @ du
    r = residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt)

    # Loop through iterations until convergence is met
    for it in range(itmax):
        # Corrector step
        u, udot, uddot, du = corr_alt(r, K, C, M, u, udot, uddot, dt, beta, gamma, alpha=alpha)

        # Update internal forces and calculate residual
        f_int += K @ du        
        r = residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt)

        # Check convergence and break if convergence is met
        converged = is_converged([np.linalg.norm(du), np.linalg.norm(r)], 
                            [tol_u, tol_r])
        if converged:
            break

        # Save current status as previous
        u_prev = 1.0*u
        udot_prev = 1.0*udot
        f_int_prev = 1.0*f_int
    
    return u, udot, uddot


def pred_lin(u, udot, uddot, dt, beta, gamma):
    """
    Predictor step in linear Newmark algorithm.

    Parameters:
    -----------
    u : double
        Current displacement (time step k), ndofs-by-1 Numpy array.
    udot : double
        Current velocity (time step k), ndofs-by-1 Numpy array.
    uddot : double
        Current acceleration (time step k), ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : double
        Scalar value specifying the beta parameter. 
    gamma : double
        Scalar value specifying the gamma parameter.

    Returns:
    -----------
    u : double
        Predicted next-step displacement (time step k+1), ndofs-by-1 Numpy array.
    udot : double
        Predicted next-step velocity (time step k+1), ndofs-by-1 Numpy array.
    uddot : double
        Predicted next-step acceleration (time step k+1), ndofs-by-1 Numpy array.
        Input is output without modification.

    """
    du = dt*udot + (0.5-beta)*dt**2*uddot
    u = u + du
    udot = udot + (1-gamma)*dt*uddot    
    
    return u, udot, uddot, du


def corr_lin(K, C, M, f, u, udot, uddot, dt, beta, gamma):
    """
    Corrector step in inear Newmark algorithm.

    Parameters:
    -----------
    K : double
        Next-step (tangent) stiffness matrix (time step k+1), 
        ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Next-step external forces (time step k+1), ndofs-by-1 Numpy array.
    u : double
        Next-step displacement, time step k+1, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : double
        Scalar value specifying the beta parameter. 
    gamma : double
        Scalar value specifying the gamma parameter.

    Returns:
    -----------
    u : double
        Predicted next-step displacement, time step k+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted next-step velocity, time step k+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted next-step acceleration, time step k+1), ndofs-by-1 Numpy array.

    """
    

    M_eff = effective_mass(M, C, K, dt, gamma, beta)
    uddot = acc_estimate(K, C, M_eff, f, udot, u=u)
    udot = udot + gamma*dt*uddot
    du = beta*dt**2*uddot
    u = u + du
        
    return u, udot, uddot, du


def dnewmark_lin(K, C, M, f, u, udot, uddot, dt, beta=1.0/4.0, gamma=0.5):
    """
    Combined (predictor-corrector) stepwise linear Newmark based on Algorithm 9.1 in Krenk, 2009.

    Parameters:
    -----------
    K : double
        Stiffness matrix, ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Next-step external forces (time step k+1), ndofs-by-1 Numpy array.
    u : double
        Next-step displacement, time step k+1, iteration i, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, iteration i, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, iteration i, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : 1.0/6.0, optional
        Scalar value specifying the beta parameter. 
    gamma : 0.5, optional
        Scalar value specifying the gamma parameter.
        

    Returns:
    -----------
    u : double
        Predicted displacement at time step k+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted velocity at time step k+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted acceleration at time step k+1, ndofs-by-1 Numpy array.
        
        
    References:
    --------------------
    :cite:`Krenk2009` 

    """
    u, udot, uddot, __ = pred_lin(u, udot, uddot, dt, beta, gamma)
    u, udot, uddot, __ = corr_lin(K, C, M, f, u, udot, uddot, dt, beta, gamma)

    return u, udot, uddot


def dnewmark_lin_alt(K, C, M, df, u, udot, uddot, dt, beta=1.0/4.0, gamma=0.5):
    """
    Alternative implementation, stepwise linear Newmark.

    Parameters:
    -----------
    K : double
        Stiffness matrix, ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Next-step external forces (time step k+1), ndofs-by-1 Numpy array.
    u : double
        Next-step displacement, time step k+1, iteration i, ndofs-by-1 Numpy array.
    udot : double
        Next-step velocity, time step k+1, iteration i, ndofs-by-1 Numpy array.
    uddot : double
        Next-step acceleration, time step k+1, iteration i, ndofs-by-1 Numpy array.
    dt : double
        Current time step, from k to k+1.
    beta : 1.0/4.0, optional
        Scalar value specifying the beta parameter. 
    gamma : 0.5, optional
        Scalar value specifying the gamma parameter.
        

    Returns:
    -----------
    u : double
        Predicted displacement at time step k+1, ndofs-by-1 Numpy array.
    udot : double
        Predicted velocity at time step k+1, ndofs-by-1 Numpy array.
    uddot : double
        Predicted acceleration at time step k+1, ndofs-by-1 Numpy array.


    """
    
    a = 1/(beta*dt)*M + gamma/beta*C
    b = 1/(2*beta)*M + dt*(gamma/(2*beta)-1)*C
    K_eff = K + gamma/(beta*dt)*C + 1/(beta*dt**2)*M

    df_hat = df + a @ udot + b @ uddot

    du = solve(K_eff, df_hat)
    
    dudot = gamma/(beta*dt)*du - gamma/beta*udot + dt*(1-gamma/(2*beta))*uddot
    duddot = 1/(beta*dt**2)*du - 1/(beta*dt)*udot - 1/(2*beta)*uddot
    
    u = u + du
    udot = udot + dudot
    uddot = uddot + duddot  
    
    return u, udot, uddot


def newmark_lin(K, C, M, f, t, u0, udot0, beta=1.0/4.0, gamma=0.5, solver='full_hht', alpha=0.0):
    """
    Combined linear Newmark (predictor-corrector), full time history.

    
    Parameters:
    -----------
    K : double
        Stiffness matrix, ndofs-by-ndofs Numpy array.
    C : double
        Damping matrix, ndofs-by-ndofs Numpy array.
    M : double
        Mass matrix, ndofs-by-ndofs Numpy array.
    f : double
        Full history of external forces, ndofs-by-nsamples Numpy array.
    u0 : double
        Initial displacement, ndofs-by-1 Numpy array.
    udot0 : double
        Initial velocity, ndofs-by-1 Numpy array.
    t : double
        Time instances corresponding to f, nsamples long Numpy array.
    beta : 1/4, optional
        Scalar value specifying the beta parameter. 
    gamma : 0.5, optional
        Scalar value specifying the gamma parameter.
    solver : {'full_hht', 'full', 'lin', 'lin_alt'}, optional
        What step-wise solver to enforce each time step. Useful for debugging.
    alpha : 0.0, optional
        Only used when 'nonlin_hht' is enforced
    

    Returns:
    -----------
    u : double
        Predicted displacement time history, ndofs-by-nsamples Numpy array.
    udot : double
        Predicted velocity time history, ndofs-by-nsamples Numpy array.
    uddot : double
        Predicted acceleration time history, ndofs-by-nsamples Numpy array.


    """    
    # Initialize response arrays
    u = f*0
    udot = f*0
    uddot = f*0
    n_samples = f.shape[1]   
    args = [[]]*n_samples

    # Assign initial conditions
    u[:, 0] = u0
    udot[:, 0] = udot0
    uddot[:, 0] = acc_estimate(K, C, M, f[:, 0], udot0, u0)
    
    # Prepare for different solver types
    if solver == 'lin':
        dnmrk_fun = dnewmark_lin
        kwargs = {}
    elif solver == 'lin_alt':   #BJF
        # This version uses df_k = f_k+1-f_k for each time step, 
        # so needs to redefine f
        dnmrk_fun = dnewmark_lin_alt
        f = np.diff(f, axis=1)
        f = np.insert(f, 0, f[:,0]*0, axis=1)
        kwargs = {}
        uddot[:, 0] = uddot[:, 0]*0     #remove initial acceleration estimate
    elif solver == 'full':
        dnmrk_fun = dnewmark
        kwargs = {'itmax': 1}
    elif solver == 'full_hht':
        dnmrk_fun = dnewmark_hht
        kwargs = {'itmax': 1, 'alpha': alpha}

        for k in range(n_samples):
            args[k] = [f[:, k]]

    # Loop through all time steps
    for k in range(n_samples-1):
        dt = t[k+1] - t[k]
        u[:, k+1], udot[:, k+1], uddot[:, k+1] = dnmrk_fun(K, C, M, f[:, k+1], u[:, k], udot[:, k], uddot[:, k], dt, beta=beta, gamma=gamma, *args[k], **kwargs)

    return u, udot, uddot


def factors_from_alpha(alpha):
    if alpha>0 or alpha<(-1.0/3.0):
        raise ValueError('alpha must be in range [-1/3, 0]')
    
    gamma = 0.5 * (1-2*alpha)
    beta = 0.25 * (1-alpha)**2
    return dict(beta=beta, gamma=gamma, alpha=alpha)


def factors(version='linear'):
    """ Gamma and beta factors for Newmark. Alpha is also output as zero, for convenience.
    
    Parameters:
    --------------
    version : {'linear', 'constant', 'average', 'fox-goodwin'}, optional
        String characterizing what method to use for Newmark simulation.
    
    Returns:
    -------------
    beta : double
        Beta factor for Newmark analysis.
    gamma : double
        Gamma factor for Newmark analysis.
    alpha : double
        Hard coded to zero for all cases herein
    """
    
    factors = {'average': {'beta': 0.25, 'gamma': 0.5, 'alpha': 0.0},
               'constant': {'beta': 0.25, 'gamma': 0.5, 'alpha': 0.0},
               'linear': {'beta': 1.0/6.0, 'gamma': 0.5, 'alpha': 0.0},
               'fox-goodwin': {'beta': 1.0/12.0, 'gamma':0.5, 'alpha': 0.0},
               'explicit': {'beta': 0, 'gamma': 0.5, 'alpha': 0.0}}
               
    return factors[version]
    
def stable_increment(omega_max, gamma=1/2, beta=1/6):
    return 1/(omega_max*np.sqrt(gamma*0.5-beta))