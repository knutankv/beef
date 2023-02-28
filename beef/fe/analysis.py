'''
FE objects submodule: analysis definitions
'''

from copy import deepcopy as copy
import numpy as np
from beef import gdof_from_nodedof, compatibility_matrix, B_to_dofpairs, dof_pairs_to_Linv, lagrange_constrain, convert_dofs, convert_dofs_list, ensure_list, gdof_ix_from_nodelabels, basic_coupled, blkdiag
from scipy.linalg import block_diag, null_space as null, solve
from beef import newmark 
from beef.modal import normalize_phi, statespace, maxreal
from beef.newmark import is_converged, factors_from_alpha
import sys

if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm

class Analysis:
    '''
    Analysis definition class.

    Arguments
    -----------------
    eldef : obj
        BEEF ElementDefinition object
    forces : obj
        list of BEEF Force objects
    prescribed_N : fun
        function (time instance is input) returning list/array of axial forces ordered in the 
        same manner as the elements are ordered in the element definition
    prescribed_displacement : fun
        TODO - not finalized
    tmax : 1
        final time value
    dt : 1  
        time step
    itmax : 10
        maximum number of iterations in each time increment
    t0 : 0
        start time
    tol : dict
        dictionary specifying tolerance levels for 'u' (displacement) 
        and 'r' (force residual) standard is no tolerances
    nr_modified : False
        whether or not to use modified Newton-Raphson (for relevant nonlinear solvers)
    newmark_factors : {'beta': 0.25, 'gamma': 0.5}
        dictionary specifying values of beta and gamma (and optionally, alpha for hht)
    rayleigh : {'stiffness': 0, 'mass': 0}
        dictionary specifying stiffness and mass proportional damping
    tol_fun : np.linalg.norm
        function to apply for tolerance checks
    '''

    def __init__(self, eldef, forces=None, prescribed_N=None, prescribed_displacements=None, 
        tmax=1, dt=1, itmax=10, t0=0, tol=None, nr_modified=False, 
        newmark_factors={'beta': 0.25, 'gamma': 0.5}, rayleigh={'stiffness': 0, 'mass':0}, 
        tol_fun=np.linalg.norm):

        if forces is None:
            forces = []
        if prescribed_displacements is None:
            prescribed_displacements = []

        self.eldef = copy(eldef)  #create copy of part, avoid messing with original part definition

        self.forces = forces
        self.prescr_disp = prescribed_displacements
        self.t = np.arange(t0, tmax+dt, dt)
        self.itmax = itmax
        self.prescribed_N = prescribed_N

        if 'alpha' not in newmark_factors:
            newmark_factors['alpha'] = 0.0

        # TODO: treat prescribed displacements
        # self.dof_pairs = np.vstack([self.eldef.dof_pairs, self.get_dof_pairs_from_prescribed_displacements()])
        self.dof_pairs = self.eldef.dof_pairs
        self.Linv = dof_pairs_to_Linv(self.dof_pairs, len(self.eldef.nodes)*(self.eldef.dim-1)*3)
        
        if len(forces)!=0:
            min_dt = np.min(np.array([force.min_dt for force in self.forces+self.prescr_disp]))

            if len(self.t)==1:
                this_dt = np.nan
            else:
                this_dt = np.diff(self.t)[0]

            if (this_dt-min_dt)>np.finfo(np.float32).eps:
                print(f'A time increment ({this_dt}) larger than the lowest used for force definitions ({min_dt}) is specified. Interpret results with caution!')
  
        # Tolerance dictionary update (add only specified values, otherwise keep as None)
        tol0 = {'u': None, 'r': None}     
        if tol is None:
            tol = {}

        tol0.update(**tol)
        self.tol = tol0
        
        self.run_all_iterations = all(v is None for v in tol.values())
        self.newmark_factors = newmark_factors
        self.nr_modified = nr_modified
        self.rayleigh = rayleigh
        self.tol_fun = tol_fun


    def get_dof_pairs_from_prescribed_displacements(self):
        '''
        Get dof pairs list from specified prescribed displacements.
        
        Returns 
        -------------
        dof_pairs : int
            list of lists indicating dof_pairs

        '''
        prescr_ix = [np.hstack([self.eldef.node_dof_lookup(nl, dof_ix=dix)for nl, dix in zip(pd.node_labels, pd.dof_ix)]).flatten() for pd in self.prescr_disp]
        dof_pairs = np.vstack([[pi, None] for pi in prescr_ix])
        return dof_pairs
        

    def get_global_forces(self, t):  
        '''
        Get global force (FE format) at specified time instance.

        Arguments
        ----------
        t : float
            time instance from which to establish global forces

        Returns
        ----------
        glob_force : array
            numpy array with force vector stacked on global format (referring to node/dof stacking of global system)

        '''
        glob_force = np.zeros(self.eldef.ndofs)
        for force in self.forces:
            dof_ix = np.hstack([self.eldef.node_dof_lookup(nl, dof_ix=dix) for nl, dix in zip(force.node_labels, force.dof_ix)]).flatten()
            glob_force[dof_ix] += force.evaluate(t)
        
        return glob_force

    def get_global_prescribed_displacement(self, t):  
        glob_displacement = np.zeros(self.eldef.ndofs)
        dof_ix_full = []
        for pd in self.prescr_disp:
            dof_ix_add = np.hstack([self.eldef.node_dof_lookup(nl, dof_ix=dix) for nl, dix in zip(pd.node_labels, pd.dof_ix)]).flatten()
            glob_displacement[dof_ix_add] += pd.evaluate(t)
            dof_ix_full.append(dof_ix_add)

        if len(dof_ix_full) != 0:
            dof_ix_full = np.hstack(dof_ix_full)
            dof_ix = np.hstack([np.where(self.eldef.unconstrained_dofs == dof)[0] for dof in dof_ix_full])    # relative to unconstrained dofs

        return glob_displacement[dof_ix_full], dof_ix

    
    def get_global_force_history(self, t):
        '''
        Get global force (FE format) history (several time instances).

        Arguments
        ----------
        t : float
            time instances from which to establish global forces

        Returns
        ----------
        glob_force : array
            2d numpy array with force vector stacked on global format 
            (referring to node/dof stacking of global system), at specified
            time instances

        '''
        return np.vstack([self.get_global_forces(ti) for ti in t]).T


    def run_dynamic(self, print_progress=True, return_results=False):
        '''
        Run dynamic (nonlinear) solution, using parameters and element definition specified in parent Analysis object.

        Arguments
        --------------
        print_progress : True
            whether or not to inform user about progress while running
        return_results : True
            whether or not to output the displacement history established (self.u)

        Returns
        --------------
        self.u 
            is returned if specified by setting return_results to True
        '''

        # Retrieve constant defintions
        L = self.eldef.L
        Linv = self.Linv
        n_increments = len(self.t)

        # Assume at rest - fix later (take last increment form last step when including in BEEF module)       
        u = Linv @ np.zeros([self.eldef.ndofs])
        udot = Linv @ np.zeros([self.eldef.ndofs])
        self.u = np.ones([self.eldef.ndofs, len(self.t)])*np.nan
        self.u[:, 0] = L @ u
        beta, gamma, alpha = self.newmark_factors['beta'], self.newmark_factors['gamma'], self.newmark_factors['alpha']

        # Initial system matrices
        K = L.T @ self.eldef.k @ L
        M = L.T @ self.eldef.m @ L              
        C = L.T @ self.eldef.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M     

        # Get first force vector and estimate initial acceleration
        f = L.T @ self.get_global_forces(0)  #initial force, f0   

        f_int_prev = L.T @ self.eldef.q   
        uddot = newmark.acc_estimate(K, C, M, f, udot, f_int=f_int_prev, beta=beta, gamma=gamma, dt=(self.t[1]-self.t[0]))        

        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=n_increments-1, initial=0, desc='Dynamic analysis')        

        # Run through load INCREMENTS -->
        for k in range(n_increments-1):   
            # Time step load increment  
            dt = self.t[k+1] - self.t[k]

            # Increment force iterator object
            f_prev = 1.0 * f  # copy previous force level (used to scale residual for convergence check)         
            f = L.T @ self.get_global_forces(self.t[k+1]) # force in increment k+1
            
            if self.prescribed_N is not None:
                N = self.prescribed_N(self.t[k+1])
                for ix, el in enumerate(self.eldef.elements):
                    el.N0 = N[ix]
            
            df = f - f_prev    # force increment
            
            # Save "previous" values
            u_prev = 1.0*u
            udot_prev = 1.0*udot

            # Predictor step Newmark
            u, udot, uddot, du = newmark.pred(u, udot, uddot, dt)
            
            # Deform part
            self.eldef.deform(L @ u)    # deform nodes in part given by u => new f_int and K from elements
            du_inc = u*0

            # Calculate internal forces and residual force
            f_int = L.T @ self.eldef.q
            K = L.T @ self.eldef.k @ L
            C = L.T @ self.eldef.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M
            r = newmark.residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt)

            # Run through increment ITERATIONS -->
            for i in range(self.itmax):
                # Iteration, new displacement (Newton corrector step)
                u, udot, uddot, du = newmark.corr_alt(r, K, C, M, u, udot, uddot, dt, beta, gamma, alpha=alpha)
                du_inc += du

                # Update residual
                self.eldef.deform(L @ u)    # deform nodes in part given by u => new f_int and K from elements
                f_int = L.T @ self.eldef.q       # new internal (stiffness) force 
                
                r = newmark.residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt)

                # Check convergence
                converged = is_converged([self.tol_fun(du), self.tol_fun(r)], 
                                         [self.tol['u'], self.tol['r']], 
                                         scaling=[self.tol_fun(du_inc), self.tol_fun(df)])

                if not self.run_all_iterations and converged:
                    break
                
                # Assemble tangent stiffness, and damping matrices 
                if ~self.nr_modified:
                    K = L.T @ self.eldef.k @ L
                    C = L.T @ self.eldef.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M 
            
                # Update "previous" step values
                u_prev = 1.0*u
                udot_prev = 1.0*udot
                f_int_prev = 1.0*f_int
            
            self.u[:, k+1] = L @ u    # save to analysis time history

            # If all iterations are used
            if not self.run_all_iterations and (not converged):                    
                if print_progress:    
                    progress_bar.close()
                print(f'>> Not converged after {self.itmax} iterations on increment {k+1}. Response from iteration {i+1} saved. \n')
                if return_results:
                    return self.u
                else:
                    return
            else:
                if print_progress:
                    progress_bar.update(1)  # iterate on progress bar (adding 1)
                    
        if print_progress:    
            progress_bar.close()

        if return_results:
            return self.u


    def run_lin_dynamic(self, solver='full_hht', print_progress=True, return_results=False):
        '''
        Run dynamic (linear) solution, using parameters and element definition specified in parent Analysis object.

        Arguments
        --------------
        solver : {'full_hht', 'full', 'lin', 'lin_alt'}, optional
            what step-wise solver to enforce each time step, useful for debugging -
            corresponds to option input to ´newmark.newmark_lin´
        print_progress : True
            whether or not to inform user about progress while running
        return_results : True
            whether or not to output the displacement history established (self.u)

        Returns
        --------------
        self.u 
            is returned if specified by setting return_results to True
        '''

        # Retrieve constant defintions
        L = self.eldef.L
        Linv = self.Linv

        n_increments = len(self.t)

        # Assume at rest - fix later (take last increment form last step when including in BEEF module)       
        u0 = Linv @ np.zeros([self.eldef.ndofs])
        udot0 = Linv @ np.zeros([self.eldef.ndofs])
        beta, gamma, alpha = self.newmark_factors['beta'], self.newmark_factors['gamma'], self.newmark_factors['alpha'] 

        # System matrices and forces
        K = L.T @ self.eldef.k @ L
        M = L.T @ self.eldef.m @ L
        C = L.T @ self.eldef.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M
        f = np.zeros([K.shape[0], n_increments])           

        for k, tk in enumerate(self.t):    
            f[:, k] = L.T @ self.get_global_forces(tk)        # also enforce compatibility (L.T @ ...), each increment 

        # Run full linear Newmark
        u, __, __ = newmark.newmark_lin(K, C, M, f, self.t, u0, udot0, beta=beta, gamma=gamma, alpha=alpha, solver=solver)

        # Use compatibility relation to assign fixed DOFs as well
        self.u = np.zeros([self.eldef.ndofs, n_increments])
        for k in range(n_increments):
            self.u[:, k] = L @ u[:, k]

        # Deform part as end step
        self.eldef.deform(self.u[:,-1])
    
        if return_results:
            return self.u


    def run_lin_buckling(self, return_only_positive=True):
        '''
        Run static linearized buckling analysis, using parameters and element definition specified in parent Analysis object.

        Arguments
        --------------
        return_only_positive : True
            whether or not to return only positive half of eigenvalues

        Returns
        --------------
        lambd_b : float
            eigenvalues from buckling analysis        
        phi_b : float
            eigenvectors (stacked as columns) from buckling analysis
        '''

        from scipy.linalg import eig as speig

        # Retrieve constant defintions
        L = self.eldef.L

        # Substep 1: Establish geometric stiffness from linear analysis
        Ke = L.T @ self.eldef.k @ L
        f = L.T @ self.get_global_forces(self.t[-1])
        u = solve(Ke, f)

        self.eldef.deform_linear(L @ u)    # deform nodes in part given by u => new f_int and K from elements
        Kg = L.T @ self.eldef.get_kg(nonlinear=False) @ L     # get kg from axial forces generated in elements

        # Substep 2: Eigenvalue solution
        lambd_b, phi_b = speig(Ke, b=-Kg)

        lambd_b = lambd_b.real
        sort_ix = np.argsort(abs(lambd_b))
        lambd_b = lambd_b[sort_ix]
        phi_b = phi_b[:, sort_ix]

        if return_only_positive:
            phi_b = phi_b[:, lambd_b>0]
            lambd_b = lambd_b[lambd_b>0]

        phi_b = np.real(np.vstack([self.eldef.L @ phi_b[:, ix] for ix in range(0, len(lambd_b))]).T)

        return lambd_b, phi_b

        
    def run_eig(self, return_full=False, return_complex=False, normalize_modes=True):
        '''
        Run dynamic (state space) eigenvalue analysis, using parameters and 
        element definition specified in parent Analysis object.

        Arguments
        --------------
        return_full : False
            whether or not to return the raw result from state space eigenproblem - if False
            only displacements and one representation of each complex conjugate pair is returned
        return_complex : False
            whether or not to return modes as complex representations
        normalize_modes : True
            whether or not to normalize modes (ensure largest value is equal to 1)

        Returns
        --------------
        lambd : float
            eigenvalues from analysis        
        phi : float
            eigenvectors (stacked as columns) from analysis
        '''

        M, C, K, __ = self.eldef.global_element_matrices(constraint_type='primal')
        C = C + M*self.rayleigh['mass'] + K*self.rayleigh['stiffness']  # Rayleigh damping

        A = statespace(K, C, M)
        lambd, phi = np.linalg.eig(A)
        
        if normalize_modes:
            n_dofs = 6 if self.eldef.domain == '3d' else 3
            include_dofs = [0,1,2] if self.eldef.domain == '3d' else [0,1]
            phi = normalize_phi(phi, include_dofs=include_dofs, n_dofs=n_dofs)

        if return_full:
            return lambd, phi
        else:
            __, ix = np.unique(np.abs(lambd), return_index=True)
            n_dofs = M.shape[0]
            phi = self.eldef.L @ phi[:n_dofs, ix]
            lambd = lambd[ix]
            
                
            if ~return_complex:
                phi = maxreal(phi)
                phi = np.real(phi)

        return lambd, phi
    

    def run_static(self, print_progress=True, return_results=False):
        '''
        Run static (nonlinear) solution, using parameters and element 
        definition specified in parent Analysis object.

        Arguments
        --------------
        print_progress : True
            whether or not to inform user about progress while running
        return_results : True
            whether or not to output the displacement history established (self.u)

        Returns
        --------------
        self.u 
            is returned if specified by setting return_results to True
        '''
        # Retrieve constant defintions
        L = self.eldef.L
        n_increments = len(self.t)
        
        u = self.Linv @ np.zeros([self.eldef.ndofs])
        self.u = np.ones([self.eldef.ndofs, len(self.t), ])*np.nan
        self.u[:, 0] = L @ u

        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=(n_increments), initial=0, desc='Static analysis')  
        
        f = u * 0     # initialize with zero force

        for k, tk in enumerate(self.t):
            # Increment force iterator object
            f_prev = 1.0 * f    # copy previous force level (used to scale residual for convergence check)         
            f = L.T @ self.get_global_forces(tk)     # force in increment k  

            df = f - f_prev   # force increment
            
            # Deform part
            self.eldef.deform(L @ u)    # deform nodes in part given by u => new f_int and K from elements
            du_inc = u*0        # total displacement during increment
            
            # Calculate internal forces and residual force
            f_int = L.T @ self.eldef.q
            
            K = L.T @ self.eldef.k @ L
            r = f - f_int       # residual force

            # Iterations for each load increment 
            for i in range(0, self.itmax):
                # Iteration, new displacement (NR iteration)
                du = solve(K, r)

                u = u + du     # add to u, NR  
                du_inc = du_inc + du

                # Update residual
                self.eldef.deform(L @ u, du=L@du)        # deform nodes in part given by u => new f_int and K from elements
                f_int = L.T @ self.eldef.q      # new internal (stiffness) force
                r = f - f_int                   # residual force

                # Check convergence
                converged = is_converged([np.linalg.norm(du), np.linalg.norm(r)],
                                         [self.tol['u'], self.tol['r']], 
                                         scaling=[np.linalg.norm(du_inc), np.linalg.norm(f)])
                
                if not self.run_all_iterations and converged:
                    break

                # Assemble tangent stiffness if a new iteration is needed
                if ~self.nr_modified:
                    K = L.T @ self.eldef.k @ L

            self.u[:, k] = L @ u    # save to analysis time history

            # If not converged after all iterations
            if not self.run_all_iterations and (not converged):  
                print(f'>> Not converged after {self.itmax} iterations on increment {k+1}. Response from iteration {i+1} saved. \n')
                if print_progress:    
                    progress_bar.close()
                return
            else:
                if print_progress:
                    progress_bar.update(1)  # iterate on progress bar (adding 1)    

        if print_progress:    
            progress_bar.close()

        if return_results:
            return self.u


    def run_lin_static(self, print_progress=True, return_results=False):
        '''
        Run static (linear) solution, using parameters and element definition specified in parent Analysis object.

        Arguments
        --------------
        print_progress : True
            whether or not to inform user about progress while running
        return_results : True
            whether or not to output the displacement history established (self.u)

        Returns
        --------------
        self.u 
            is returned if specified by setting return_results to True
        '''
        # Retrieve constant defintions
        L = self.eldef.L
        n_increments = len(self.t)
        
        self.u = np.ones([self.eldef.ndofs, len(self.t)])*np.nan

        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=(n_increments), initial=0, desc='Static analysis')  
        
        K = L.T @ self.eldef.k @ L

        for k, tk in enumerate(self.t): 
            f = L.T @ self.get_global_forces(tk)     # force in increment k
            self.u[:, k] = L @ solve(K, f)    # save to analysis time history

            if print_progress:
                progress_bar.update(1)  # iterate on progress bar (adding 1)    
        
        self.eldef.deform_linear(self.u[:,-1])    # deform nodes in part given by u => new f_int and K from elements
        
        if print_progress:    
            progress_bar.close()

        if return_results:
            return self.u
