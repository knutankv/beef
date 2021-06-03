from copy import deepcopy as copy
import numpy as np
from beef import gdof_from_nodedof, compatibility_matrix, B_to_dofpairs, dof_pairs_to_Linv, lagrange_constrain, convert_dofs, convert_dofs_list, ensure_list, gdof_ix_from_nodelabels, basic_coupled, blkdiag
from scipy.linalg import block_diag, null_space as null, solve
from beef import newmark
from beef.newmark import is_converged, factors_from_alpha
import sys

if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm
## OLD CODE HERE ##

#%% Analysis class definition
class Analysis:
    def __init__(self, eldef, steps=None, constraint_type='lagrange'):
        self.eldef = copy(eldef) # keep a copy of the assembly - avoid tampering with original assembly
        self.steps = steps
        self.ready = False
        self.constraint_type = constraint_type
        #inheritance from previous steps not possible     
    
        for step in self.steps:
            step.analysis = self

    # CORE METHODS
    def __str__(self):
        return f'BEEF Analysis ({len(self.steps)} steps, {self.eldef} element definition)'

    def __repr__(self):
        return f'BEEF Analysis ({len(self.steps)} steps, {self.eldef} element definition)'

    # USEFUL
    def prepare(self):
        print('Preparing analysis...')
        self.eldef.assemble()
        
        for step in self.steps:
            step.prepare()
        
        self.ready = True
        
        
    def plot(self, **kwargs):             
        return _plotters.plot_step_3d(self, **kwargs) 
    
    def run(self):
        if not self.ready:    
            self.prepare()
            
            
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('Analysis started {}'.format(now))
        
        for step_ix, step in enumerate(self.steps):
            print('Solving step {}: {}'.format((step_ix+1), (step.type.capitalize()+ ' step') ) )
            step.solve(self)
    
        self.create_node_results()
        
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('Analysis finalized {}'.format(now) )   
    
    def create_node_results(self):
        self.node_results = copy(self.eldef.nodes)
        for node in self.node_results:
            node.steps = [None]*len(self.steps)
        
        for step_ix, step in enumerate(self.steps):

            for node in self.node_results:
                dof_ix = self.eldef.node_label_to_dof_ix(node.label)
                node.steps[step_ix] = step.results['u'][dof_ix, :]
                
    def global_load(self, step):        #consider to redefine as Step method  
        #Not ready for n_dofs != 6
        
        all_node_labels = self.eldef.get_node_labels()
        g_load = np.zeros([len(all_node_labels)*6, 1])
        
        for load in step.loads:
            for nodeload in load.nodeloads:
                node_ix = np.where(all_node_labels == nodeload.node_label)[0]
                if nodeload.local:
                    T = self.eldef.local_node_csys(nodeload.node_label)
                else:
                    T = np.eye(6)
                    
                f_local = np.zeros([6,1])
                f_local[nodeload.dof_ix,0] = np.array(nodeload.amplitudes)               

                g_load[node_ix*6 + np.arange(0, 6), 0] += (T.T @ f_local)[:,0]
                
        if self.eldef.constraint_type == 'lagrange':
            g_load = np.vstack([g_load, np.zeros([self.eldef.constraint_dof_count(),1])])
            
        elif self.eldef.constraint_type == 'primal':
            g_load = self.eldef.L.T @ g_load

        return g_load
        

## New code placed here ##

class AnalysisCR:
    def __init__(self, eldef, forces=None, prescribed_displacements=None, tmax=1, dt=1, itmax=10, t0=0, tol=None, nr_modified=False, newmark_factors={'beta': 0.25, 'gamma': 0.5}, rayleigh={'stiffness': 0, 'mass':0}, outputs=['u'], tol_fun=np.linalg.norm):
        if forces is None:
            forces = []
        if prescribed_displacements is None:
            prescribed_displacements = []

        
        self.eldef = copy(eldef)  #create copy of part, avoid messing with original part definition
        
        self.forces = forces
        self.prescr_disp = prescribed_displacements
        self.t = np.arange(t0, tmax+dt, dt)
        self.itmax = itmax

        # Change later:
        # self.dof_pairs = np.vstack([self.eldef.dof_pairs, self.get_dof_pairs_from_prescribed_displacements()])
        self.dof_pairs = self.eldef.dof_pairs
        self.Linv = dof_pairs_to_Linv(self.dof_pairs, len(self.eldef.nodes)*3)
        
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
        self.outputs = outputs
        self.tol_fun = tol_fun


    def get_dof_pairs_from_prescribed_displacements(self):
        prescr_ix = [np.hstack([self.eldef.gdof_ix_from_nodelabels(nl, dix) for nl, dix in zip(pd.node_labels, pd.dof_ix)]).flatten() for pd in self.prescr_disp]
        dof_pairs = np.vstack([[pi, None] for pi in prescr_ix])
        return dof_pairs
        

    def get_global_forces(self, t):  
        glob_force = np.zeros(self.eldef.n_dofs)
        for force in self.forces:
            dof_ix = np.hstack([self.eldef.gdof_ix_from_nodelabels(nl, dix) for nl, dix in zip(force.node_labels, force.dof_ix)]).flatten()
            glob_force[dof_ix] += force.evaluate(t)
        
        return glob_force

    def get_global_prescribed_displacement(self, t):  
        glob_displacement = np.zeros(self.eldef.n_dofs)
        dof_ix_full = []
        for pd in self.prescr_disp:
            dof_ix_add = np.hstack([self.eldef.gdof_ix_from_nodelabels(nl, dix) for nl, dix in zip(pd.node_labels, pd.dof_ix)]).flatten()
            glob_displacement[dof_ix_add] += pd.evaluate(t)
            dof_ix_full.append(dof_ix_add)

        if len(dof_ix_full) != 0:
            dof_ix_full = np.hstack(dof_ix_full)
            dof_ix = np.hstack([np.where(self.eldef.unconstrained_dofs == dof)[0] for dof in dof_ix_full])    # relative to unconstrained dofs

        return glob_displacement[dof_ix_full], dof_ix

    
    def get_global_force_history(self, t):
        return np.vstack([self.get_global_forces(ti) for ti in t]).T



    def run_dynamic(self, print_progress=True, return_results=False):
        # Retrieve constant defintions
        L = self.eldef.L
        Linv = self.Linv
        n_increments = len(self.t)

        # Assume at rest - fix later (take last increment form last step when including in BEEF module)       
        u = Linv @ np.zeros([self.eldef.n_dofs])
        udot = Linv @ np.zeros([self.eldef.n_dofs])
        self.u = np.ones([self.eldef.n_dofs, len(self.t)])*np.nan
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


    def run_lin_dynamic(self, print_progress=True, solver='full_hht', return_results=False):
        # Retrieve constant defintions
        L = self.eldef.L
        Linv = self.Linv

        n_increments = len(self.t)

        # Assume at rest - fix later (take last increment form last step when including in BEEF module)       
        u0 = Linv @ np.zeros([self.eldef.n_dofs])
        udot0 = Linv @ np.zeros([self.eldef.n_dofs])
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
        self.u = np.zeros([self.eldef.n_dofs, n_increments])
        for k in range(n_increments):
            self.u[:, k] = L @ u[:, k]

        # Deform part as end step
        self.eldef.deform(self.u[:,-1])
    
        if return_results:
            return self.u


    def run_lin_buckling(self, return_only_positive=True):
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
        

    def run_static(self, print_progress=True, return_results=False):
        # Retrieve constant defintions
        L = self.eldef.L
        n_increments = len(self.t)
        
        u = self.Linv @ np.zeros([self.eldef.n_dofs])
        self.u = np.ones([self.eldef.n_dofs, len(self.t), ])*np.nan
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
                self.eldef.deform(L @ u)    # deform nodes in part given by u => new f_int and K from elements
                f_int = L.T @ self.eldef.q       # new internal (stiffness) force
                r = f - f_int                   # residual force

                # Check convergence
                converged = is_converged([np.linalg.norm(du), np.linalg.norm(r)], [self.tol['u'], self.tol['r']], scaling=[np.linalg.norm(du_inc), np.linalg.norm(df)])
                
                if not self.run_all_iterations and converged:
                    break

                # Assemble tangent stiffness if a new iteration is needed
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
        # Retrieve constant defintions
        L = self.eldef.L
        n_increments = len(self.t)
        
        self.u = np.ones([self.eldef.n_dofs, len(self.t)])*np.nan

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
