# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:52:18 2020

@author: knutankv
"""
import sys

# Corotated UL Beam FE
import numpy as np
from scipy.linalg import block_diag, null_space as null, solve
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import pdb

if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm

from knutils.tools import print_progress as pprogress, clear_progress
from beef import newmark
from beef.newmark import is_converged, factors_from_alpha
from beef import gdof_from_nodedof, compatibility_matrix, B_to_dofpairs, dof_pairs_to_Linv, lagrange_constrain, convert_dofs, convert_dofs_list, ensure_list, gdof_ix_from_nodelabels, basic_coupled
from scipy.interpolate import interp1d


class Force:
    def __init__(self, amplitudes, node_labels, dof_ix, t=None, force_type='force'):
        """
        Parameters
        -----------------
        amplitudes : double
            Numpy array where axis 1 corresponds to the DOFs (same size as nodelabels and dof_ix) and the second axis
            corresponds to the specified t. 

        Returns
        ------------------


        Notes
        --------------------
        If t is either not given (amplitude assumed constant) or a scalar (amplitude assumed ramped), length of second axis must be 1.
        """

        self.node_labels = node_labels       
        self.dof_ix = self.adjust_dof_ix(dof_ix, len(node_labels))
        amplitudes = self.adjust_amplitudes(amplitudes, len(node_labels))
        self.min_dt = np.inf
        self.type = force_type

        if t is None:
            self.evaluate = lambda __: amplitudes[:, 0]  # output constant force regardless
        else:
            if np.array(t).ndim==0:
                t = np.array([0, t])    # assume max time is specified (ramping)
                if amplitudes.shape[1] == 1:
                    amplitudes = np.hstack([amplitudes*0, amplitudes])
            else:
                self.min_dt = np.min(np.diff(t))

            if amplitudes.shape != tuple([len(node_labels), len(t)]):
                raise ValueError('Please fix form of amplitude input.')

            self.evaluate = interp1d(t, amplitudes, fill_value=amplitudes[:,0]*0, bounds_error=False)
        
        self.amplitudes = amplitudes
        

    @staticmethod
    def adjust_dof_ix(dix, n_nodes):
        if np.array(dix).ndim != 2:
            if np.array(dix).ndim == 0:
                dix = [[dix]]*n_nodes
            elif np.array(dix).ndim == 1:
                dix = [dix]*n_nodes

        return dix


    @staticmethod
    def adjust_amplitudes(amplitudes, n_nodes):

        if type(amplitudes) is list:
            amplitudes = np.array(amplitudes)
        elif type(amplitudes) in [float, int]:
            amplitudes = np.array([amplitudes])
        
        if amplitudes.ndim == 1:
            amplitudes = amplitudes[np.newaxis, :]

        if amplitudes.shape[0] == 1 and n_nodes>1:
            amplitudes = np.repeat(amplitudes, n_nodes, axis=0)

        return amplitudes


class PrescribedDisplacement(Force):
    def __init__(self, amplitudes, node_labels, dof_ix, t=None):
        """
        Parameters
        -----------------
        amplitudes : double
            Numpy array where axis 1 corresponds to the DOFs (same size as nodelabels and dof_ix) and the second axis
            corresponds to the specified t. 

        Returns
        ------------------

        Notes
        --------------------
        If t is either not given (amplitude assumed constant) or a scalar (amplitude assumed ramped), length of second axis must be 1.
        """

        super().__init__(amplitudes, node_labels, dof_ix, t=t, force_type='force')
       

class Results:
    def __init__(self, analysis, element_results=['M', 'N', 'V'], node_results=[]):
        self.analysis = copy(analysis)
        self.output = None
        self.element_results = element_results
        self.node_results = node_results
        self.element_cogs = np.array([el.get_cog() for el in self.analysis.part.elements])
        
    def process(self, print_progress=True):
        self.output = dict()
        for key in self.element_results:
            self.output[key] = np.zeros([len(self.analysis.part.elements), len(self.analysis.t)])
        
        for key in self.node_results:
            self.output[key] = np.zeros([len(self.analysis.part.nodes), len(self.analysis.t)])
            
        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=len(self.analysis.t)-1, initial=0, desc='Post processing')        

        for k, ti in enumerate(self.analysis.t):
            self.analysis.part.deform_part(self.analysis.u[:, k])
            for out in list(self.element_results):
                self.output[out][:, k] = np.array([el.extract_load_effect(out) for el in self.analysis.part.elements])


            if print_progress:
                progress_bar.update(1)  # iterate on progress bar (adding 1)
        
        if print_progress:    
            progress_bar.close()
            
        
class Analysis:
    def __init__(self, part, forces=None, prescribed_displacements=None, tmax=1, dt=1, itmax=10, t0=0, tol=None, nr_modified=False, newmark_factors={'beta': 0.25, 'gamma': 0.5}, rayleigh={'stiffness': 0, 'mass':0}, outputs=['u'], tol_fun=np.linalg.norm):
        if forces is None:
            forces = []
        if prescribed_displacements is None:
            prescribed_displacements = []

        self.part = copy(part)  #create copy of part, avoid messing with original part definition
        self.forces = forces
        self.prescr_disp = prescribed_displacements
        self.t = np.arange(t0, tmax+dt, dt)
        self.itmax = itmax
        # Change later:
        # self.dof_pairs = np.vstack([self.part.dof_pairs, self.get_dof_pairs_from_prescribed_displacements()])
        self.dof_pairs = self.part.dof_pairs
        
        self.B = compatibility_matrix(self.dof_pairs, len(self.part.nodes)*3)
        self.L = null(self.B) 
        self.Linv = dof_pairs_to_Linv(self.dof_pairs, len(self.part.nodes)*3)

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
        prescr_ix = [np.hstack([self.part.gdof_ix_from_nodelabels(nl, dix) for nl, dix in zip(pd.node_labels, pd.dof_ix)]).flatten() for pd in self.prescr_disp]
        dof_pairs = np.vstack([[pi, None] for pi in prescr_ix])
        return dof_pairs
        

    def get_global_forces(self, t):  
        n_dofs = len(self.part.nodes)*3 
        glob_force = np.zeros(n_dofs)
        for force in self.forces:
            dof_ix = np.hstack([self.part.gdof_ix_from_nodelabels(nl, dix) for nl, dix in zip(force.node_labels, force.dof_ix)]).flatten()
            glob_force[dof_ix] += force.evaluate(t)
        
        return glob_force

    def get_global_prescribed_displacement(self, t):  
        n_dofs = len(self.part.nodes)*3 
        glob_displacement = np.zeros(n_dofs)
        dof_ix_full = []
        for pd in self.prescr_disp:
            dof_ix_add = np.hstack([self.part.gdof_ix_from_nodelabels(nl, dix) for nl, dix in zip(pd.node_labels, pd.dof_ix)]).flatten()
            glob_displacement[dof_ix_add] += pd.evaluate(t)
            dof_ix_full.append(dof_ix_add)

        if len(dof_ix_full) != 0:
            dof_ix_full = np.hstack(dof_ix_full)
            dof_ix = np.hstack([np.where(self.part.unconstrained_dofs == dof)[0] for dof in dof_ix_full])    # relative to unconstrained dofs

        return glob_displacement[dof_ix_full], dof_ix

    
    def get_global_force_history(self, t):
        return np.vstack([self.get_global_forces(ti) for ti in t]).T


    def run_dynamic(self, print_progress=True, return_results=False):
        # Retrieve constant defintions
        L = self.L
        Linv = self.Linv
        n_increments = len(self.t)

        # Assume at rest - fix later (take last increment form last step when including in BEEF module)       
        u = Linv @ np.zeros([len(self.part.nodes)*3])
        udot = Linv @ np.zeros([len(self.part.nodes)*3])
        self.u = np.ones([len(self.part.nodes)*3, len(self.t)])*np.nan
        self.u[:, 0] = L @ u
        beta, gamma, alpha = self.newmark_factors['beta'], self.newmark_factors['gamma'], self.newmark_factors['alpha']

        # Initial system matrices
        K = L.T @ self.part.k @ L
        M = L.T @ self.part.m @ L
        C = L.T @ self.part.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M     
        
        # Get first force vector and estimate initial acceleration
        f = L.T @ self.get_global_forces(0)  #initial force, f0   
        # prescr_disp, prescr_disp_ix  = self.get_global_prescribed_displacement(0)
        
        f_int_prev = L.T @ self.part.q     
        uddot = newmark.acc_estimate(K, C, M, f, udot, f_int=f_int_prev, beta=beta, gamma=gamma, dt=(self.t[1]-self.t[0]))        

        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=n_increments-1, initial=0, desc='Dynamic analysis')        

        # Run through load increments
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

            # Increment displacement iterator object
            # if len(prescr_disp)>0:   
            #     prescr_disp, prescr_disp_ix = self.get_global_prescribed_displacement(self.t[k+1])
            #     u[prescr_disp_ix] = prescr_disp   #overwrite prescribed displacements

            # Deform part
            self.part.deform_part(L @ u)    # deform nodes in part given by u => new f_int and K from elements
            du_inc = u*0

            # Calculate internal forces and residual force
            f_int = L.T @ self.part.q
            K = L.T @ self.part.k @ L
            C = L.T @ self.part.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M
            r = newmark.residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt)

            # Iterations for each load increment 
            for i in range(self.itmax):
                # Iteration, new displacement (Newton corrector step)
                u, udot, uddot, du = newmark.corr_alt(r, K, C, M, u, udot, uddot, dt, beta, gamma, alpha=alpha)

                # if len(prescr_disp)>0:   
                #     u[prescr_disp_ix] = prescr_disp   #overwrite prescribed displacements
                #     du[prescr_disp_ix] = 0

                du_inc += du

                # Update residual
                self.part.deform_part(L @ u)    # deform nodes in part given by u => new f_int and K from elements
                f_int = L.T @ self.part.q       # new internal (stiffness) force 
                
                r = newmark.residual_hht(f, f_prev, f_int, f_int_prev, K, C, M, u_prev, udot, udot_prev, uddot, alpha, gamma, beta, dt)

                # Check convergence
                converged = is_converged([self.tol_fun(du), self.tol_fun(r)], 
                                         [self.tol['u'], self.tol['r']], 
                                         scaling=[self.tol_fun(du_inc), self.tol_fun(df)])

                if not self.run_all_iterations and converged:
                    break
                
                # Assemble tangent stiffness, and damping matrices
                K = L.T @ self.part.k @ L
                C = L.T @ self.part.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M 
            
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
        L = self.part.L
        n_increments = len(self.t)

        # Assume at rest - fix later (take last increment form last step when including in BEEF module)       
        u0 = self.part.Linv @ np.zeros([len(self.part.nodes)*3])
        udot0 = self.part.Linv @ np.zeros([len(self.part.nodes)*3])
        beta, gamma, alpha = self.newmark_factors['beta'], self.newmark_factors['gamma'], self.newmark_factors['alpha'] 

        # System matrices and forces
        K = L.T @ self.part.k @ L
        M = L.T @ self.part.m @ L
        C = L.T @ self.part.c @ L + self.rayleigh['stiffness']*K + self.rayleigh['mass']*M
        f = np.zeros([K.shape[0], n_increments])
        
        for k, tk in enumerate(self.t):    
            f[:, k] = L.T @ self.get_global_forces(tk)        # also enforce compatibility (L.T @ ...), each increment 

        # Run full linear Newmark
        u, __, __ = newmark.newmark_lin(K, C, M, f, self.t, u0, udot0, beta=beta, gamma=gamma, alpha=alpha, solver=solver)

        # Use compatibility relation to assign fixed DOFs as well
        self.u = np.zeros([self.part.k.shape[0], n_increments])
        for k in range(n_increments):
            self.u[:, k] = L @ u[:, k]

        # Deform part as end step
        self.part.deform_part(self.u[:,-1])
    
        if return_results:
            return self.u


    def run_lin_buckling(self):
        from scipy.linalg import eig as speig

        # Retrieve constant defintions
        L = self.part.L

        # Substep 1: Establish geometric stiffness from linear analysis
        Ke = L.T @ self.part.k @ L
        f = L.T @ self.get_global_forces(self.t[-1])
        u = solve(Ke, f)

        self.part.deform_part_linear(L @ u)    # deform nodes in part given by u => new f_int and K from elements
        Kg = L.T @ self.part.get_kg() @ L

        # Substep 2: Eigenvalue solution
        lambd_b, phi_b = speig(Ke, b=Kg)
        lambd_b = np.abs(lambd_b)
        sort_ix = np.argsort(lambd_b)
        lambd_b = lambd_b[sort_ix]
        phi_b = phi_b[:, sort_ix]

        phi_b = np.real(np.vstack([self.part.L @ phi_b[:, ix] for ix in range(0, len(lambd_b))]).T)

        return lambd_b, phi_b
        

    def run_static(self, print_progress=True, return_results=False):
        # Retrieve constant defintions
        L = self.part.L
        n_increments = len(self.t)
        
        u = self.part.Linv @ np.zeros([len(self.part.nodes)*3])
        self.u = np.ones([len(self.part.nodes)*3, len(self.t), ])*np.nan
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
            self.part.deform_part(L @ u)    # deform nodes in part given by u => new f_int and K from elements
            du_inc = u*0        # total displacement during increment

            # Calculate internal forces and residual force
            f_int = L.T @ self.part.q
            K = L.T @ self.part.k @ L
            r = f - f_int       # residual force

            # Iterations for each load increment 
            for i in range(0, self.itmax):
                # Iteration, new displacement (NR iteration)
                du = solve(K, r)
                u = u + du     # add to u, NR  
                du_inc = du_inc + du

                # Update residual
                self.part.deform_part(L @ u)    # deform nodes in part given by u => new f_int and K from elements
                f_int = L.T @ self.part.q       # new internal (stiffness) force
                r = f - f_int                   # residual force

                # Check convergence
                converged = is_converged([np.linalg.norm(du), np.linalg.norm(r)], [self.tol['u'], self.tol['r']], scaling=[np.linalg.norm(du_inc), np.linalg.norm(df)])
                
                if not self.run_all_iterations and converged:
                    break

                # Assemble tangent stiffness if a new iteration is needed
                K = L.T @ self.part.k @ L

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
        L = self.part.L
        n_increments = len(self.t)
        
        u = self.part.Linv @ np.zeros([len(self.part.nodes)*3])
        self.u = np.ones([len(self.part.nodes)*3, len(self.t)])*np.nan

        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=(n_increments), initial=0, desc='Static analysis')  
        
        K = L.T @ self.part.k @ L

        for k, tk in enumerate(self.t): 
            f = L.T @ self.get_global_forces(tk)     # force in increment k
            self.u[:, k] = L @ solve(K, f)    # save to analysis time history

            if print_progress:
                progress_bar.update(1)  # iterate on progress bar (adding 1)    
        
        self.part.deform_part_linear(L @ u)    # deform nodes in part given by u => new f_int and K from elements
        
        if print_progress:    
            progress_bar.close()

        if return_results:
            return self.u

        
class SectionProperties:
    def __init__(self, E=None, A=None, I=None, poisson=0.3, m=0):
        self.E = E
        self.A = A
        self.poisson = poisson
        self.I = I
        self.m = m
        
        if E is None:
            self.G = None
        else:
            self.G = E/(2*(1+poisson))
           
class Node:
    def __init__(self, nodelabel, x, y):
        self.label = int(nodelabel)
        self.coordinates = np.array([x,y])
        self.x0 = np.array([x,y,0])
        self.x = self.x0*1

class BeamElement2d:
    def __init__(self, nodes, label, properties=SectionProperties(), shear_flexible=False, force_psi=None, mass_formulation='constitutive_timoshenko'):
        self.nodes = nodes
        self.label = int(label)
        self.properties = properties
        self.L0 = self.get_length()
        self.phi0 = self.get_element_angle()
        self.tmat = np.eye(6)
        self.v = np.zeros(3)
        self.force_psi = force_psi
        self.shear_flexible = shear_flexible
        
        # Assign mass matrix function
        if mass_formulation not in ['timoshenko', 'euler', 'lumped', 'euler_trans']:
            raise ValueError("{timoshenko', 'euler', 'lumped', 'euler_trans'} are allowed values for mass_formulation. Please correct input.")
        elif mass_formulation is 'timoshenko':
            self.get_local_m = self.local_m_timo
        elif mass_formulation is 'euler':
            self.get_local_m = self.local_m_euler
        elif mass_formulation  is 'euler_trans':
            self.get_local_m = self.local_m_euler_trans
        elif mass_formulation is 'lumped':
            self.get_local_m = self.local_m_lumped
        
        self.update_all()
        
    def update_all(self):
        self.update_pos()                   # update all node positions and element geometry     
        self.update_corot()        # --> new internal forces (corotational)
        self.update_tangent_stiffness()     # --> new tangent stiffness and new mass
        self.update_element_mass()      

    def update_pos(self):
        self.L = self.get_length()
        self.e = self.get_e()
        self.tmat = self.get_tmat()   
        self.psi = self.get_shear_flexibility()

    def update_linear(self):
        self.update_v_lin()
        self.t = self.get_Kd_c() @ self.v              # new internal forces (element forces) based on the two above  
        self.N = self.t[0]                  # update internal force N from t
        self.M = self.t[1]
        self.Q = -2*self.t[2]/self.L        # update internal force Q from t   
        self.q = self.tmat.T @ self.S @ self.t  # calculate internal forces in global format

    def update_corot(self, linear=False):
        self.update_v()        # compute displacement mode
        Kd_c = self.get_Kd_c()

        self.t = Kd_c @ self.v              # new internal forces (element forces) based on the two above  

        self.N = self.t[0]                  # update internal force N from t
        self.M = self.t[1]
        self.Q = -2*self.t[2]/self.L        # update internal force Q from t   

        self.Kd = Kd_c + self.get_Kd_g()    # update modal Kd (geometric and constitutive part)

        self.S = self.get_S()               # connectivity, local to global
        self.Kr = self.get_Kr()             # local Kr

        self.q = self.tmat.T @ self.S @ self.t  # calculate internal forces in global format

    def update_tangent_stiffness(self):
        self.k = self.tmat.T @ self.get_local_k() @ self.tmat


    def get_shear_flexibility(self):  
        if not hasattr(self, 'force_psi') or self.force_psi is None:
            props = self.properties
            if (self.shear_flexible is False) or (props.G*props.A == 0):
                phi = 0
            else:
                phi = 12*props.E*props.I/(self.L**2*props.G*props.A)
            return 1/(1+phi)       
        else:
            return self.force_psi
    
        
    def get_cog(self):
        return np.average([node.x[0:2] for node in self.nodes], axis=0)
        
    
    def get_node_labels(self):
        return [node.label for node in self.nodes]
        
    
    def get_length(self):
        return np.linalg.norm(self.nodes[1].x[0:2] - self.nodes[0].x[0:2])
    
    
    def get_e(self):
        return (self.nodes[1].x[0:2] - self.nodes[0].x[0:2])/self.L
    
    
    def get_n(self):
        return np.array([-self.e[1], self.e[0]])


    def get_tmat(self):
        T = np.eye(6)
        T[0, :2] = self.e
        T[1, :2] = self.get_n()
        T[3, 3:5] = T[0, :2]
        T[4, 3:5] = T[1, :2]
        
        return T
    
       
    def get_local_k_alt(self):
        # Original version
        k_local = np.zeros([6,6])
        props = self.properties

        k_local[:3, :3] = (1/self.L**3) * np.array([[props.E*props.A*self.L**2,-self.Q*self.L**2, 0],
                                  [-self.Q*self.L**2, 12*self.psi*props.E*props.I+6/5*self.N*self.L**2, 6*self.psi*props.E*props.I*self.L+1/10*self.N*self.L**3],
                                  [0, 6*self.psi*props.E*props.I*self.L+1/10*self.N*self.L**3, (3*self.psi+1)*props.E*props.I*self.L**2+2/15*self.N*self.L**4]])

        k_local[3:, 3:] = (1/self.L**3) * np.array([[props.E*props.A*self.L**2,-self.Q*self.L**2,0],
                                  [-self.Q*self.L**2, 12*self.psi*props.E*props.I+6/5*self.N*self.L**2, -6*self.psi*props.E*props.I*self.L-1/10*self.N*self.L**3],
                                  [0, -6*self.psi*props.E*props.I*self.L-1/10*self.N*self.L**3, (3*self.psi+1)*props.E*props.I*self.L**2+2/15*self.N*self.L**4]])
        
        k_local[:3, 3:] = (1/self.L**3) * np.array([[-props.E*props.A*self.L**2,self.Q*self.L**2,0],
                                  [self.Q*self.L**2, -12*self.psi*props.E*props.I-6/5*self.N*self.L**2, 6*self.psi*props.E*props.I*self.L+1/10*self.N*self.L**3],
                                  [0, -6*self.psi*props.E*props.I*self.L-1/10*self.N*self.L**3, (3*self.psi-1)*props.E*props.I*self.L**2-1/30*self.N*self.L**4]])
        
        k_local[3:, :3] = k_local[0:3,3:].T
        
        return k_local
    
    
    def get_local_k(self):
        return self.S @ self.Kd @ self.S.T + self.Kr
           
    
    def update_v_lin(self):
        u1 = self.nodes[0].x-self.nodes[0].x0
        u2 = self.nodes[1].x-self.nodes[1].x0
        print(u1)
        print(u2)
        self.v[0] = u1[0]-u2[0]
        self.v[1] = (u1[2]-u2[2])/2     # symmetric angle
        self.v[2] = (u1[2]+u2[2])/2     # asymmetric angle

    def update_v(self):
        el_angle = self.get_element_angle()

        self.v[0] = self.L - self.L0
        self.v[1] = self.nodes[1].x[2] - self.nodes[0].x[2]

        phi_a = self.nodes[0].x[2] + self.nodes[1].x[2] - 2*(el_angle - self.phi0)  #asymmetric bending
        self.v[2] = ((phi_a + np.pi) % (2*np.pi)) - np.pi # % is the modulus operator, this ensures 0<phi_a<2pi


    def get_element_angle(self):
        x_a = self.nodes[0].x
        x_b = self.nodes[1].x       
            
        dx = (x_b - x_a)[0:2]
        el_ang = np.arctan2(dx[1],dx[0])
        
        return el_ang


    def get_Kd_c(self):
        props = self.properties
        Kd_c = 1/self.L * np.array([
            [props.E*props.A, 0, 0], 
            [0, props.E*props.I, 0],
            [0, 0, 3*self.psi*props.E*props.I]])
            
        return Kd_c
      
    def get_Kd_g(self):
        return self.L*self.N*np.array([[0,0,0], [0,1/12,0], [0,0, 1/20]])
        
    def get_Kr(self):
        Kr = np.zeros([6,6])
        Kr[0:3,0:3] = 1/self.L * np.array([[0, -self.Q, 0], [-self.Q, self.N, 0], [0, 0, 0]]) 
        Kr[0:3,3:] = -Kr[0:3,0:3]
        Kr[3:,0:3] = -Kr[0:3,0:3]
        Kr[3:,3:] = Kr[0:3,0:3]
        
        return Kr
    
    
    def get_S(self):
        return np.array([[-1,0,0], 
                         [0,0,2/self.L], 
                         [0,-1,1], 
                         [1,0,0], 
                         [0,0,-2/self.L], 
                         [0, 1, 1]])            
    
    def local_m_lumped(self):
        m = self.properties.m
        L = self.L0
        m_lumped = np.diag([m*L/2, m*L/2, m*L**2/4, m*L/2, m*L/2, m*L**2/4])
        
        return m_lumped
    
    def local_m_euler_trans(self):
        m_et = self.local_m_euler()
        m_et[np.ix_([2,5],[2,5])] = self.local_m_lumped()[np.ix_([2,5],[2,5])]
        
        return m_et
        
        
    
    def local_m_euler(self):
        m = self.properties.m
        L = self.L0
        
        return m*L/420 * np.array([
                               [140,          0,          0,          70,         0,          0        ],
                               [0,          156,        22*L,         0,         54,       -13*L    ],
                               [0,          22*L,         4*L**2,        0,         13*L,       -3*L**2    ],          
                               [70,          0,          0,          140,   0,          0        ],
                               [0,          54,       13*L,       0,         156,    -22*L      ],
                               [0,         -13*L,       -3*L**2,      0,        -22*L,     4*L**2   ]
                               ])
    
    def local_m_timo(self):
        rho = self.properties.m/self.properties.A
        I = self.properties.I
        L = self.L0
        A = self.properties.A
        psi = self.psi
        
        m_t = rho*A*L/(1+psi)**2 * np.array([[13/35+7/10*psi+1/3*psi**2, (11/210+11/210*psi+1/24*psi**2)*L, 9/70+3/10*psi+1/6*psi**2, -(13/420+3/40*psi+1/24*psi**2)*L],
                                             [0, (1/105+1/60*psi+1/20*psi**2)*L**2, (13/420+3/40*psi+1/24*psi**2)*L, -(1/140+1/60*psi+1/120*psi**2)*L**2],
                                             [0, 0, 13/35+7/10*psi+1/3*psi**2, (11/210+11/120*psi+1/24*psi**2)*L],
                                             [0,0,0,(1/105+1/60*psi+1/120*psi**2)*L**2]])
        m_t = m_t + m_t.T - np.diag(np.diag(m_t))   # copy values above diagonal to below diagonal
        
        m_r = rho*I/((1+psi**2)*L) * np.array([[6/5, (1/10-1/2*psi)*L,-6/5,(1/10-1/2*psi)*L],
                                               [0, (12/15+1/6*psi+1/3*psi**2)*L**2,(-1/10+1/2*psi)*L,-(1/30+1/6*psi-1/6*psi**2)*L**2],
                                               [0,0,6/5,(-1/10+1/2*psi)*L],
                                               [0,0,0,(2/15+1/6*psi+1/3*psi**2)*L**2]])
        
        m_r = m_r + m_r.T - np.diag(np.diag(m_r))   # copy values above diagonal to below diagonal
        
        m = np.zeros([6,6])
        m[np.ix_([1,2,4,5],[1,2,4,5])] = m_r + m_t
        m[np.ix_([0,3],[0,3])] = np.array([[2, 1], [1, 2]])*rho*A*L/6
        
        return m

    def update_element_mass(self):
        self.m = self.tmat.T @ self.get_local_m() @ self.tmat
    
    def get_local_kg(self, N):
        L = self.L0
        return np.array([
                    [0, 0, 0, 0, 0, 0],
                    [0, 36, 3*L, 0, -36, 3*L],
                    [0, 3*L, 4*L**2, 0, -3*L, -L**2],
                    [0, 0, 0, 0, 0, 0],
                    [0, -36, -3*L, 0, 36, -3*L],
                    [0, 3*L, -L**2, 0, -3*L, 4*L**2]
                ]) * N/(30*L)
        
        
    def get_kg(self, N=None, from_element_deformation=True):  # element level function (global DOFs)
        if N is None:
            N = self.N

        if from_element_deformation:
            return self.tmat.T @ self.S @ self.get_Kd_g() @ self.S.T @ self.tmat #from corotated formulation
        else:
            return self.tmat.T @ self.get_local_kg(N) @ self.tmat
    
    
    def extract_load_effect(self, load_effect):
        if load_effect is 'M':
            return (self.q[5]-self.q[2])/2
        elif load_effect is 'V':
            return (self.q[4]-self.q[1])/2
        elif load_effect is 'N':
            return self.N

class Feature:
    def __init__(self, matrix_type, node_labels, dof_ixs, value, local=False):
        
        if len(node_labels) == 1 or node_labels[1]==None:
            matrix = basic_coupled()[0:1, 0:1]*value
            node_labels = [node_labels[0]]
            dof_ixs = [dof_ixs[0]]
        else:
            matrix = basic_coupled()*value
        
        self.type = matrix_type
        self.node_labels = node_labels
        self.dof_ixs = dof_ixs
        self.matrix = matrix
        self.local = local
        
class Spring(Feature):
    def __init__(self, node_labels, dof_ixs, k):
        super().__init__('k', node_labels, dof_ixs, k)

class Dashpot(Feature):    
    def __init__(self, node_labels, dof_ixs, c):
        super().__init__('c', node_labels, dof_ixs, c)

class PointMass(Feature):    
    def __init__(self, node_label, dof_ixs, m):
        super().__init__('m', node_label, dof_ixs, m)


class Part:
    def __init__(self, node_matrix, element_matrix, properties, constraints, features=None, constraint_type='primal', shear_flexible=False, force_psi=None, mass_formulation='timoshenko'):
       
        # Initialization
        n_els = element_matrix.shape[0]
        n_nodes = node_matrix.shape[0]
        self.elements = [None]*n_els
        self.nodes = [None]*n_nodes
        self.node_labels = node_matrix[:,0]

        if node_matrix.shape[1] == 3:
            self.domain = '2d'
            z = np.zeros(n_nodes)
        else:
            self.domain = '3d'
            z = node_matrix[:,2]

        # Constraint definitions
        self.constraints = constraints
        self.constraint_type = constraint_type
        self.dof_pairs = self.constraint_dof_ix()
        self.n_constraints = self.dof_pairs.shape[0]
        self.B = compatibility_matrix(self.dof_pairs, len(self.nodes)*3)
        self.L = null(self.B) 
        self.Linv = dof_pairs_to_Linv(self.dof_pairs, len(self.nodes)*3)
        
        self.constrained_dofs = self.dof_pairs[self.dof_pairs[:,1]==None, 0]
        self.unconstrained_dofs = np.delete(np.arange(0, np.shape(self.B)[1]), self.constrained_dofs)
                
        # Properties treatment (copy properties if not list)
        if type(properties) is not list:
            properties = [properties] * n_els

        # Create node objects    
        for n_ix in range(0, n_nodes):
            self.nodes[n_ix] = Node(node_matrix[n_ix, 0], node_matrix[n_ix, 1], z[n_ix])

        # Assign global indices to all node objects        
        self.assign_global_ix_to_nodes()
        
        # Create element objects
        for el_ix in range(0, n_els):
            el_node_labels = element_matrix[el_ix, 1:]
            ix1 = np.where(self.node_labels==el_node_labels[0])[0][0]
            ix2 = np.where(self.node_labels==el_node_labels[1])[0][0]
            
            self.elements[el_ix] = BeamElement2d([self.nodes[ix1], self.nodes[ix2]], element_matrix[el_ix, 0], properties=properties[el_ix], shear_flexible=shear_flexible, force_psi=force_psi, mass_formulation=mass_formulation)
            
            dof_ix1 = self.node_label_to_dof_ix(el_node_labels[0])
            dof_ix2 = self.node_label_to_dof_ix(el_node_labels[1])
            
            self.elements[el_ix].dof_ix = np.hstack([dof_ix1, dof_ix2])

        # Establish matrices from features
        if features is None:
            features = []
        self.features = features
        self.feature_mats = self.global_matrices_from_features()

        # Update global matrices and vectors
        self.update_tangent_stiffness()
        self.update_internal_forces()
        self.update_mass_matrix()
        self.c = self.feature_mats['c']


    def global_matrices_from_features(self):
        n_dofs = np.shape(self.B)[1]
        feature_mats = dict(k=np.zeros([n_dofs, n_dofs]), 
                            c=np.zeros([n_dofs, n_dofs]), 
                            m=np.zeros([n_dofs, n_dofs]))

        for feature in self.features:
            ixs = np.array([self.node_dof_lookup(node_label, dof_ix) for node_label, dof_ix in zip(feature.node_labels, feature.dof_ixs)])      
            feature_mats[feature.type][np.ix_(ixs, ixs)] = feature.matrix

        return feature_mats
        

    def node_dof_lookup(self, node_label, dof_ix):
        return self.nodes[np.where(self.node_labels == node_label)[0][0]].global_dof_ixs[dof_ix]


    def assign_global_ix_to_nodes(self):
        for node in self.nodes:
            node.global_dof_ixs = self.gdof_ix_from_nodelabels(node.label, dof_ix=[0,1,2])
               
    def get_kg(self, N=None, from_element_deformation=True):       # part level function
        node_labels = self.node_labels
        ndim = len(node_labels)*3
        kg = np.zeros([ndim, ndim])        

        for el in self.elements:
            kg[np.ix_(el.dof_ix, el.dof_ix)] += el.get_kg(N=N, from_element_deformation=from_element_deformation)

        return kg   


    def update_mass_matrix(self):
        node_labels = self.node_labels
        ndim = len(node_labels)*3
        self.m = self.feature_mats['m']*1   

        for el in self.elements:
            self.m[np.ix_(el.dof_ix, el.dof_ix)] += el.m
            
        
    def update_tangent_stiffness(self):
        node_labels = self.node_labels
        ndim = len(node_labels)*3
        self.k = self.feature_mats['k']*1          
        
        for el in self.elements:
            self.k[np.ix_(el.dof_ix, el.dof_ix)] += el.k


    def update_internal_forces(self, u=None):       # on part level
        node_labels = self.node_labels
        ndim = len(node_labels)*3
        if u is None:
            u = np.zeros([ndim])

        if hasattr(self, 'feature_mats'):
            self.q = self.feature_mats['k'] @ u   
        else:
            self.q = u*0

        for el_ix, el in enumerate(self.elements):
            node_ix1 = np.where(node_labels == el.nodes[0].label)[0][0]
            node_ix2 = np.where(node_labels == el.nodes[1].label)[0][0]
            ixs = np.hstack([node_ix1*3+np.arange(0,3), node_ix2*3+np.arange(0,3)])

            self.q[ixs] += el.q

    def lagrange_constrain(self, mat):
        n_constraints = self.B.shape[0]
        O = np.zeros([n_constraints, n_constraints])
        return np.vstack([np.hstack([mat, self.B.T]), np.hstack([self.B, O])])

    
    def node_label_to_dof_ix(self, node_label): 
        node_ix = self.node_label_to_node_ix(node_label)   
        dof_ix = 3*node_ix + np.arange(0, 3)
        return dof_ix


    def node_label_to_node_ix(self, node_label):
        return np.where(self.node_labels==node_label)[0][0].astype(int)
    
    
    def constraint_dof_ix(self):        
            if self.constraints is None:
                raise ValueError("Can't output constraint DOF indices as no constraints are given.")
 
            c_dof_ix = []
            
            for constraint in self.constraints:   
                for node_constraint in constraint.node_constraints:
                    dofs = np.array(node_constraint.dof_ix)
                    dof_ixs = self.gdof_ix_from_nodelabels(node_constraint.master_node)[dofs]
                    
                    if node_constraint.slave_node is not None:
                        conn_dof_ixs = self.gdof_ix_from_nodelabels(node_constraint.slave_node)[dofs]
                    else:
                        conn_dof_ixs = [None]*len(dof_ixs)
    
                    dof_ixs = np.vstack([dof_ixs, conn_dof_ixs]).T
                    c_dof_ix.append(dof_ixs)
                    
            c_dof_ix = np.vstack(c_dof_ix)
            return c_dof_ix
    
    
    def plot(self, u=None, color='Gray', plot_nodes=False, node_labels=False, element_labels=False, ax=None):
        
        if ax is None:
            ax = plt.gca()
        
        dxy = np.zeros([2, 2])
        
        for el in self.elements:
            xy = np.vstack([node.x for node in el.nodes])

            if u is not None:
                ix1 = np.where(self.node_labels==el.nodes[0].label)[0][0]
                ix2 = np.where(self.node_labels==el.nodes[1].label)[0][0]
                
                dxy[0, :] = u[gdof_from_nodedof(ix1, [0,1]), 0]
                dxy[1, :] = u[gdof_from_nodedof(ix2, [0,1]), 0]

            
            plt.plot(xy[:,0]+dxy[:,0], xy[:,1]+dxy[:,1], color=color)
            
            if plot_nodes:
                plt.plot(xy[:,0], xy[:,1], linestyle='none', marker='.', color='Black')
                
            if element_labels:
                plt.text(el.get_cog()[0], el.get_cog()[1], el.label, color='DarkOrange')
        
        if node_labels:
            for node in self.nodes:
                plt.text(node.x[0], node.x[1], node.label, color='Black')

        return ax
    

    def gdof_ix_from_nodelabels(self, node_labels, dof_ix=[0,1,2]):   # copy general function from beef
        return gdof_ix_from_nodelabels(self.node_labels, node_labels, dof_ix=dof_ix)

    
    def deform_part_linear(self, u):
        for node in self.nodes:
            node.x = node.x0 + u[node.global_dof_ixs]

        for element in self.elements:
            element.update_linear()    

        self.update_internal_forces(u)   # update part level q (elements are not affected by this)
    
    
    def deform_part(self, u):
        for node in self.nodes:
            node.x = node.x0 + u[node.global_dof_ixs]

        for element in self.elements:
            element.update_all()
            
        self.update_tangent_stiffness()
        self.update_internal_forces(u)   
        self.update_mass_matrix()   
        


    def node_from_gdof(self, gdof_ix, n_dofs=3):
        ix = int(np.floor(gdof_ix/n_dofs))
        loc_ix = int(gdof_ix - ix*n_dofs)
        return self.nodes[ix], loc_ix
            
#%% Constraint class definition
class Constraint:
    def __init__(self, master_nodes, slave_nodes=None, dofs='all', relative_to='global'):
        
        dofs = convert_dofs_list(dofs, len(master_nodes), node_type='beam2d')
        if slave_nodes is None:
            self.type = 'node-to-ground'
        else:
            self.type = 'node-to-node'
            
        self.node_constraints = [None]*len(master_nodes)
        self.relative_to = relative_to
        
        if self.relative_to != 'global':
            raise ValueError("Only 'global' constraints supported currently, specified with variable 'relative_to'")
        
        for ix, master_node in enumerate(master_nodes):
            if self.type == 'node-to-ground':
                self.node_constraints[ix] = NodeConstraint(master_node, dofs[ix], None, relative_to)
            else:
                self.node_constraints[ix] = NodeConstraint(master_node, dofs[ix], slave_nodes[ix], relative_to)

class NodeConstraint:
    def __init__(self, master_node, dof_ix, slave_node, relative_to):
        self.slave_node = slave_node
        self.master_node = master_node
        self.dof_ix = dof_ix
        self.relative_to = relative_to




