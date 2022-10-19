'''
FE objects submodule: step definition (**NOT USED CURRENTLY**)
'''

import numpy as np
class Step:
    '''
    Step class. (**NOT USED CURRENTLY**)

    Arguments
    ------------
    step_type : {'eigenvalue', 'dynamic', 'static'}
        string describing step type
    initial_state_from : None, optional
        name of step to base analysis on - this would be defined as initial state of the current step
    forces : Force obj
        list of force objects relevant in step
    '''
    def __init__(self, step_type, initial_state_from=None, forces=None):
        self.type = step_type    
        self.forces = forces
        self.initial_state_from = initial_state_from
        self.results = dict()
        self.global_forces = None


    # CORE METHODS
    def __str__(self):
        return f'BEEF Step'

    def __repr__(self):
        return f'BEEF Step'


    def prepare(self):
        '''
        Prepare step (not finalized)
        '''
        if self.forces is not None:
            self.global_forces = self.analysis.global_forces(self)


    def adjust_response_final(self, r_in, as_type=float):
        '''
        Adjust format of response to constrained.

        Arguments
        -----------
        r_in : float
        as_type : type
            float type is default

        Returns
        ----------
        r_out : float
            numpy array describing adjusted response (in constrained format)
        '''
        if self.analysis.eldef.constraint_type == 'primal':
            r_out = self.analysis.eldef.L @ r_in
                
        elif self.analysis.eldef.constraint_type == 'lagrange':
            r_out = r_in[:-self.analysis.eldef.constraint_dof_count(), :] 
            
        return r_out
    
class EigenvalueStep(Step):
    '''
    Eigenvalue step class.

    Arguments
    -------------
    initial_state_from : None, optional
        name of step to base analysis on - this would be defined as initial state of the current step    
    n_modes : None
        number of modes to solve - if None is input, all modes are solved
    keep_constraint_modes : False
        whether or not to keep constraint modes (Lagrange constraints)
    normalize : True
        whether or not to normalize modes prior to output
    '''
    def __init__(self, initial_state_from=None, n_modes=None, keep_constraint_modes=False, normalize=True):
        super().__init__('eigenvalue', initial_state_from=initial_state_from)
        self.n_modes = n_modes
        self.keep_constraint_modes = keep_constraint_modes
        self.normalize = normalize
        
    def solve(self, analysis):
        '''
        Run solution of eigenvalue step.

        Arguments
        ------------
        analysis 
            **SHOULD BE THE OTHER WAY AROUND - STEPS ARE NESTED IN LIST OF ANALYSIS***
            then solution methods should be part of analysis object, and the specified steps are chosen as input (all standard)


        '''
        mats = analysis.eldef.global_matrices
        
        if self.n_modes is None:
            lambd, phi = eig(np.linalg.inv(mats['M']) @ mats['K'])
        else:
            lambd, phi = eigsh(np.linalg.inv(mats['M']) @ mats['K'], self.n_modes)
        
        sort_ix = np.argsort(lambd)
        lambd = lambd[sort_ix]
        phi = phi[:, sort_ix]
        
        self.results['lambda'] = lambd
        self.results['u'] = self.adjust_response_final(phi, as_type=complex)    
        
        # if not self.keep_constraint_modes:
        #     self.results['lambda'], self.results['u'] = get_phys_modes(self.results['u'], mats['B'], lambd=lambd)
        
        if self.normalize:
            self.results['u'],__ = normalize_phi(self.results['u'], include_dofs=[0,1,2])
            
class StaticStep(Step):
    def __init__(self, **kwargs):
        super().__init__('static', **kwargs)

    def solve(self, analysis):

        '''
        Run solution of static step.

        Arguments
        ------------
        analysis 
            **SHOULD BE THE OTHER WAY AROUND - STEPS ARE NESTED IN LIST OF ANALYSIS***
            then solution methods should be part of analysis object, and the specified steps are chosen as input (all standard)

        '''
        
        K = analysis.eldef.global_matrices['K']
        u = (np.linalg.inv(K) @ self.global_forces) 
        self.results['u'] = self.adjust_response_final(u) 
        self.results['R'] = -self.analysis.eldef.B.T @ u[-self.analysis.eldef.constraint_dof_count():, :] 

class DynamicStep(Step):
    pass
    