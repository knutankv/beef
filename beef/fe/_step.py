from . import *

#%% Step class definitions
class Step:
    def __init__(self, step_type, initial_state_from=None, loads=None):
        self.type = step_type    
        self.loads = loads
        self.initial_state_from = initial_state_from
        self.results = dict()
        self.global_loads = None


    # CORE METHODS
    def __str__(self):
        return f'BEEF Step'

    def __repr__(self):
        return f'BEEF Step'


    def prepare(self):
        if self.loads is not None:
            self.global_loads = self.analysis.global_load(self)


    def adjust_response_final(self, r_in, as_type=float):
        if self.analysis.eldef.constraint_type == 'primal':
            r_out = self.analysis.eldef.L @ r_in
                
        elif self.analysis.eldef.constraint_type == 'lagrange':
            r_out = r_in[:-self.analysis.eldef.constraint_dof_count(), :] 
            
        return r_out
    
class EigenvalueStep(Step):
    def __init__(self, initial_state_from=None, n_modes=None, keep_constraint_modes=False, normalize=True):
        super().__init__('eigenvalue problem', initial_state_from=initial_state_from)
        self.n_modes = n_modes
        self.keep_constraint_modes = keep_constraint_modes
        self.normalize = normalize
        
    def solve(self, analysis):
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
        K = analysis.eldef.global_matrices['K']
        u = (np.linalg.inv(K) @ self.global_loads) 
        self.results['u'] = self.adjust_response_final(u) 
        self.results['R'] = -self.analysis.eldef.B.T @ u[-self.analysis.eldef.constraint_dof_count():, :] 