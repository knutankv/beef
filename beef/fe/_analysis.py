from . import *
from . import _plotters
from copy import deepcopy as copy

#%% Analysis class definition
class Analysis:
    def __init__(self, assembly, steps=None, constraint_type='lagrange'):
        self.eldef = copy(assembly) # keep a copy of the assembly - avoid tampering with original assembly
        self.steps = steps
        self.ready = False
        self.constraint_type = constraint_type
        #inheritance from previous steps not possible     
    
        for step in self.steps:
            step.analysis = self
            
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
        
