import numpy as np
from copy import deepcopy as copy
import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm

class Results:
    def __init__(self, analysis, element_results=['M', 'N', 'V'], node_results=[], fun_dict={}):
        # fun_dict uses analysis level as expected input
        
        self.analysis = copy(analysis)
        self.output = None
        self.element_results = element_results
        self.node_results = node_results
        self.element_cogs = np.array([el.get_cog() for el in self.analysis.eldef.elements])
        self.fun_dict = fun_dict
        
    def process(self, print_progress=True, nonlinear=True):
        self.output = dict()
        for key in self.element_results:
            self.output[key] = np.zeros([len(self.analysis.eldef.elements), len(self.analysis.t)])
        
        for key in self.node_results:
            self.output[key] = np.zeros([len(self.analysis.eldef.nodes), len(self.analysis.t)])
        
        for key in self.fun_dict:
            self.output[key] = [None]*len(self.analysis.t)
            
        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=len(self.analysis.t)-1, initial=0, desc='Post processing')        

        for k, ti in enumerate(self.analysis.t):
            if nonlinear:
                self.analysis.eldef.deform(self.analysis.u[:, k], update_tangents=True)
            else:
                self.analysis.eldef.deform_linear(self.analysis.u[:, k])

            for out in list(self.element_results):
                self.output[out][:, k] = np.array([el.extract_load_effect(out) for el in self.analysis.eldef.elements])
            
            for key in self.fun_dict:
                self.output[key][k] = self.fun_dict[key](self.analysis)
            
            if print_progress:
                progress_bar.update(1)
                
    # CORE METHODS
    def __str__(self):
        return f'BEEF Results ({len(self.steps)} steps, {self.assembly} assembly)'

    def __repr__(self):
        return f'BEEF Results ({len(self.steps)} steps, {self.assembly} assembly)'

