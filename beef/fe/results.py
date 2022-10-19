'''
FE objects submodule: results and post-processing definitions
'''

import numpy as np
from copy import deepcopy as copy
import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm

class Results:
    '''
    Force definition class. 

    Arguments
    -------------
    analysis : Analysis obj
        `Analysis` object to post-process
    element_results : str, optional
        list of requested element results ('M', 'N' and 'V' currently available)
    node_results : [], optional
        list of requested node results (no standard values available yet)
    fun_dict : dict, optional
        dictionary


    Example
    -------------
    The usage of `fun_dict` is exemplified using an moving state space eigenvalue solution. First, 
    a custom function returning a function acting on the analysis object is required (the returned function is required to have 
    the `Analysis` object as its only input). The function used in this example returns the eigenvalues of the analysis:
        
        def fun_ss_eigs():
            def f(analysis):
                A = analysis.eldef.get_state_matrix()
                eigs,__ = np.linalg.eig(A)
                return eigs
                
            return f
    
    Thereafter, the `fun_dict` can be passed as a variable into the Results constructor function as follows:

        fun_dict = {'stab_eigs': fun_ss_eigs()}
        results = fe.Results(analysis, fun_dict=fun_dict)
        results.process()

    The resulting output in results will have the same structure, i.e., the eigenvalues can be accessed from `results.output['stab_eigs']`.

    '''


    def __init__(self, analysis, element_results=['M', 'N', 'V'], node_results=[], fun_dict={}):        
        self.analysis = copy(analysis)
        self.output = None
        self.element_results = element_results
        self.node_results = node_results
        self.element_cogs = np.array([el.get_cog() for el in self.analysis.eldef.elements])
        self.fun_dict = fun_dict
        
    def process(self, print_progress=True, nonlinear=True):
        ''' 
        Process the requested results.

        Arguments
        ----------
        print_progress : True, optional
            whether or not progress should be printed to terminal
        nonlinear : True, optional
            whether or not the tangents of the stiffness matrix should be modified throughout the
            post-processing

        Notes
        -------------
        The resulting output from the `results` object (instance of `Results` class) is stored in `results.output['stab_eigs']`.

        '''
        
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

