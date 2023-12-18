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
    Results definition class. 

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
    elements : int, optional
        list of elements to output (all are output as ordered in ElDef if not specified)
    nodes : int, optional
        list of nodes to output (all are output as ordered in ElDef if not specified)

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


    def __init__(self, analysis, element_results=[], node_results=[], fun_dict={}, elements=None, nodes=None):        
        self.analysis = copy(analysis)
        self.output = None
        self.element_results = element_results
        self.node_results = node_results
        self.element_cogs = np.array([el.get_cog() for el in self.analysis.eldef.elements])
        self.fun_dict = fun_dict

        self.elements = elements
        self.nodes = nodes
        
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
        
        if self.elements is None:
            elements = self.analysis.eldef.elements
        else:
            elements = self.analysis.eldef.get_elements(np.array(self.elements).astype(int)) # convert to list of element objects

        if self.nodes is None:
            nodes = self.analysis.eldef.nodes
        else:
            nodes = self.analysis.eldef.get_nodes(np.array(self.nodes).astype(int))    # convert to list of node objects

        self.output = dict()
        for key in self.element_results:
            self.output[key] = np.zeros([len(elements), len(self.analysis.t)])
        
        for key in self.node_results:
            self.output[key] = np.zeros([len(nodes), len(self.analysis.t)])
        
        for key in self.fun_dict:
            self.output[key] = [None]*len(self.analysis.t)
            
        # Initiate progress bar
        if print_progress:
            progress_bar = tqdm(total=len(self.analysis.t)-1, initial=0, desc='Post processing')        

        for k, ti in enumerate(self.analysis.t):
            self.analysis.eldef.reset_deformation()
            if nonlinear:
                self.analysis.eldef.deform(self.analysis.u[:, k], update_tangents=True)
            else:
                self.analysis.eldef.deform_linear(self.analysis.u[:, k])
            
            # Element results
            for out in list(self.element_results):
                self.output[out][:, k] = np.array([getattr(el, out) for el in elements])
                        
            for out in list(self.node_results):
                            self.output[out][:, k] = np.array([getattr(el, out) for el in nodes])
                        

            for key in self.fun_dict:
                self.output[key][k] = self.fun_dict[key](self.analysis)
            
            if print_progress:
                progress_bar.update(1)

    def combine_output(self, keys=['N', 'Qy', 'Qz', 'Mx', 'My', 'Mz']):
        '''
        Combine or interweave chosen outputs (must match in length - either only element results or only node results).

        Arguments
        ----------
        keys : ['N', 'Qy', 'Qz', 'Mx', 'My', 'Mz']
            list of keys to access from output dictionary of `Results` object

        Returns 
        ----------
        data : array
            numpy array with data interweaved 
            results from each position are given by a n_keys x n_samples subarray that is merged
            in a big array with all data

        '''
        data = np.zeros([len(keys)*len(self.output[keys[0]]), len(self.analysis.t)])
        for k, ti in enumerate(self.analysis.t):
            data[:, k] = np.vstack([self.output[key][:,k] for key in keys]).T.flatten()

        return data
    
    @staticmethod
    def get_modal_forces(analysis, phi, keys=['N', 'Qy', 'Qz', 'Mx', 'My', 'Mz'], elements=None, return_pos=True):
        '''
        Convenience method to establish modal forces (forces corresponding to modal displacements / mode shapes).

        Arguments
        ----------
        analysis : `Analysis` object
            analysis object to deform (copy is made)
        phi : float
            numpy array describing modal transformation matrix (mode shapes stacked column-wise), e.g.
            that resulting from the `run_eig` method of an `Analysis` object
        keys : ['N', 'Qy', 'Qz', 'Mx', 'My', 'Mz']
            list of keys to access from output dictionary of `Results` object
        elements : int
            list of elements to grab forces from, if not specified (None), all elements in order of `ElDef` are used
        return_pos : bool
            whether or not to also return position of elements returned (convenient for many purposes), as a numpy array
            with each element position on a row with coordinates (x and y or x, y and z) column-wise

        Returns
        ----------
        phi_forces : array
            numpy array with forces corresponding to specified mode shapes
        [pos] : array
            numpy array with each element position on a row with coordinates (x and y or x, y and z) column-wise
        '''

        analysis_modal = analysis.define_deformation_history(phi)
        results = Results(analysis_modal, element_results=keys, elements=elements)
        results.process()
        phi_forces = results.combine_output(keys=keys)

        if return_pos:
            pos = np.vstack([el.get_cog() for el in analysis.eldef.get_elements(elements)])
            return phi_forces, pos
        else:
            return phi_forces

    
    # CORE METHODS
    def __str__(self):
        return f'BEEF Results ({len(self.steps)} steps, {self.assembly} assembly)'

    def __repr__(self):
        return f'BEEF Results ({len(self.steps)} steps, {self.assembly} assembly)'

