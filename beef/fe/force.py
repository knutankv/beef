'''
FE objects submodule: force definitions
'''

from scipy.interpolate import interp1d
import numpy as np

class Force:
    '''
    Force definition class. 

    Arguments
    -------------
    node_labels : int
        list of node labels to excite
    dofs : int
        list/array of local DOFs to excite
    amplitudes : float
        array defining amplitudes; shape should be either n_nodelabels x n_samples (length of time axis) or n_samples x 1 in dimensions - if latter
        the same force is assumed for all node labels
    name : 'Force-0', optional
        name of force object
    plotcolor : 'DarkOrange', optional
        color used for plotting of analysis with force definitions
    t : None, optional
        time axis corresponding to input force history (n_samples)

    Notes 
    ------
    Generates interpolant object that can be evaluated at time instance to generate force vector in global format, i.e., `self.evaluate`

    '''

    def __init__(self, node_labels, dofs, amplitudes, name='Force-0', plotcolor='DarkOrange', t=None):
        self.plotcolor = plotcolor
        self.name = name
        self.dof_ix = self.adjust_dof_ix(dofs, len(node_labels))
        amplitudes = self.adjust_amplitudes(amplitudes, len(node_labels))

        self.min_dt = np.inf
        self.t = t
        self.node_labels = node_labels

        if t is None:
            self.evaluate = lambda __: amplitudes[:, 0]  # output constant force regardless
        else:
            if np.array(t).ndim==0:
                t = np.array([0, t])    # assume max time is specified (linear ramp)
                if amplitudes.shape[1] == 1:
                    amplitudes = np.hstack([amplitudes*0, amplitudes])
            else:
                self.min_dt = np.min(np.diff(t))

            if amplitudes.shape != tuple([len(node_labels), len(t)]):
                raise ValueError('Please fix form of amplitude input. It should be either n_nodelabels x n_samples or n_samples x 1 in dimensions.')

            self.evaluate = interp1d(t, amplitudes, fill_value=amplitudes[:, 0]*0, bounds_error=False)
            self.amplitudes = amplitudes

    # CORE METHODS
    def __str__(self):
        return f'BEEF Force: {self.name}'

    def __repr__(self):
        return f'BEEF Force: {self.name}'

    @staticmethod
    def adjust_dof_ix(dix, n_nodes):
        '''
        Adjust DOF index format.
        '''
        if np.array(dix).ndim != 2:
            if np.array(dix).ndim == 0:
                dix = [[dix]]*n_nodes
            elif np.array(dix).ndim == 1:
                dix = [dix]*n_nodes

        return dix

    @staticmethod
    def adjust_amplitudes(amplitudes, n_nodes):
        '''
        Adjust amplitudes array to match the number of nodes.

        Arguments
        -------------
        amplitudes : float 
            array defining amplitudes; shape should be either n_nodelabels x n_samples (length of time axis) or 
            n_samples x 1 in dimensions - if latter the same force is assumed for all node labels
        n_nodes : int
            number of nodes

        Returns
        -------------
        amplitudes : float
            adjusted array defining amplitudes, corrected to match the required format to generate the force interpolant callable with `Force.evaluate`.

        '''

        if type(amplitudes) is list:
            amplitudes = np.array(amplitudes)
        elif type(amplitudes) in [float, int]:
            amplitudes = np.array([amplitudes])
        
        if amplitudes.ndim == 1:
            amplitudes = amplitudes[np.newaxis, :]

        if amplitudes.shape[0] == 1 and n_nodes>1:
            amplitudes = np.repeat(amplitudes, n_nodes, axis=0)

        return amplitudes