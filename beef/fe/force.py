from . import *
from scipy.interpolate import interp1d

#%% Load class definition 
class Force:
    def __init__(self, node_labels, dofs, amplitudes, name='Force-0', plotcolor='DarkOrange'):
        self.plotcolor = plotcolor
        self.name = name
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

            self.evaluate = interp1d(t, amplitudes, fill_value=amplitudes[:, 0]*0, bounds_error=False)
            self.amplitudes = amplitudes

    # CORE METHODS
    def __str__(self):
        return f'BEEF Force: {self.name}'

    def __repr__(self):
        return f'BEEF Force: {self.name}'

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


class NodeForce:
    def __init__(self, node_label, dof_ix, amplitudes, local):
        
        if type(dof_ix) is not list:
            dof_ix = [dof_ix]
        if type(amplitudes) is not list:
            amplitudes = [amplitudes]
            
        self.node_label = node_label
        self.dof_ix = dof_ix            # potentially multiple dofs per node
        self.amplitudes = amplitudes      # matching number of amplitudes (as dofs) per node
        self.local = local              # True or False, currently assumed False
