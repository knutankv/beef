from . import *

#%% Load class definition 
class Load:
    def __init__(self, node_labels, dofs, amplitudes, name='load-0', local=False, plotcolor='DarkOrange'):
        self.nodeloads = []
        self.plotcolor = plotcolor
               
        if (type(dofs) is not list) or (np.array(dofs).ndim!=2):
            dofs = [dofs]*len(node_labels)
        
        if (type(amplitudes) is not list) or (np.array(amplitudes).ndim!=2):
            amplitudes = [amplitudes]*len(node_labels) 
        
        for ix, node_label in enumerate(node_labels):
            self.nodeloads.append(NodeLoad(node_label, ensure_list(dofs[ix]), ensure_list(amplitudes[ix]), local)) 

    # CORE METHODS
    def __str__(self):
        return f'BEEF Load: {self.name}'

    def __repr__(self):
        return f'BEEF Load: {self.name}'

class NodeLoad:
    def __init__(self, node_label, dof_ix, amplitudes, local):
        
        if type(dof_ix) is not list:
            dof_ix = [dof_ix]
        if type(amplitudes) is not list:
            amplitudes = [amplitudes]
            
        self.node_label = node_label
        self.dof_ix = dof_ix            # potentially multiple dofs per node
        self.amplitudes = amplitudes      # matching number of amplitudes (as dofs) per node
        self.local = local              # True or False, currently assumed False
