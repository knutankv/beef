import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def plot_elements(elements, color='Gray', plot_nodes=False, highlighted_nodes=None, node_labels=False, 
         element_labels=False, fig=None, ax=None, element_settings={},
         node_settings={}, node_label_settings={}, element_label_settings={}):
        
    e_dict = {'color': 'DarkGreen', 'alpha': 1}
    e_dict.update(**element_settings)

    n_dict = {'color':'Black', 'linestyle':'', 'marker':'.', 'markersize':4, 'alpha':0.8}
    n_dict.update(**node_settings)

    l_n_dict = {'color':'Black', 'fontsize': 8, 'fontweight':'normal'}
    l_n_dict.update(**node_label_settings)
    
    l_e_dict = {'color':'LimeGreen', 'fontsize': 8, 'fontweight':'bold', 'style':'italic'}
    l_e_dict.update(**element_label_settings)

    if ax is None and fig is None:
        fig = plt.gcf()
            
    if ax == None:
        ax = fig.gca(projection='3d')
        ax = fig.gca()
    
    h = [None]*len(elements)

    for ix, el in enumerate(elements):
        xy = np.vstack([node.coordinates for node in el.nodes])
        h[ix] = plt.plot(xy[:,0], xy[:,1], xy[:,2], **e_dict)[0]
        
        if plot_nodes:
            ax.plot(xy[:,0], xy[:,1], xy[:,2], **n_dict)
            
        if element_labels:
            cog = el.cog
            ax.text(cog[0], cog[1], cog[2], el.label, **l_e_dict)
        
    nodes = [el.nodes for el in elements]
    nodes = [a for b in nodes for a in b] #flatten
    nodes = list(set(nodes))    #only keep unique
    
    if highlighted_nodes is not None:
        node_labels = [node.label for node in nodes]
        sel_nodes = [node for node in nodes if node.label in highlighted_nodes]
        
        if len(sel_nodes) >= 1:
            xy = np.vstack([node.coordinates for node in sel_nodes])
            ax.plot(xy[:,0], xy[:,1], xy[:,2], '.r')
        else:
            print('Requested nodes to highlight not found.')

    
    if node_labels:
        for node in nodes:
            ax.text(node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, **l_n_dict)