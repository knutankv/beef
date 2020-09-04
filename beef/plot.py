import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection as LC


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
    
    element_lines = [None]*len(elements)

    for ix, el in enumerate(elements):
        element_lines[ix] = np.vstack([node.coordinates for node in el.nodes])
        
        if element_labels:
            cog = el.cog
            ax.text(cog[0], cog[1], cog[2], el.label, **l_e_dict)
    
    ax.add_collection(LC(element_lines, **e_dict))

    nodes = [el.nodes for el in elements]
    nodes = [a for b in nodes for a in b] #flatten
    nodes = list(set(nodes))    #only keep unique
    
    if plot_nodes:
        all_node_coords = np.vstack([node.coordinates for node in nodes])
        ax.plot(all_node_coords[:,0], all_node_coords[:,1], all_node_coords[:,2], **n_dict)
    
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

    

def frame_creator(frames=30, repeats=1, swing=False, full_cycle=False):
    
    if full_cycle:
        d = 2/frames
        start = -1
    else:
        d = 1/frames
        start = 0
    
    if swing:
        base_scaling = np.hstack([np.linspace(start,1-d,frames), np.linspace(1,start+d,frames)])
    else:
        base_scaling = np.linspace(start,1-d,frames) 

    return np.tile(base_scaling, repeats)