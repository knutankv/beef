import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection as LC
import vispy
vispy.use('PyQt5')
from vispy import visuals, scene
    

def plot_elements(elements, sel_nodes=None, sel_elements=None, canvas={}, cam={}, tmat_scaling=1, plot_tmat_ax=None, plot_nodes=False, node_labels=False, element_labels=False, element_label_settings={}, node_label_settings={}, element_settings={}, node_settings={}, sel_node_settings={}, sel_element_settings={}, sel_node_label_settings={}, sel_element_label_settings={},tmat_settings={}):       
    el_settings = dict(color='#008800')
    el_settings.update(**element_settings)

    elsel_settings = dict(color='#ff0055',width=3)
    elsel_settings.update(**sel_element_settings)

    elsellab_settings = dict(color='#ff0055', bold=True, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    elsellab_settings.update(**sel_element_label_settings)

    ellab_settings = dict(color='#008800', bold=False, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    ellab_settings.update(**element_label_settings)
 
    n_settings = dict(face_color='#0000ff', edge_color=None, size=4)
    n_settings.update(**node_settings)

    nlab_settings = dict(color='#0000ff', bold=False, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    nlab_settings.update(**node_label_settings)

    nsel_settings = dict(face_color='#ff0000', edge_color='#ff0000', size=8)
    nsel_settings.update(**sel_node_settings)

    nsellab_settings = dict(color='#ff0000', bold=True, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    nsellab_settings.update(**sel_node_label_settings)
    
    tmat_colors = ['#0000ff', '#00ff00', '#ff0000']
    tmatax_settings = dict(arrow_size=1)
    tmatax_settings.update(**tmat_settings)

    # Node coordinates
    nodes = list(set([a for b in [el.nodes for el in elements] for a in b])) #flat list of unique nodes
    node_pos = np.vstack([node.coordinates for node in nodes])

    # Selected elements
    if sel_elements is None:
        sel_elements = []

    unsel_elements = [el for el in elements if el.label not in sel_elements]
    sel_elements = [el for el in elements if el.label in sel_elements]

    if type(canvas) is not scene.SceneCanvas:
        sc_settings = dict(bgcolor='white', title='BEEF Element plot')
        sc_settings.update(**canvas)
        canvas = scene.SceneCanvas(**sc_settings)
        
    view = canvas.central_widget.add_view()

    # Camera settings
    if type(cam) in [scene.cameras.TurntableCamera, scene.cameras.BaseCamera, scene.cameras.FlyCamera, scene.cameras.ArcballCamera, scene.cameras.PanZoomCamera, scene.cameras.Magnify1DCamera]:
        view.camera = cam
    else:   
        global_cog = np.mean(node_pos, axis=0)
        cam_settings = dict(up='z', fov=0, distance=1000, center=global_cog)
        cam_settings.update(**cam)
        view.camera = scene.ArcballCamera(**cam_settings)
    
    # Establish element lines
    if len(unsel_elements)>0:
        element_lines = [None]*len(unsel_elements)
        for ix, el in enumerate(unsel_elements):
            element_lines[ix] = np.vstack([node.coordinates for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), connect='segments', **el_settings)
        view.add(element_visual)

    # Establish selected element lines
    if len(sel_elements)>0:
        element_lines = [None]*len(sel_elements)
        for ix, el in enumerate(sel_elements):
            element_lines[ix] = np.vstack([node.coordinates for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), connect='segments', **elsel_settings)
        view.add(element_visual)

    # Establish element labels
    if element_labels and len(unsel_elements)>0:
        el_cog = [el.get_cog() for el in unsel_elements]
        el_labels = [str(el.label) for el in unsel_elements]

        element_label_visual = scene.Text(text=el_labels,  pos=el_cog, **ellab_settings)
        view.add(element_label_visual)
   
    if len(sel_elements)>0:
        el_cog = [el.get_cog() for el in sel_elements]
        el_labels = [str(el.label) for el in sel_elements]

        element_label_visual = scene.Text(text=el_labels,  pos=el_cog, **elsellab_settings)
        view.add(element_label_visual)       


    # Node labels
    if node_labels:
        node_labels = [str(node.label) for node in nodes]
        element_label_visual = scene.Text(text=node_labels,  pos=node_pos, **nlab_settings)
        view.add(element_label_visual)

    # Nodes
    if plot_nodes:  
        node_visual = scene.visuals.Markers(pos=node_pos, **n_settings)
        view.add(node_visual)

    if sel_nodes is not None:
        sel_nodes = [node for node in nodes if node.label in sel_nodes]
        sel_node_labels = [str(node.label) for node in sel_nodes]        

        if len(sel_nodes) >= 1:
            node_pos = np.vstack([node.coordinates for node in sel_nodes])
            sel_node_label_visual = scene.Text(text=sel_node_labels,  pos=node_pos, **nsellab_settings)
            sel_node_visual = scene.visuals.Markers(pos=node_pos, **nsel_settings)

            view.add(sel_node_label_visual)
            view.add(sel_node_visual)
        else:
            print('Requested nodes to highlight not found.')
    
    # Add transformation matrices
    if plot_tmat_ax is not None:
        for ax in plot_tmat_ax:
            el_vecs = np.vstack([element.tmat[ax, 0:3] for element in elements])*tmat_scaling
            el_cogs = np.vstack([element.get_cog() for element in elements])
            
            # INTERTWINE TMAT AND ELCOGS
            arrow_data = np.hstack([el_cogs, el_cogs+el_vecs])
 
            tmat_visual = scene.Arrow(pos=arrow_data.reshape([el_vecs.shape[0]*2, 3]), color=tmat_colors[ax], arrow_color=tmat_colors[ax], connect='segments', arrows=arrow_data,**tmatax_settings)
            view.add(tmat_visual)   

    canvas.show()
    axis = scene.visuals.XYZAxis(parent=view.scene)

    return axis
    
    

def plot_elements_legacy(elements, color='Gray', plot_nodes=False, highlighted_nodes=None, node_labels=False, 
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
            
    if ax is None:
        ax = fig.gca(projection='3d')
    
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

    return ax

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