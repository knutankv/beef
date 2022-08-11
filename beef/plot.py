import cmath
import numpy as np
import vispy
from vispy import visuals, scene
from copy import deepcopy
from vispy.color import Colormap
from vispy.color import get_colormaps

def rm_visuals(view):
    for child in view.children[0].children:
        if type(child) in [scene.Line, scene.visuals.Markers, scene.Text]:
            child.parent = None        

def initialize_plot(canvas={}, view=None, cam={}, elements=None, title='BEEF Element plot'):
    if elements is not None:
        nodes = list(set([a for b in [el.nodes for el in elements] for a in b])) #flat list of unique nodes
        node_pos = np.vstack([node.coordinates for node in nodes])
        global_cog = np.mean(node_pos, axis=0)
    else: 
        global_cog = np.array([0,0,0])

    cam_settings = dict(up='z', fov=0, distance=2000, center=global_cog)    #standard values
            

    if type(canvas) is not scene.SceneCanvas:
        sc_settings = dict(bgcolor='white', title=title)
        sc_settings.update(**canvas)
        canvas = scene.SceneCanvas(**sc_settings)

    if view == None:    
        view = canvas.central_widget.add_view()
    else:
        cam = view.camera

    if type(cam) in [scene.cameras.TurntableCamera, scene.cameras.BaseCamera, scene.cameras.FlyCamera, scene.cameras.ArcballCamera, scene.cameras.PanZoomCamera, scene.cameras.Magnify1DCamera]:
        view.camera = cam
    else: # still a dict
        cam_settings.update(**cam)
        view.camera = scene.ArcballCamera(**cam_settings)

    return view, canvas, view.camera

def plot_elements(elements, overlay_deformed=False, sel_nodes=None, sel_elements=None, canvas={}, hold_on=False, view=None, cam={}, 
                  tmat_scaling=1, plot_tmat_ax=None, plot_nodes=False, node_labels=False, element_labels=False, element_label_settings={}, node_label_settings={}, 
                  element_settings={}, node_settings={}, sel_node_settings={}, sel_element_settings={}, sel_node_label_settings={}, sel_element_label_settings={}, 
                  tmat_settings={}, deformed_element_settings={}, title='BEEF Element plot', domain='3d',
                  element_colors=None, colormap_range=None, colormap_name='viridis'):   
    
    # TODO: MAKE 2D compatible
    
    el_settings = dict(color='#008800')
    el_settings.update(**element_settings)

    def_el_settings = dict(color='#ff2288')
    def_el_settings.update(**deformed_element_settings)

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

    # Element colormap
    cm = get_colormaps()[colormap_name]
    if element_colors is not None:
        if colormap_range is not None:
            element_colors = (np.array(element_colors) - colormap_range[0])/(colormap_range[1] - colormap_range[0])

        element_colors = cm[np.array(np.repeat(element_colors,2, axis=0))].rgba
        el_settings['color'] = element_colors

    # Node coordinates
    nodes = list(set([a for b in [el.nodes for el in elements] for a in b])) #flat list of unique nodes
    node_pos = np.vstack([node.coordinates for node in nodes])

    # Selected elements
    if sel_elements is None:
        sel_elements = []

    unsel_elements = [el for el in elements if el.label not in sel_elements]
    sel_elements = [el for el in elements if el.label in sel_elements]
    
    if view is None:
        view, canvas, cam = initialize_plot(canvas=canvas, cam=cam, elements=elements, title=title)
    else:
        canvas = view.canvas
        cam = view.camera

    if not hold_on:
        rm_visuals(view)
   
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

    # Overlay deformed plot if 
    if overlay_deformed:
        element_lines = [None]*len(elements)
        for ix, el in enumerate(elements):
            element_lines[ix] = np.vstack([node.x[:3] for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), connect='segments', **def_el_settings)
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

    return canvas, view


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
