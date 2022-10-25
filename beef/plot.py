'''
Plotting functions.
'''

import cmath
import numpy as np
import vispy
from vispy import visuals, scene
from copy import deepcopy
from vispy.color import Colormap
from vispy.color import get_colormaps
from vispy.util.quaternion import Quaternion

def rm_visuals(view):
    '''
    Remove visuals from vispy-generated view.

    Arguments
    ------------
    view : vispy View object
        the view to remove visuals from

    '''
    for child in view.children[0].children:
        if type(child) in [scene.Line, scene.visuals.Markers, scene.Text]:
            child.parent = None        

def initialize_plot(canvas={}, view=None, cam={}, elements=None, title='BEEF Element plot'):
    '''
    Initialize vispy plot with specified settings. Used prior to `plot_elements`.

    Arguments
    ------------
    canvas : SceneCanvas (vispy) object
        canvas definition for plot
    view : View (vispy) object, optional
        view definition for plot
    cam : Camera (vispy) object, optional
    elements : Element object, optional
        list of Element objects used to establish reasonable zoom level
    title : str, optional
        name of plot; 'BEEF Element plot' is standard

    Returns
    ------------
    view : View (vispy) object
        resulting view
    canvas : SceneCanvas (vispy) object
        resulting canvas
    camera : Camera (vispy) object
        resulting camera spec

    '''
    if elements is not None:
        nodes = list(set([a for b in [el.nodes for el in elements] for a in b])) #flat list of unique nodes
        node_pos = np.vstack([node.coordinates for node in nodes])
        global_cog = np.mean(node_pos, axis=0)
        d_nodes = np.sum(node_pos**2, axis=1)

        d_max = np.max((d_nodes[:,np.newaxis]-d_nodes[np.newaxis,:]).flatten())
        distance = 1e-3*d_max
    else: 
        global_cog = np.array([0,0,0])
        distance = 1000
    
   
    cam_settings = dict(up='z', fov=0, distance=distance, center=global_cog)    #standard values
            

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
        q = Quaternion(w=-0.5, x=0.654, y=0.302, z=0.245)
        view.camera.set_state(dict(_quaternion=q))

    return view, canvas, view.camera

def plot_elements(elements, overlay_deformed=False, sel_nodes=None, sel_elements=None, canvas={}, hold_on=False, view=None, cam={}, 
                  tmat_scaling=1, plot_tmat_ax=None, plot_nodes=False, node_labels=False, element_labels=False, element_label_settings={}, node_label_settings={}, 
                  element_settings={}, node_settings={}, sel_node_settings={}, sel_element_settings={}, sel_node_label_settings={}, sel_element_label_settings={}, 
                  tmat_settings={}, deformed_element_settings={}, title='BEEF Element plot', domain='3d',
                  element_colors=None, colormap_range=None, colormap_name='viridis', colorbar_limit_format=':.2e'):   

    '''
    Plot beam/bar elements.

    Arguments
    ------------
    elements : Element object
        list of Element objects to plot
    overlay_deformed : False, optional
        whether or not to plot deformed state of elements overlayed
    sel_nodes : int, optional
        list of integer node labels to highlight
    sel_elements : int, optional
        list of integer element labels to highlight
    canvas : vispy.SceneCanvas object
        canvas definition for plot
    hold_on : False
        whether or not to keep visuals from canvas input
    view : vispy.View object, optional
        view definition for plot
    cam : vispy.Camera object, optional
    tmat_scaling : 1.0, optional
        scaling used for alternative plotting of transformation matrices
    plot_tmat_ax : int, optional
        list of indices describing the axes to be plot (0,1,2 possible); standard value None results in no plot of unit vectors
    plot_nodes : False, optional
        whether or not to plot markers at nodes
    node_labels : False, optional
        whether or not to show node labels
    element_labels : False, optional
        whether or not to show element labels
    element_label_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard element label properties; 
        refer to vispy manual for allowed properties for Text objects
    node_label_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard node label properties; 
        refer to vispy manual for allowed properties for Text objects
    element_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard element properties; 
        refer to vispy manual for allowed properties for Line objects
    node_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard node properties; 
        refer to vispy manual for allowed properties for Markers objects
    sel_node_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard selected node properties; 
        refer to vispy manual for allowed properties for Markers objects
    sel_element_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard selected element properties; 
        refer to vispy manual for allowed properties for Line objects
    sel_node_label_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard selected node label properties; 
        refer to vispy manual for allowed properties for Text objects
    sel_element_label_settings : dict, optional 
        dictionary describing parameters to be added/overwritten to the standard selected element label properties; 
        refer to vispy manual for allowed properties for Text objects
    tmat_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard unit vector (describing the transformation matrix) properties; 
        refer to vispy manual for allowed properties for Arrow objects
    deformed_element_settings : dict, optional
        dictionary describing parameters to be added/overwritten to the standard deformed element properties; 
        refer to vispy manual for allowed properties for Line objects      
    title : str, optional
        name of plot; 'BEEF Element plot' is standard
    domain : {'3d', '2d'}
        specification of dimensionality of plot (**only 3d plots are currently supported**)
    element_colors : float, optional    
        colors / values to plot elements with, used to show contour plots; length should match the 
        number of elements
    colormap_range : float, optional
        list of min and max values of colormap (standard value None activates autoscaling to max and min values)
    colormap_name : str, optional 
        name of colormap ('viridis' is standard)
    colorbar_limit_format : str, optional
        format used to create colorbar (':.2e' is standard)


    Returns
    ------------
    view : vispy.View object
        resulting view
    canvas : vispy.SceneCanvas object
        resulting canvas
    camera : vispy.Camera object
        resulting camera spec

    '''
    
    # Establish function to convert 2d coordinates to 3d for plotting 2d-domained elements
    if elements[0].domain == '2d':
        def conv_fun(xyz):
            if len(xyz)==2:
                return np.hstack([xyz, 0])
            elif len(xyz)==3:
                return np.hstack([xyz[:2], 0, 0, xyz[2], 0])
            else:
                raise ValueError('Wrong size of xyz')
    else:
        conv_fun = lambda xyz: xyz
    
    # Settings
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

    # Establish view and canvas    
    if view is None:
        view, canvas, cam = initialize_plot(canvas=canvas, cam=cam, elements=elements, title=title)
    else:
        canvas = view.canvas
        cam = view.camera

    if not hold_on:
        rm_visuals(view)

    # Selected elements
    if sel_elements is None:
        sel_elements = []

    sel_ix = np.array([el.label in sel_elements for el in elements])
    unsel_elements = [el for el in elements if el.label not in sel_elements]
    sel_elements = [el for el in elements if el.label in sel_elements]

    # Element colormap
    cm = get_colormaps()[colormap_name]

    if element_colors is not None:
        element_values = element_colors*1
        element_colors = np.array(element_colors)
        nan_ix = np.isnan(element_colors)
        
        if colormap_range is not None:
            element_colors = (np.array(element_colors) - colormap_range[0])/(colormap_range[1] - colormap_range[0])

        element_colors = cm[np.array(np.repeat(element_colors, 2, axis=0))].rgba
        
        nan_ix = np.isnan(element_colors[:,0])
        if any(nan_ix):
            element_colors[nan_ix, :] = 0
            element_colors[nan_ix, -1] = 0.1

        el_settings['color'] = element_colors[np.repeat(~sel_ix, 2, axis=0),:]
        elsel_settings['color'] = element_colors[np.repeat(sel_ix, 2, axis=0),:]

        grid = canvas.central_widget.add_grid()
        cam_cb = scene.TurntableCamera(distance=1.3, fov=0, azimuth=180, roll=0, elevation=90, center= (3.8, 5, 0), interactive=True)
        
        view = grid.add_view(row=0, col=0, camera=cam, col_span=4)
        view_cb = grid.add_view(row=0, col=4, camera=cam_cb, col_span=1) 
        
        cb = scene.ColorBarWidget(clim=colormap_range, cmap=cm, orientation="right",
                                            border_width=1, padding=[0,0], axis_ratio=0.08)
        
        text_upper = scene.visuals.Text(text=('{lim1' + colorbar_limit_format + '}').format(lim1=colormap_range[0]), pos=(3.4,10,0), anchor_y='top')
        text_lower = scene.visuals.Text(text=('{lim1' + colorbar_limit_format + '}').format(lim1=colormap_range[1]), pos=(3.4,0,0), anchor_y='bottom')
            
        view_cb.add(cb)     
        view_cb.add(text_upper)    
        view_cb.add(text_lower)    

    # Node coordinates
    nodes = list(set([a for b in [el.nodes for el in elements] for a in b])) #flat list of unique nodes
    node_pos = np.vstack([conv_fun(node.coordinates) for node in nodes])
   
    # Establish element lines
    if len(unsel_elements)>0:
        element_lines = [None]*len(unsel_elements)
        for ix, el in enumerate(unsel_elements):
            element_lines[ix] = np.vstack([conv_fun(node.coordinates) for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), connect='segments', **el_settings)
        view.add(element_visual)
 
    # Establish selected element lines
    if len(sel_elements)>0:
        element_lines = [None]*len(sel_elements)
        for ix, el in enumerate(sel_elements):
            element_lines[ix] = np.vstack([conv_fun(node.coordinates) for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), connect='segments', **elsel_settings)
        view.add(element_visual)

    # Overlay deformed plot if 
    if overlay_deformed:
        element_lines = [None]*len(elements)
        for ix, el in enumerate(elements):
            element_lines[ix] = np.vstack([conv_fun(node.x)[:3] for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), connect='segments', **def_el_settings)
        view.add(element_visual)

    # Establish element labels
    if element_labels and len(unsel_elements)>0:
        el_cog = [conv_fun(el.get_cog()) for el in unsel_elements]
        el_labels = [str(el.label) for el in unsel_elements]

        element_label_visual = scene.Text(text=el_labels,  pos=el_cog, **ellab_settings)
        view.add(element_label_visual)
   
    if len(sel_elements)>0:
  
        el_cog = [el.get_cog() for el in sel_elements]
        el_labels = [f'{el.label} ({element_values[sel_ix][ix]:.3f})' for ix,el in enumerate(sel_elements)]
        
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
            node_pos = np.vstack([conv_fun(node.coordinates) for node in sel_nodes])
            sel_node_label_visual = scene.Text(text=sel_node_labels,  pos=node_pos, **nsellab_settings)
            sel_node_visual = scene.visuals.Markers(pos=node_pos, **nsel_settings)

            view.add(sel_node_label_visual)
            view.add(sel_node_visual)
        else:
            print('Requested nodes to highlight not found.')
    
    # Add transformation matrices
    if plot_tmat_ax is not None:
        for ax in plot_tmat_ax:
            el_vecs = np.vstack([conv_fun(element.tmat[ax, :])[0:3] for element in elements])*tmat_scaling
            el_cogs = np.vstack([conv_fun(element.get_cog()) for element in elements])
            
            # INTERTWINE TMAT AND ELCOGS
            arrow_data = np.hstack([el_cogs, el_cogs+el_vecs])
 
            tmat_visual = scene.Arrow(pos=arrow_data.reshape([el_vecs.shape[0]*2, 3]), color=tmat_colors[ax], arrow_color=tmat_colors[ax], connect='segments', arrows=arrow_data,**tmatax_settings)
            view.add(tmat_visual)   

    canvas.show()
    axis = scene.visuals.XYZAxis(parent=view.scene)


    return canvas, view

def frame_creator(frames=30, repeats=1, swing=False, full_cycle=False):
    '''
    Construct scaling of value.

    Arguments
    ------------
    frames : int, optional
        number of frames (30 is standard)
    repeats : int, optional
        number of repeats (1 is standard)
    swing : False, optional
        whether or not to swing back and forth
    full_cycle : False, optional
        whether or not to include the full cycle (resulting in initial scaling being -1 rather than 0)

    Returns
    ----------
    scaling : float
        numpy array describing scaling of value to animate
    '''
    
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
