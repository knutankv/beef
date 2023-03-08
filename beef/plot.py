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
import pyvista as pv

def flat(l):
  return [y for x in l for y in x]

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
                  element_values=None, colormap_range=None, colormap_name='viridis', sel_val_format=':.2e', colorbar_limit_format=':.2e', highlight=[]):   

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
        specification of dimensionality of plot
    element_values : float, optional    
        values to plot elements with, used to show contour plots; length should match the 
        number of elements (if more values per entry in list --> assumed gradients on line)
    colormap_range : float, optional
        list of min and max values of colormap (standard value None activates autoscaling to max and min values)
    colormap_name : str, optional 
        name of colormap ('viridis' is standard)
    colorbar_limit_format : str, optional
        format used to create colorbar (':.2e' is standard)
    sel_val_format : str, optional
        format used to create values in texts of selected elements (':.2e' is standard)
    highglight : str, optional
        list of special values to highlight ('max' and 'min' supported currently)


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
    invalid_color = np.array([0.8,0.8,0.8,0.2])
    
    el_settings = dict(color='#008800')
    el_settings.update(**element_settings)

    def_el_settings = dict(color='#ff2288')
    def_el_settings.update(**deformed_element_settings)

    elsel_settings = dict(color='#ff0055',width=7)
    elsel_settings.update(**sel_element_settings)

    elsellab_settings = dict(color='#ff0055', bold=True, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    elsellab_settings.update(**sel_element_label_settings)

    ellab_settings = dict(color='#008800', bold=False, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    ellab_settings.update(**element_label_settings)
 
    n_settings = dict(face_color='#0000ff', edge_color=None, size=6)
    n_settings.update(**node_settings)

    nlab_settings = dict(color='#0000ff', bold=False, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    nlab_settings.update(**node_label_settings)

    nsel_settings = dict(face_color='#ff0000', edge_color='#ff0000', size=6)
    nsel_settings.update(**sel_node_settings)

    nsellab_settings = dict(color='#ff0000', bold=True, italic=False, face='OpenSans', font_size=10, anchor_x='left')
    nsellab_settings.update(**sel_node_label_settings)
    
    tmat_colors = ['#0000ff', '#00ff00', '#ff0000']
    tmatax_settings = dict(arrow_size=1)
    tmatax_settings.update(**tmat_settings)
    
    elements_org = deepcopy(elements)

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

    sel_elements = [el for el in elements if el.label in sel_elements]
   
    # Create lists of selected (and non-selected) els
    unsel_elements_org = [el for el in elements if el not in sel_elements]
    sel_ix_org = np.array([el in sel_elements for el in elements])

    # Reorder selected
    sel_elements_org = [elements[i] for i in np.where(sel_ix_org)[0]]

    
    # Initial treatment of element values / colors
    colorvals_per_element = 1
    
    if element_values is not None:
        if (np.ndim(element_values)==1) or (np.shape(element_values)[1] == 1):
            element_values = [[ev, ev] for ev in element_values]
        
        colorvals_per_element = np.shape(element_values)[1]
        
        max_values = [np.nanmax(elvals) for elvals in element_values]
        min_values = [np.nanmin(elvals) for elvals in element_values]
        
        if 'max' in highlight:
            sel_elements.append(elements[np.nanargmax(max_values)])
        if 'min' in highlight:
            sel_elements.append(elements[np.nanargmin(min_values)])

    # Element colormap
    cm = get_colormaps()[colormap_name]

    if colorvals_per_element > 2:   #further divide to accomodate within along element
        elements = flat([el.subdivide(colorvals_per_element-1) for el in elements])
        
    # Create lists of selected (and non-selected) els
    unsel_elements = [el for el in elements if el not in sel_elements]
    sel_ix = np.array([el in sel_elements for el in elements])

    # Reorder selected
    sel_elements = [elements[i] for i in np.where(sel_ix)[0]]

    # Establish color ranges, color specs, and so on
    if element_values is not None:
        if colormap_range is None:
            colormap_range = [np.nanmin(min_values), np.nanmax(max_values)]
        
        # Establish element colors and rearrange values
        element_values_extended = []
        for elvals in element_values:
            ev_flat = elvals.flatten()
            ev_flat = np.hstack([ev_flat[0], 
                                 np.repeat(ev_flat[1:-1], 2, axis=0), 
                                 ev_flat[-1]])
            
            element_values_extended.append(np.reshape(ev_flat, [-1, 2]))
        
        element_values = np.vstack(element_values_extended)
        element_values_normalized = [(np.array(ev) - colormap_range[0])/(colormap_range[1] - colormap_range[0]) for ev in element_values]

        element_colors = [cm[ev].rgba for ev in element_values_normalized]
        element_colors = np.vstack(element_colors)
        
        for ix, vals in enumerate(element_colors):
            if np.any(np.isnan(vals)):
                element_colors[ix, :] = invalid_color
                
        # Establish updated indexing (selected/not selected)
        sel_ix_colors = np.repeat(sel_ix, 2, axis=0)
 
        # Assign colors
        elsel_settings['color'] = element_colors[sel_ix_colors, :]
        el_settings['color'] = element_colors[~sel_ix_colors, :]
        
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
        
    else:
        # Create lists of selected (and non-selected) els
        unsel_elements = [el for el in elements if el not in sel_elements]
        sel_ix = np.array([el in sel_elements for el in elements])
        
        # Reorder selected
        sel_elements = [elements[i] for i in np.where(sel_ix)[0]]

    # Node coordinates
    nodes = list(set([a for b in [el.nodes for el in elements_org] for a in b])) #flat list of unique nodes
    node_pos = np.vstack([conv_fun(node.coordinates) for node in nodes])
   
    # Establish element lines
    if len(unsel_elements)>0:
        element_lines = [None]*len(unsel_elements)
        for ix, el in enumerate(unsel_elements):
            element_lines[ix] = np.vstack([conv_fun(node.coordinates) for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), 
                                    connect='segments', 
                                    **el_settings)
        view.add(element_visual)
 
    # Establish selected element lines
    if len(sel_elements)>0:
        element_lines = [None]*len(sel_elements)
        for ix, el in enumerate(sel_elements):
            element_lines[ix] = np.vstack([conv_fun(node.coordinates) for node in el.nodes])

        element_visual = scene.Line(pos=np.vstack(element_lines), 
                                    connect='segments', 
                                    **elsel_settings)
        view.add(element_visual)

    # Overlay deformed plot if 
    if overlay_deformed:
        element_lines = [None]*len(elements)
        for ix, el in enumerate(elements):
            element_lines[ix] = np.vstack([conv_fun(node.x)[:3] for node in el.nodes])
        
        element_visual = scene.Line(pos=np.vstack(element_lines), 
                                    connect='segments', 
                                    **def_el_settings)
        view.add(element_visual)

    # Establish element labels
    if element_labels and len(unsel_elements)>0:
        el_cog = [conv_fun(el.get_cog()) for el in unsel_elements_org]
        el_labels = [str(el.label) for el in unsel_elements_org]

        element_label_visual = scene.Text(text=el_labels, pos=el_cog, **ellab_settings)
        view.add(element_label_visual)
   
    if len(sel_elements)>0:

        if element_values is not None:
            sel_values = element_values[sel_ix]
            sel_element_labels = np.unique([el.label for el in sel_elements])
            el_cogs = [np.mean([el.get_cog() for el in sel_elements if el.label==el_label], axis=0) for el_label in sel_element_labels]
            el_labels = [None]*len(sel_element_labels)

            for ix, el_label in enumerate(sel_element_labels):        
                vals = [] 
                for sub_el_ix, element in enumerate([el for el in sel_elements if el.label==el_label]):
                    el_ix = sub_el_ix+sel_elements.index(element)
                    el_vals = sel_values[el_ix]
                    vals.append(el_vals)
                vals = np.array(vals).flatten()

                if np.nanmin(vals)==np.nanmax(vals): 
                    selval = ('{val1' + sel_val_format + '}').format(val1=np.nanmin(vals))
                elif np.all(np.isnan(vals)):
                    selval = 'N/A'
                else:
                    selval = ('{val1' + sel_val_format + '}-' + '{val2' + sel_val_format + '}').format(val1=np.nanmin(vals), val2=np.nanmax(vals))
                    
                el_labels[ix] = f'{el_label} ({selval})'
        else:
            el_cogs = [el.get_cog() for el in sel_elements_org]
            el_labels = [str(el.label) for el in sel_elements_org]

        element_label_visual = scene.Text(text=el_labels,  pos=el_cogs, **elsellab_settings)
        view.add(element_label_visual)       


    # Node labels
    if node_labels:
        node_labels = [str(node.label) for node in nodes]
        element_label_visual = scene.Text(text=node_labels, pos=node_pos, **nlab_settings)
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
            el_vecs = np.vstack([conv_fun(element.tmat[ax, :])[0:3] for element in elements_org])*tmat_scaling
            el_cogs = np.vstack([conv_fun(element.get_cog()) for element in elements_org])
            
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


def plot_elements(elements, plot_states=['undeformed'], plot_nodes=False, vals=None, el_opts={}, def_el_opts={}, node_opts={}, canvas_opts={},
                  show=True, plot_tmat_ax=[0,1,2], tmat_opts={}, tmat_scaling=10, tmat_on=[], val_fun=None,
                  vals_on=['deformed'], colorbar_opts={}, clim=None, annotate_vals={}):

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

    def generate_mesh(els, field='x', pad_size=2):
        # Coordinates of nodes
        nodes = list(set([a for b in [el.nodes for el in els] for a in b])) #flat list of unique nodes
        node_pos = np.vstack([conv_fun(getattr(node, field))[:3] for node in nodes])

        nodes = [n.label for n in nodes]
        edges = np.vstack([[nodes.index(el.nodes[0]), nodes.index(el.nodes[1])] for el in els])

        # # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(edges.shape[0], int) * 2
        padding[:] = pad_size
        edges_w_padding = np.vstack((padding, edges.T)).T
        mesh = pv.PolyData(node_pos, edges_w_padding)
        return mesh
    
    tmat_colors = ['#ff0000', '#00ff00', '#0000ff']

    if vals is None and val_fun is None:
        el_opts['color'] = '#44ff88'

    if val_fun is not None:
        if type(val_fun) is str:
            vstr = val_fun + ''
            if 'title' not in colorbar_opts:
                colorbar_opts['title'] = vstr + ''

            val_fun = lambda el: getattr(el, vstr)

        vals = np.vstack([val_fun(el) for el in elements])

    scalars = dict(undeformed=None, deformed=None)
    scalars.update({key: vals for key in vals_on})

    show_scalarbar = dict(undeformed=False, deformed=False)
    show_scalarbar.update({key: True for key in vals_on})
    el_opts['clim'] = clim
    def_el_opts['clim'] = clim


    canvas_settings = dict(background_color='white')
    tmat_settings = dict(show_edges=False)
    
    scalar_bar_settings = dict(
        title_font_size=20,
        label_font_size=16,
        n_labels=4,
        italic=False,
        color='black',
        fmt="%.2e",
        font_family="arial",
    )
    
    def_el_settings = dict(
        scalars=scalars['deformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        line_width=4,
        cmap='viridis',
        show_scalar_bar=show_scalarbar['deformed'],
        color='#ee8899'
        )
    
    el_settings = dict(
        scalars=scalars['undeformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        line_width=4,
        cmap='viridis',
        show_scalar_bar=show_scalarbar['undeformed'],
        )
    

    node_settings = dict(
        render_points_as_spheres=True,
        color='magenta',
        point_size=5
    )  

    scalar_bar_settings.update(colorbar_opts)
    def_el_settings.update(def_el_opts)
    canvas_settings.update(canvas_opts)
    el_settings.update(el_opts)
    node_settings.update(node_opts)
    tmat_settings.update(tmat_opts)

    if vals is None:
        vals = [0]*len(elements)

    pl = pv.Plotter()
    mesh = generate_mesh(elements,'x0')

    if 'undeformed' in plot_states:
        pl.add_mesh(mesh,annotations=annotate_vals, scalar_bar_args=scalar_bar_settings, **el_settings )

    if 'deformed' in plot_states:
        pl.add_mesh(generate_mesh(elements, 'x'),annotations=annotate_vals, scalar_bar_args=scalar_bar_settings, **def_el_settings)
    
    if plot_nodes:
        pl.add_points(mesh.extract_surface().points, **node_settings)

    for key in canvas_settings:
        setattr(pl, key, canvas_settings[key])

    if plot_tmat_ax is not None:
        for state in tmat_on:
            if state == 'deformed':
                T_field = 'Tn'
                deformed = True
            else:
                T_field = 'T0'
                deformed = False
            
            for ax in plot_tmat_ax:
                vec = []
                pos = []

                for el in elements:
                    vec.append(conv_fun(getattr(el,T_field)[ax, :][:3])[:3]*tmat_scaling)
                    pos.append(conv_fun(el.get_cog(deformed=deformed)))

                pl.add_arrows(np.vstack(pos), np.vstack(vec), color=tmat_colors[ax], **tmat_settings)
    
    # if vals is not None:
        # pl.add_scalar_bar(**scalar_bar_settings)
    
    if elements[0].domain == '2d':
        pl.view_xy()
    else:
        pl.view_isometric()
    
    pl.show_axes()
    if show:
        pl.show()

    return pl
