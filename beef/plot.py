'''
Plotting functions.
'''
import numpy as np
from copy import deepcopy
import pyvista as pv

def flat(l):
  return [y for x in l for y in x]
    

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
                  show=True, plot_tmat_ax=[1,2], tmat_opts={}, tmat_scaling=10, tmat_on=[], val_fun=None,
                  vals_on=['deformed'], colorbar_opts={}, clim=None, annotate_vals={}, pl=None, node_labels=False, 
                  element_labels=False, thickness_scaling=None):
        
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


    def group_elements(els):
        sections = list(set([el.section for el in els]))
        grouped_els = {sec: [el for el in els if el.section==sec] for sec in sections}
        
        return grouped_els
        
        
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
        lighting=True,
        cmap='viridis',
        show_scalar_bar=show_scalarbar['deformed'],
        color='#ee8899'
        )
    
    el_settings = dict(
        scalars=scalars['undeformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        lighting=True,
        line_width=4,
        cmap='viridis',
        show_scalar_bar=show_scalarbar['undeformed'],
        )

        
    node_settings = dict(
        render_points_as_spheres=True,
        color='magenta',
        lighting=True,
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
        
    if pl is None:
        pl = pv.Plotter()
        
    mesh = generate_mesh(elements,'x0')

    if 'undeformed' in plot_states:
        if thickness_scaling is not None:
            grouped_els = group_elements(elements)
            lw0 = el_settings['line_width']*1.0
            for sec in grouped_els:
                el_settings['line_width'] = lw0*thickness_scaling(sec)
                pl.add_mesh(generate_mesh(grouped_els[sec],'x0'), **el_settings)     
        else:
            pl.add_mesh(mesh, annotations=annotate_vals, scalar_bar_args=scalar_bar_settings, **el_settings)

    if 'deformed' in plot_states:
        if thickness_scaling is not None:
            grouped_els = group_elements(elements)
            lw0 = def_el_settings['line_width']*1.0
            for sec in grouped_els:
                def_el_settings['line_width'] = lw0*thickness_scaling(sec)
                pl.add_mesh(generate_mesh(grouped_els[sec],'x'), **def_el_settings)     
        else:
            pl.add_mesh(generate_mesh(elements, 'x'), annotations=annotate_vals, scalar_bar_args=scalar_bar_settings, **def_el_settings)
    
    if plot_nodes:
        if 'undeformed' in plot_states:
            pl.add_points(mesh.extract_surface().points, **node_settings)
        if 'deformed' in plot_states:
            pl.add_points(generate_mesh(elements, 'x').extract_surface().points, **node_settings)

    if node_labels:
        nodes = list(set([a for b in [el.nodes for el in elements] for a in  b]))
        lbl = [str(node.label) for node in nodes]
        if 'deformed' in plot_states:
            pl.add_point_labels(np.vstack([node.x[:3] for node in nodes]), lbl)
        else:
            pl.add_point_labels(np.vstack([node.x0[:3] for node in nodes]), lbl)            
            
    if element_labels:
        lbl = [str(el.label) for el in elements]
        if 'deformed' in plot_states:
            pl.add_point_labels(np.vstack([el.get_cog(deformed=True) for el in elements]), lbl, text_color='blue', shape_color='white', shape_opacity=0.4)
        else:
            pl.add_point_labels(np.vstack([el.get_cog(deformed=False) for el in elements]), lbl, text_color='blue', shape_color='white', shape_opacity=0.4)            
            
        
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
