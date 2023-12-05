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
                  vals_on=[], colorbar_opts={}, clim=None, annotate_vals={}, pl=None, node_labels=False, 
                  element_labels=False, thickness_scaling=None, cmap='viridis', view=None, nodelabel_opts={}, elementlabel_opts={},
                  element_label_fun=None, node_label_fun=None):
        
    if element_label_fun is None:
        element_label_fun = lambda el: str(int(el.label))
    if node_label_fun is None:
        node_label_fun = lambda n: str(int(n.label))
    
    
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
        group_ixs = {sec: np.array([el in grouped_els[sec] for el in els]) for sec in grouped_els}
        
        return grouped_els, group_ixs
    

        
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
    
    scalar_bar_settings = dict(
        title_font_size=20,
        label_font_size=16,
        n_labels=4,
        italic=False,
        color='black',
        fmt="%.2e",
        font_family="arial"
    )
    
    scalar_bar_settings.update(colorbar_opts)
    
    if vals is None and val_fun is None and 'color' not in el_opts:
        el_opts['color'] = '#44ff88'

    if val_fun is not None:
        if type(val_fun) is str:
            vstr = val_fun + ''
            if 'title' not in colorbar_opts:
                colorbar_opts['title'] = vstr + ''

            val_fun = lambda el: getattr(el, vstr)

        vals = np.vstack([val_fun(el) for el in elements])

    scalars = dict(undeformed=None, deformed=None)
    
    show_scalarbar = dict(undeformed=False, deformed=False)
    show_scalarbar.update({key: True for key in vals_on})
    scalars.update({key: vals for key in vals_on})
    
    if vals is not None:
        if clim is None:
            clim = [np.min(vals[~np.isnan(vals)]), np.max(vals[~np.isnan(vals)])]
        
        if np.min(vals[~np.isnan(vals)])<0:     #adjust for negative values
            # print(vals[~np.isnan(vals)])
            shift = -np.min(vals[~np.isnan(vals)])
        else:
            shift = 0
            
        vals = vals + shift
        new_annotations = {(c+shift): c for c in clim}
        
        # TODO: SUUPER PATCHY - fix later
        if "fmt" in scalar_bar_settings:
            fmt = scalar_bar_settings["fmt"].split('%.')[1][:-1]
        else:
            fmt = '2'
                
        annotate_vals = {(c+shift): f'{annotate_vals[c]:.{fmt}f}' for c in annotate_vals}
        annotate_vals.update(new_annotations) 
        clim = np.array(clim) + shift    
        cmap = pv.LookupTable(cmap=cmap, scalar_range=clim, annotations=annotate_vals,
                              nan_color='#dddddd') 
  

    else:
        cmap = None
        scalar_bar_settings = {}
        show_scalarbar = dict(undeformed=False, deformed=False)

    el_opts['clim'] = clim
    def_el_opts['clim'] = clim

    canvas_settings = dict(background_color='white')
    tmat_settings = dict(show_edges=False)    

    nodelabel_settings = dict(always_visible=True)
    elementlabel_settings = dict(always_visible=True)
    
    def_el_settings = dict(
        scalars=scalars['deformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        line_width=4,
        lighting=True,
        cmap=cmap,
        show_scalar_bar=show_scalarbar['deformed'],
        color='#ee8899'
        )
    
    el_settings = dict(
        scalars=scalars['undeformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        lighting=True,
        line_width=4,
        cmap=cmap,
        show_scalar_bar=show_scalarbar['undeformed'],
        )

        
    node_settings = dict(
        render_points_as_spheres=True,
        color='magenta',
        lighting=True,
        point_size=5
    )  

    
    def_el_settings.update(def_el_opts)
    canvas_settings.update(canvas_opts)
    el_settings.update(el_opts)
    node_settings.update(node_opts)
    tmat_settings.update(tmat_opts)
    nodelabel_settings.update(nodelabel_opts)
    elementlabel_settings.update(elementlabel_opts)
     
    if pl is None:
        pl = pv.Plotter()
        
    mesh = generate_mesh(elements,'x0')

    if 'undeformed' in plot_states:
        if thickness_scaling is not None:
            grouped_els, group_ixs = group_elements(elements)
            lw0 = el_settings['line_width']*1.0
            for sec in grouped_els:
                el_settings['line_width'] = lw0*thickness_scaling(sec)

                if 'undeformed' in vals_on:
                    el_settings['scalars'] = scalars['undeformed'][group_ixs[sec]]

                pl.add_mesh(generate_mesh(grouped_els[sec],'x0'), 
                             scalar_bar_args=scalar_bar_settings, annotations=annotate_vals, 
                            **el_settings)     
        else:
            pl.add_mesh(mesh, scalar_bar_args=scalar_bar_settings, 
                        annotations=annotate_vals, **el_settings)

    if 'deformed' in plot_states:
        if thickness_scaling is not None:
            grouped_els, group_ixs = group_elements(elements)
            lw0 = def_el_settings['line_width']*1.0
            for sec in grouped_els:
                def_el_settings['line_width'] = lw0*thickness_scaling(sec)
                
                if 'deformed' in vals_on:
                    el_settings['scalars'] = scalars['deformed'][group_ixs[sec]]

                pl.add_mesh(generate_mesh(grouped_els[sec], 'x'), annotations=annotate_vals,
                            scalar_bar_args=scalar_bar_settings, **def_el_settings)     
        else:
            pl.add_mesh(generate_mesh(elements, 'x'), annotations=annotate_vals,
                        scalar_bar_args=scalar_bar_settings, **def_el_settings)
    
    if plot_nodes:
        if 'undeformed' in plot_states:
            pl.add_points(mesh.extract_surface().points, **node_settings)
        if 'deformed' in plot_states:
            pl.add_points(generate_mesh(elements, 'x').extract_surface().points, **node_settings)

    if node_labels is not False:
        if node_labels is not True:
            all_nodes = list(set([a for b in [el.nodes for el in elements] for a in b]))
            nodes = [node for node in all_nodes if node in node_labels]
        else:    #assume all should be labeled
            nodes = list(set([a for b in [el.nodes for el in elements] for a in b]))
            
        lbl = [node_label_fun(node) for node in nodes]
        
        if 'deformed' in plot_states:
            pl.add_point_labels(np.vstack([node.x[:3] for node in nodes]), lbl, **nodelabel_settings)
        else:
            pl.add_point_labels(np.vstack([node.x0[:3] for node in nodes]), lbl, **nodelabel_settings)            
            
    if element_labels  is not False:
        if element_labels is not True:
            elements_labeled = [element for element in elements if element.label in element_labels]
        else:
            elements_labeled = elements
        
        lbl = [element_label_fun(el) for el in elements_labeled]

        if 'deformed' in plot_states:
            pl.add_point_labels(np.vstack([conv_fun(el.get_cog(deformed=True)) for el in elements_labeled]), 
                                lbl, text_color='blue', shape_color='white', shape_opacity=0.4, **nodelabel_settings)
        else:
            pl.add_point_labels(np.vstack([conv_fun(el.get_cog(deformed=False)) for el in elements_labeled]), 
                                lbl, text_color='blue', shape_color='white', shape_opacity=0.4, **nodelabel_settings)            
            
        
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
    
    if vals is not None:
        pl.add_scalar_bar(**scalar_bar_settings)
    
    if elements[0].domain == '2d':
        pl.view_xy()
    else:
        pl.view_isometric()
    
    if view is not None:
        if view in ['xy', 'top']:
            pl.view_xy()
        if view in ['yz', 'front']:
            pl.view_yz()
        if view in ['xz', 'side']:
            pl.view_xz()
        
    
    pl.show_axes()
    if show:
        pl.show()

    return pl


def plot_eldef(eldef, plot_elements=True, plot_states=['undeformed'], plot_nodes=False, vals=None, el_opts={}, def_el_opts={}, node_opts={}, canvas_opts={},
                  show=True, plot_tmat_ax=[1,2], tmat_opts={}, tmat_scaling=10, tmat_on=[], val_fun=None,
                  vals_on=[], colorbar_opts={}, clim=None, annotate_vals={}, pl=None, node_labels=False, 
                  element_labels=False, thickness_scaling=None, cmap='viridis', view=None, nodelabel_opts={}, elementlabel_opts={},
                  element_label_fun=None, constraints_on=['undeformed','deformed'], def_constraint_opts={}, constraint_opts={}, plot_constraints=[], node_label_fun=None, plot_node_states=None):
    
    if plot_node_states is None:
        plot_node_states = plot_states
        
    if element_label_fun is None:
        element_label_fun = lambda el: str(int(el.label))
    if node_label_fun is None:
        node_label_fun = lambda n: str(int(n.label))
        
    if plot_elements is not False:
        if plot_elements is True:
            elements = eldef.elements
        else:
            elements = [el for el in eldef.elements if el in plot_elements]

    nodes = eldef.nodes
    
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
        group_ixs = {sec: np.array([el in grouped_els[sec] for el in els]) for sec in grouped_els}
        
        return grouped_els, group_ixs
        
    def generate_constraint_mesh(constraints, field='x', pad_size=2):
        nodes = []
        for c in constraints:
            nodes.append([[nc.master_node, nc.slave_node] for nc in c.node_constraints])
        
        nodes = [a for b in nodes for a in b]
        nodes_pairs = [n for n in nodes if n[1] is not None]
        
        nodes = [a for b in nodes_pairs for a in b]
        node_pos = np.vstack([conv_fun(getattr(node, field))[:3] for node in nodes])

        nodes = [n.label for n in nodes]
        edges = np.vstack([[ix*2, ix*2+1] for ix in range(len(nodes_pairs))])

        # # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(edges.shape[0], int) * 2
        padding[:] = pad_size
        edges_w_padding = np.vstack((padding, edges.T)).T
        mesh = pv.PolyData(node_pos, edges_w_padding)
        return mesh
        
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
    
    def generate_node_mesh(nodes, field='x', pad_size=2):
        # Coordinates of nodes
        node_pos = np.vstack([conv_fun(getattr(node, field))[:3] for node in nodes])
        return node_pos
    
    
    tmat_colors = ['#ff0000', '#00ff00', '#0000ff']
    
    scalar_bar_settings = dict(
        title_font_size=20,
        label_font_size=16,
        n_labels=4,
        italic=False,
        color='black',
        fmt="%.2e",
        font_family="arial"
    )
    
    scalar_bar_settings.update(colorbar_opts)
    
    if vals is None and val_fun is None and 'color' not in el_opts:
        el_opts['color'] = '#44ff88'

    if val_fun is not None:
        if type(val_fun) is str:
            vstr = val_fun + ''
            if 'title' not in colorbar_opts:
                colorbar_opts['title'] = vstr + ''

            val_fun = lambda el: getattr(el, vstr)

        vals = np.vstack([val_fun(el) for el in elements])

    scalars = dict(undeformed=None, deformed=None)
    
    show_scalarbar = dict(undeformed=False, deformed=False)
    show_scalarbar.update({key: True for key in vals_on})
    scalars.update({key: vals for key in vals_on})
    
    if vals is not None:
        if clim is None:
            clim = [np.min(vals[~np.isnan(vals)]), np.max(vals[~np.isnan(vals)])]
        
        if np.min(vals[~np.isnan(vals)])<0:     #adjust for negative values
            # print(vals[~np.isnan(vals)])
            shift = -np.min(vals[~np.isnan(vals)])
        else:
            shift = 0
            
        vals = vals + shift
        new_annotations = {(c+shift): c for c in clim}
        
        # TODO: SUUPER PATCHY - fix later
        if "fmt" in scalar_bar_settings:
            fmt = scalar_bar_settings["fmt"].split('%.')[1][:-1]
        else:
            fmt = '2'
                
        annotate_vals = {(c+shift): f'{annotate_vals[c]:.{fmt}f}' for c in annotate_vals}
        annotate_vals.update(new_annotations) 
        clim = np.array(clim) + shift    
        cmap = pv.LookupTable(cmap=cmap, scalar_range=clim, annotations=annotate_vals,
                              nan_color='#dddddd') 
  

    else:
        cmap = None
        scalar_bar_settings = {}
        show_scalarbar = dict(undeformed=False, deformed=False)

    el_opts['clim'] = clim
    def_el_opts['clim'] = clim

    canvas_settings = dict(background_color='white')
    tmat_settings = dict(show_edges=False)    

    nodelabel_settings = dict(always_visible=True)
    elementlabel_settings = dict(always_visible=True, text_color='blue', shape_color='white', shape_opacity=0.4)
    
    def_el_settings = dict(
        scalars=scalars['deformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        line_width=4,
        lighting=True,
        cmap=cmap,
        show_scalar_bar=show_scalarbar['deformed'],
        color='#ee8899'
        )
    
    el_settings = dict(
        scalars=scalars['undeformed'],
        render_lines_as_tubes=True,
        style='wireframe',
        lighting=True,
        line_width=4,
        cmap=cmap,
        show_scalar_bar=show_scalarbar['undeformed'],
        )

    
    node_settings = dict(
        render_points_as_spheres=True,
        color='magenta',
        lighting=True,
        point_size=5
    )  
    
    def_el_settings.update(def_el_opts)
    canvas_settings.update(canvas_opts)
    el_settings.update(el_opts)
    node_settings.update(node_opts)
    tmat_settings.update(tmat_opts)
    nodelabel_settings.update(nodelabel_opts)
    elementlabel_settings.update(elementlabel_opts)
    
    rel_constraint_settings = dict(el_settings)
    def_rel_constraint_settings = dict(def_el_settings)
    
    rel_constraint_settings.update(constraint_opts)
    def_rel_constraint_settings.update(def_constraint_opts)
    
    
    if pl is None:
        pl = pv.Plotter()
        
    mesh = generate_mesh(elements,'x0')

    if 'undeformed' in plot_states:
        if thickness_scaling is not None:
            grouped_els, group_ixs = group_elements(elements)
            lw0 = el_settings['line_width']*1.0
            for sec in grouped_els:
                el_settings['line_width'] = lw0*thickness_scaling(sec)

                if 'undeformed' in vals_on:
                    el_settings['scalars'] = scalars['undeformed'][group_ixs[sec]]

                pl.add_mesh(generate_mesh(grouped_els[sec],'x0'), 
                             scalar_bar_args=scalar_bar_settings, annotations=annotate_vals, 
                            **el_settings)     
        else:
            pl.add_mesh(mesh, scalar_bar_args=scalar_bar_settings, 
                        annotations=annotate_vals, **el_settings)

    if 'deformed' in plot_states:
        if thickness_scaling is not None:
            grouped_els, group_ixs = group_elements(elements)
            lw0 = def_el_settings['line_width']*1.0
            for sec in grouped_els:
                def_el_settings['line_width'] = lw0*thickness_scaling(sec)
                
                if 'deformed' in vals_on:
                    el_settings['scalars'] = scalars['deformed'][group_ixs[sec]]

                pl.add_mesh(generate_mesh(grouped_els[sec], 'x'), annotations=annotate_vals,
                            scalar_bar_args=scalar_bar_settings, **def_el_settings)     
        else:
            pl.add_mesh(generate_mesh(elements, 'x'), annotations=annotate_vals,
                        scalar_bar_args=scalar_bar_settings, **def_el_settings)
    
    if 'relative' in plot_constraints:
        fields = {'undeformed':'x0', 'deformed':'x'}
        settings = {'undeformed': rel_constraint_settings,
                    'deformed': def_rel_constraint_settings}
        for state in constraints_on:
            mesh = generate_constraint_mesh(eldef.constraints, field=fields[state])
            pl.add_mesh(mesh, **settings[state])     

       
    
    if plot_nodes is not False:
        if plot_nodes is True:
            nodes_to_plot = nodes*1
        else:
            nodes_to_plot = [node for node in nodes if node in plot_nodes]
            
        if 'undeformed' in plot_node_states:
            pl.add_points(generate_node_mesh(nodes_to_plot, 'x0'), **node_settings)
        if 'deformed' in plot_node_states:
            pl.add_points(generate_node_mesh(nodes_to_plot, 'x'), **node_settings)

    if node_labels is not False:
        if node_labels is not True:
            nodes = [node for node in nodes if node in node_labels]
            
        lbl = [node_label_fun(node) for node in nodes]
        
        if 'deformed' in plot_states:
            pl.add_point_labels(np.vstack([node.x[:3] for node in nodes]), lbl, **nodelabel_settings)
        else:
            pl.add_point_labels(np.vstack([node.x0[:3] for node in nodes]), lbl, **nodelabel_settings)            
            
    if element_labels  is not False:
        if element_labels is not True:
            elements_labeled = [element for element in elements if element.label in element_labels]
        else:
            elements_labeled = elements
        
        lbl = [element_label_fun(el) for el in elements_labeled]

        if 'deformed' in plot_states:
            pl.add_point_labels(np.vstack([conv_fun(el.get_cog(deformed=True)) for el in elements_labeled]), lbl,  **elementlabel_settings)
        else:
            pl.add_point_labels(np.vstack([conv_fun(el.get_cog(deformed=False)) for el in elements_labeled]), lbl, **elementlabel_settings)            
            
        
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
    #     pl.add_scalar_bar(**scalar_bar_settings)
    
    if elements[0].domain == '2d':
        pl.view_xy()
    else:
        pl.view_isometric()
    
    if view is not None:
        if view in ['xy', 'top']:
            pl.view_xy()
        if view in ['yz', 'front']:
            pl.view_yz()
        if view in ['xz', 'side']:
            pl.view_xz()
        
    
    pl.show_axes()
    if show:
        pl.show()

    return pl


