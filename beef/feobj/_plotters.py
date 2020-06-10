# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:51:28 2020

@author: knutankv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from .plot import *

def equal_3d(ax=plt.gca()):
     x_lims = np.array(ax.get_xlim())
     y_lims = np.array(ax.get_ylim())
     z_lims = np.array(ax.get_zlim())
 
     x_range = np.diff(x_lims)
     y_range = np.diff(y_lims)
     z_range = np.diff(z_lims)
 
     max_range = np.max([x_range,y_range,z_range])/2
 
     ax.set_xlim(np.mean(x_lims) - max_range, np.mean(x_lims) + max_range)
     ax.set_ylim(np.mean(y_lims) - max_range, np.mean(y_lims) + max_range)
     ax.set_zlim(np.mean(z_lims) - max_range, np.mean(z_lims) + max_range)
     # ax.set_aspect(1)
     
     return ax


def plot_eldef_3d(self, u=None, color='Gray', plot_nodes=False, node_labels=False, 
         element_labels=False, fig=None, ax=None, element_settings={},
         node_settings={}, node_label_settings={}, element_label_settings={}, 
         constraints=False, load_scaling=1, tmat_scaling=None, element_orientation=False):
        
    e_dict = {'color': 'DarkGreen', 'alpha': 1}
    e_dict.update(**element_settings)

    n_dict = {'color':'Black', 'linestyle':'', 'marker':'.', 'markersize':4, 'alpha':0.8}
    n_dict.update(**node_settings)

    l_n_dict = {'color':'Black', 'fontsize': 8, 'fontweight':'normal'}
    l_n_dict.update(**node_label_settings)
    
    l_e_dict = {'color':'LimeGreen', 'fontsize': 8, 'fontweight':'bold', 'style':'italic'}
    l_e_dict.update(**element_label_settings)

    if tmat_scaling is None:
        tmat_scaling = self.get_max_dim()*0.05
    
    if ax is None and fig is None:
        fig = plt.gcf()
            
    if ax == None:
        ax = fig.gca(projection='3d')
        ax = fig.gca()
    
    h = [None]*len(self.elements)
    for ix, el in enumerate(self.elements):
        xy = np.vstack([node.coordinates for node in el.nodes])
        
        if u is not None: 
            xy[0, :] += u[self.node_label_to_dof_ix(el.nodes[0].label)[0:3]]
            xy[1, :] += u[self.node_label_to_dof_ix(el.nodes[1].label)[0:3]]

        h[ix] = plt.plot(xy[:,0], xy[:,1], xy[:,2], **e_dict)[0]
        
        if plot_nodes:
            plt.plot(xy[:,0], xy[:,1], xy[:,2], **n_dict)

        if element_orientation:
            h_tmat = plot_tmat(ax, el.cog(), el.tmat(reps=1)*tmat_scaling)
            
        if element_labels:
            cog = el.cog()
            ax.text(cog[0], cog[1], cog[2], el.label, **l_e_dict)
        
            
    if node_labels:
        for node in self.nodes:
            ax.text(node.coordinates[0], node.coordinates[1], node.coordinates[2], node.label, **l_n_dict)


    # CONSTRAINTS  
    if constraints:
        for constraint in self.constraints:
            c_node_labels = np.array([[nc.master_node, nc.slave_node] for nc in constraint.node_constraints])
            c_node_labels = c_node_labels.flatten()
            c_node_labels = c_node_labels[c_node_labels!=None]

            c_length = len(c_node_labels)
            marker = [None]*c_length
            
            # Assign marker based on type of constraint
            if constraint.type == 'node-to-ground':
                for nc_ix, node_constraint in enumerate(constraint.node_constraints):
                    if np.array_equal(node_constraint.dof_ix, [0,1,2,3,4,5]):
                        marker[nc_ix] = 'X'
                    elif np.array_equal(node_constraint.dof_ix, [0,1,2]):
                        marker[nc_ix] = 'o'    
                    elif np.array_equal(node_constraint.dof_ix, [3,4,5]):
                        marker[nc_ix] = 's'    
                    elif np.array_equal(node_constraint.dof_ix, [2]):
                        marker[nc_ix] = '^'  
                    else:
                        marker[nc_ix] = 'D'
                
                color = 'DimGray'
            else:
                marker=['*']*c_length
                color = 'DimGray'

            # Plot each type of marker group by itself (limitation in scatter function)
            coordinates = np.vstack([self.get_node(nl).coordinates for nl in c_node_labels])
            unique_markers = np.unique(marker)  # or yo can use: np.unique(m)

            for um in unique_markers:
                mask = np.array([ i for i in range(len(marker)) if marker[i] == um ])
                ax.scatter(coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2], marker=um, color=color, s=48)
                    
    equal_3d(ax)
    # ax.grid('off')

    if u is not None:
        u_handle = h
    else:
        u_handle = None
        
    return ax, u_handle


def plot_step_3d(self, disp_element_settings={}, response=True, loads=False, 
                 load_scaling=None, response_scaling=1, t=-1, step_ix=0,
                 animate=False, animation_settings=None,
                 **kwargs):
    disp_e_dict = {'color': 'IndianRed', 'alpha': 1}
    disp_e_dict.update(**disp_element_settings)
    
    if load_scaling is None:
        load_scaling = self.eldef.get_max_dim()*0.05     #5 percent of max extension
    
    if response_scaling is None:
        response_scaling = self.eldef.get_max_dim()*0.05     #5 percent of max extension
        
    step = self.steps[step_ix] 
    ax, __ = self.eldef.plot(**kwargs)
    
    if response:       
        ax, u_handle = self.eldef.plot(u=np.real(step.results['u'][:, t]*response_scaling), ax=ax, element_settings=disp_e_dict, **kwargs)
    else:
        u_handle = None
        
    if loads and (step.loads is not None):
            zf = np.zeros([6,1])
            for load in step.loads:
                scaling = 1/np.max([1,np.max(abs(np.vstack([nl.amplitudes for nl in load.nodeloads])))]) * load_scaling 

                for nodeload in load.nodeloads:
                    if nodeload.local:
                        T = self.eldef.local_node_csys(nodeload.node_label)
                    else:
                        T = np.eye(6)
                    
                    x0 = np.array(self.eldef.get_node(nodeload.node_label).coordinates)
                    x_add = zf*1.0  #zero force
                    x_add[nodeload.dof_ix, 0] = np.array(nodeload.amplitudes)

                    x_add = (T.T @ x_add)[:,0]
                    
                    if not np.all(x_add[0:3] == 0):
                        ax.quiver(x0[0], x0[1], x0[2], x_add[0]*scaling, x_add[1]*scaling, x_add[2]*scaling, color=load.plotcolor,arrow_length_ratio =0.06)
                    if not np.all(x_add[3:] == 0):
                        ax.quiver(x0[0], x0[1], x0[2], x_add[3]*scaling, x_add[4]*scaling, x_add[5]*scaling, color=load.plotcolor, linestyle='-.', arrow_length_ratio =0.06)
                
    ax.set_title('Step {}: {}\n Time index: {} \n Scaling: {}'.format(step_ix+1, step.type.title(), t, response_scaling))
    # plt.axis('off')
    
    if animate:
        if animation_settings is None:
            if step.type == 'static':
                animation_settings = {'full_cycle': False, 'repeats': 10, 'swing':False, 'frames':20}
            elif step.type == 'eigenvalue problem':
                animation_settings = {'full_cycle': True, 'repeats': 10, 'swing':True, 'frames':20}
        
        for sc in frame_creator(**animation_settings):
            self.eldef.update_u_plot(u_handle, step.results['u'][:, t]*response_scaling*sc)
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()
    
    return ax, u_handle


def plot_tmat(ax, x0, tmat):
    colors = ['tab:red', 'tab:blue', 'tab:green']
    h = [None]*3
    for comp in range(0,3):
        h[comp] = ax.quiver(x0[0], x0[1], x0[2], tmat[comp,0], tmat[comp,1], tmat[comp,2], color=colors[comp], arrow_length_ratio=0.06)
   
    return h
            
def update_plot_u_eldef(self, u_handle, u):
    for ix, el in enumerate(self.elements):
        xy = np.vstack([node.coordinates for node in el.nodes])
        
        if u is not None: 
            xy[0,:] += u[self.node_label_to_dof_ix(el.nodes[0].label)[0:3]].real
            xy[1,:] += u[self.node_label_to_dof_ix(el.nodes[1].label)[0:3]].real

        u_handle[ix].set_data(xy[:,0], xy[:,1])
        u_handle[ix].set_3d_properties(xy[:,2])
