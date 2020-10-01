# -*- coding: utf-8 -*-
import numpy as np
from . import plotters
from .node import *
from .element import *
from .section import *
from scipy.linalg import null_space as null
from ..general import ensure_list

class ElDef:
    def __init__(self, nodes, elements, constraints=None, constraint_type='lagrange', domain='3d'):
        self.nodes = nodes
        self.elements = elements
        self.assign_node_dofcounts()
        self.k, self.m, self.c, self.kg = None, None, None, None
        self.domain = domain

        if set([el.domain for el in self.elements]) != set([domain]):
            raise ValueError('Element domains has to match ElDef/Part/Assembly.')
        
        # Constraints
        self.constraints = constraints 
        self.constraint_type = constraint_type
        
        if len(set(self.get_node_labels()))!=len(self.get_node_labels()):
            raise ValueError('Non-unique node labels defined.')
        
        if constraints is not None:
            self.B = self.compatibility_matrix()
            self.L = null(self.B)
        else:
            self.B = None
            self.L = None       

    # CORE METHODS
    def __str__(self):
        return f'BEEF ElDef ({len(self.nodes)} nodes, {len(self.elements)} elements)'

    def __repr__(self):
        return f'BEEF ElDef ({len(self.nodes)} nodes, {len(self.elements)} elements)'

    # ADDITIONAL
    def plot(self, **kwargs):        
        return _plotters.plot_eldef_3d(self, **kwargs)      
    
    def update_u_plot(self, u_handle, u):
        _plotters.update_plot_u_eldef(self, u_handle, u)

    # ASSIGNMENT AND PREPARATION METHODS
    def assemble(self):
        self.assign_global_dofs()
        self.m, self.c, self.k, self.kg = self.global_element_matrices(constraint_type=self.constraint_type)

    def assign_global_dofs(self):
        for node in self.nodes:
            node.global_dofs = self.node_label_to_dof_ix(node.label)

    def assign_node_dofcounts(self):
        for node in self.nodes:
            els = self.elements_with_node(node.label, return_node_ix=False)
            if els != []:
                node.ndofs = els[0].dofs_per_node
            else:
                node.ndofs = 0
            
        self.ndofs = self.all_ndofs()
        
            
    def all_ndofs(self):
        return np.array([node.ndofs for node in self.nodes])
    
    
    def get_node_labels(self):
        return np.array([node.label for node in self.nodes])
    
    # NODE/DOF LOOKUP AND BOOKKEEPING METHODS      
    def get_node(self, node_label):
        return [node for node in self.nodes if node.label==node_label][0]
    
    def all_dof_ixs(self):
        # Ready for elements with fewer dofs per node!
    
        n_dofs = 0
        
        for node_label in self.get_node_labels():
            n_dofs += max([el.dofs_per_node for el in self.elements_with_node(node_label, return_node_ix=False)])
            
        dof_ix = np.arange(0, int(n_dofs))
        return dof_ix  
    
    
    def get_elements_with_nodes(self, node_label_list, return_only_labels=False):
        els = []
        for element in self.elements:
            nodes_in_use = [node_label for node_label in element.node_labels() if node_label in node_label_list]
            if len(nodes_in_use)>0:
                els.append(element)
        
        if return_only_labels:
            els = [el.label for el in els]
        
        return els
    
    
    def element_labels(self):
        element_labels = np.hstack([element.label for element in self.elements])
        return element_labels


    def get_element(self, element_label):
        ix = np.where(self.element_labels()==element_label)[0][0].astype(int)
        return self.elements[ix]
              
        
    def node_label_to_node_ix(self, node_label):
        node_ix = np.where(self.get_node_labels()==node_label)[0][0].astype(int)
        return node_ix
    
    
    def node_label_to_dof_ix(self, node_label): 
        node_ix = self.node_label_to_node_ix(node_label)   
        lower_dofs = sum(self.ndofs[:node_ix])
        dof_ix = lower_dofs + np.arange(0, self.ndofs[node_ix])
        return dof_ix


    # GET METHODS
    def get_sections(self):
        sections = []
        
        for el in self.elements:
            if el.section not in sections: sections.append(el.section)
        
        return sections
    
    def get_tmats(self):
        return [el.tmat for el in self.elements]
        
    
    # CONSTRAINT METHODS   
    def constraint_dof_ix(self):        
        if self.constraints is None:
            raise ValueError("Can't output constraint DOF indices as no constraints are given.")

        c_dof_ix = []
        
        for constraint in self.constraints:   
            for node_constraint in constraint.node_constraints:
                dofs = np.array(node_constraint.dof_ix)
                dof_ixs = self.node_label_to_dof_ix(node_constraint.master_node)[dofs]
                
                if node_constraint.slave_node is not None:
                    conn_dof_ixs = self.node_label_to_dof_ix(node_constraint.slave_node)[dofs]
                else:
                    conn_dof_ixs = [None]*len(dof_ixs)

                dof_ixs = np.vstack([dof_ixs, conn_dof_ixs]).T
                c_dof_ix.append(dof_ixs)
                
        c_dof_ix = np.vstack(c_dof_ix)
        return c_dof_ix
    
    
    # def unwrap_primal_constraint_dofs(self):
    #     dof_pairs = self.constraint_dof_ix()
        
    #     mask_gnd = dof_pairs[:,1]==None     #mask for grounded
    #     mask_slave = dof_pairs[:,1]!=None   #mask for node-to-node

    #     dofs_to_remove = np.unique(np.hstack([dof_pairs[mask_gnd,0], dof_pairs[mask_slave,1]]))    #grounded master nodes and slave nodes
    #     equal_dofs = dof_pairs[mask_slave,:].astype(int)

    #     return dofs_to_remove, equal_dofs
    
    
    def constraint_dof_count(self):
        count = 0
        for constraint in self.constraints:
            n_dofs = [len(nc.dof_ix) for nc in constraint.node_constraints]
            count = count + np.sum(n_dofs)
        
        return count
    
    
    def compatibility_matrix(self):
        dof_pairs = self.constraint_dof_ix()
        ndim = np.sum(self.ndofs)
        compat_mat = compatibility_matrix(dof_pairs, ndim)
        
        return compat_mat   
    
    # TRANSFORMATION METHODS
    def local_node_csys(self, node_label, average_ok=True):
        elements, node_ix = self.elements_with_node(node_label, merge_parts=True)
        
        t_mats = [el.get_tmat(reps=1) for el in elements]
        
        if (len(elements)>1) and not average_ok:
            raise ValueError('More elements connected to node. Use average_ok=True if averaging of transformation mats is fine.')
        
        t_mat = blkdiag(np.mean(np.stack(t_mats, axis=2), axis=2), 2)
        
        return t_mat

    def get_kg(self):
        ndim = len(self.get_node_labels())*6

        geometric_stiffness = np.zeros([ndim, ndim])
        
        for el in self.elements:
            glob_dofs = np.r_[el.nodes[0].global_dofs, el.nodes[1].global_dofs].astype(int)
            local_dofs = np.r_[0:len(el.nodes[0].global_dofs), 6:6+len(el.nodes[1].global_dofs)]    #added for cases where one node in element is not present in self.nodes, check speed effect later
  
            geometric_stiffness[np.ix_(glob_dofs, glob_dofs)] = geometric_stiffness[np.ix_(glob_dofs, glob_dofs)] + el.get_kg()[np.ix_(local_dofs, local_dofs)]

                        
        return geometric_stiffness


    # GENERATE OUTPUT FOR ANALYSIS    
    def global_element_matrices(self, constraint_type=None):
        ndim = len(self.nodes)*6
        
        mass = np.zeros([ndim, ndim])
        stiffness = np.zeros([ndim, ndim])
        geometric_stiffness = np.zeros([ndim, ndim])
        damping = np.zeros([ndim, ndim])
        
        # Engineering features (springs, dashpots, point masses, etc.) not implemented
        # Should add possibility to add spring/dashpot between two nodes also.

        for el in self.elements:
            dof_ix1, dof_ix2 = el.nodes[0].global_dofs, el.nodes[1].global_dofs
            dof_range = np.r_[dof_ix1, dof_ix2]
            T = el.tmat
            
            mass[np.ix_(dof_range, dof_range)] += el.get_m()
            stiffness[np.ix_(dof_range, dof_range)] += el.get_k()
            geometric_stiffness[np.ix_(dof_range, dof_range)] += el.get_kg()
        
        removed_ix = None  
        keep_ix = None
        
        if self.constraints != None:  
            if constraint_type == 'lagrange':
                dof_pairs = self.constraint_dof_ix()
                mass = lagrange_constrain(mass, dof_pairs)
                stiffness = lagrange_constrain(stiffness, dof_pairs)
                geometric_stiffness = lagrange_constrain(geometric_stiffness, dof_pairs)
            
            elif constraint_type == 'primal':
                stiffness = self.L.T @ stiffness @ self.L
                damping = self.L.T @ damping @ self.L
                mass = self.L.T @ mass @ self.L
                geometric_stiffness = self.L.T @ geometric_stiffness @ self.L
                
        return mass, damping, stiffness, geometric_stiffness
    
        
    # MISC METHODS
    def get_max_dim(self, max_of_max=True):
        n,__ = self.export_eldef()
        coors = n[:,1:]
        
        max_dim = np.ptp(coors, axis=0)
        
        if max_of_max:
            max_dim = np.max(max_dim)
            
        return max_dim
    
    
    def elements_with_node(self, node_label, merge_parts=True, return_node_ix=True):
        elements = [el for el in self.elements if node_label in el.nodes]  
        
        node_ix = [np.where(el.nodes==node_label)[0] for el in elements]
            
        if return_node_ix:
            return elements, node_ix
        else:
            return elements     
        
        
    def export_eldef(self, part_ix=None):
        
        if part_ix is None:
            els = self.elements
        else:
            els = self.parts[part_ix].elements
        
        n_elements = len(els)
        element_matrix = np.zeros([n_elements, 3])
        node_matrix = []
        
        for el_ix, el in enumerate(els):
            node1 = el.nodes[0]
            node2 = el.nodes[1]
            
            element_matrix[el_ix, :] = np.array([el.label, node1.label, node2.label])
            node_matrix.append(np.hstack([node1.label, node1.coordinates]))
            node_matrix.append(np.hstack([node2.label, node2.coordinates]))
        
        node_matrix = np.vstack(node_matrix)
        __, keep_ix = np.unique(node_matrix[:,0], return_index=True)
        node_matrix = node_matrix[keep_ix, :]
        
        return node_matrix, element_matrix
    

class Assembly(ElDef):
    def __init__(self, parts, **kwargs):
        domains = [part.domain for part in parts]
        if not all([domain==domains[0] for domain in domains]):
            raise ValueError('Cannot combine parts in different domains. Use either 2d or 3d for all parts!')

        super().__init__(self.all_nodes(parts), self.all_elements(parts), **kwargs)
        self.parts = parts        


    def all_nodes(self, parts, sort_by_label=True):
        nodes = []
        for part in parts:
            part_nodes = [el.nodes for el in part.elements]
            nodes.append(list(set([item for sublist in part_nodes for item in sublist])))

        nodes = list(set([item for sublist in nodes for item in sublist]))
            
        if sort_by_label:
            node_labels = [node.label for node in nodes]
            ix = np.argsort(node_labels)
            nodes = [nodes[i] for i in ix]
        
        return nodes
    
    def all_elements(self, parts):
        return [sublist for l in [part.elements for part in parts] for sublist in l]       
            
    
class Part(ElDef):
    def __init__(self, node_matrix, element_matrix, sections=None, constraints=None, **kwargs):      
        nodes, elements = create_nodes_and_elements(node_matrix, element_matrix, sections=sections)
        if node_matrix.shape[1] == 3:
            domain = '2d'
        elif node_matrix.shape[1] == 4:
            domain = '3d'

        super().__init__(nodes, elements, constraints, domain=domain, **kwargs)


def create_nodes(node_matrix):
    n_nodes = node_matrix.shape[0]
    nodes = [None]*n_nodes

    for node_ix in range(0, n_nodes):
        nodes[node_ix] = Node(node_matrix[node_ix, 0], node_matrix[node_ix, 1:])
    
    return nodes

def create_nodes_and_elements(node_matrix, element_matrix, sections=None):
    nodes = create_nodes(node_matrix)
    node_labels = np.array([node.label for node in nodes])
    
    n_els = element_matrix.shape[0]
    elements = [None]*n_els
    
    if sections is None:
        sections = [Section(name='Generic')]*n_els
    else:
        if sections is not None:
            sections = ensure_list(sections)
        if len(sections)==1:
            sections = sections*n_els
                
    for el_ix in range(0, n_els):
        el_node_label = element_matrix[el_ix, 1:]

        ix1 = np.where(node_labels==el_node_label[0])[0][0]
        ix2 = np.where(node_labels==el_node_label[1])[0][0]

        elements[el_ix] = BeamElement3d([nodes[ix1], nodes[ix2]], label=element_matrix[el_ix, 0], section=sections[el_ix])
    return nodes, elements