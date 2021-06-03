# -*- coding: utf-8 -*-
import numpy as np
from .node import *
from .element import *
from .section import *
from scipy.linalg import null_space as null
from ..general import ensure_list, compatibility_matrix as compmat, lagrange_constrain, gdof_ix_from_nodelabels
from copy import deepcopy as copy

class ElDef:
    def __init__(self, nodes, elements, constraints=None, constraint_type='none', domain='3d', features=None, assemble=True):
        self.nodes = nodes
        self.elements = elements
        self.assign_node_dofcounts()
        self.k, self.m, self.c, self.kg = None, None, None, None
        self.domain = domain
        self.dim = 2 if domain=='2d' else 3

        if set([el.domain for el in self.elements]) != set([domain]):
            raise ValueError('Element domains has to match ElDef/Part/Assembly.')
        
        # Constraints
        self.constraints = constraints 
        self.constraint_type = constraint_type
        self.dof_pairs = self.constraint_dof_ix()               #PATCH for compatibility with nlfe2d module
        self.gdof_ix_from_nodelabels = lambda node_labels, dof_ix: gdof_ix_from_nodelabels(self.get_node_labels(), node_labels, dof_ix=dof_ix)  #PATCH for compatibility with nlfe2d module
        
        if len(set(self.get_node_labels()))!=len(self.get_node_labels()):
            raise ValueError('Non-unique node labels defined.')
        
        if constraints is not None:
            self.B = self.compatibility_matrix()
            self.L = null(self.B)
        else:
            self.B = None
            self.L = None   

        self.constrained_dofs = self.dof_pairs[self.dof_pairs[:,1]==None, 0]
        # self.unconstrained_dofs = np.delete(np.arange(0, np.shape(self.B)[1]), self.constrained_dofs)
      
        if features is None:
            features = []

        self.features = features
        self.feature_mats = self.global_matrices_from_features()

        # Update global matrices and vectors
        self.assign_node_dofcounts()
        self.assign_global_dofs()

        self.update_tangent_stiffness()
        self.update_internal_forces()
        self.update_mass_matrix()
        self.c = self.feature_mats['c']

        if assemble:
            self.assemble()

    @property
    def n_dofs(self):
        return np.sum([node.ndofs for node in self.nodes])


    def global_matrices_from_features(self):
        n_dofs = np.shape(self.B)[1]
        feature_mats = dict(k=np.zeros([n_dofs, n_dofs]), 
                            c=np.zeros([n_dofs, n_dofs]), 
                            m=np.zeros([n_dofs, n_dofs]))

        for feature in self.features:
            ixs = np.array([self.node_dof_lookup(node_label, dof_ix) for node_label, dof_ix in zip(feature.node_labels, feature.dof_ixs)])      
            feature_mats[feature.type][np.ix_(ixs, ixs)] = feature.matrix

        return feature_mats

    def node_dof_lookup(self, node_label, dof_ix):
        return self.nodes[np.where(self.get_node_labels() == node_label)[0][0]].global_dof_ixs[dof_ix]
    

    # CORE METHODS
    def __str__(self):
        return f'BEEF ElDef ({len(self.nodes)} nodes, {len(self.elements)} elements)'

    def __repr__(self):
        return f'BEEF ElDef ({len(self.nodes)} nodes, {len(self.elements)} elements)'

    # ADDITIONAL
    def plot(self, **kwargs):       
        from ..plot import plot_elements 
        return plot_elements(self.elements, **kwargs)      
    
    # ASSIGNMENT AND PREPARATION METHODS
    def assemble(self, constraint_type=None):
        if constraint_type is None:
            constraint_type = self.constraint_type

        self.assign_node_dofcounts() # ? 
        self.assign_global_dofs()
        self.m, self.c, self.k, self.kg = self.global_element_matrices(constraint_type=constraint_type)

    def discard_unused_elements(self): #only elements connected to two nodes are kept
        discard_ix = []
        for element in self.elements:
            if (element.nodes[0] not in self.nodes) or (element.nodes[1] not in self.nodes):
                discard_ix.append(self.elements.index(element))

        self.elements = [self.elements[ix] for ix in range(len(self.elements)) if ix not in discard_ix]

    def arrange_nodes(self, nodelabels):
        self.nodes = self.get_nodes(nodelabels)

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
        
    
    # NODE/DOF LOOKUP AND BOOKKEEPING METHODS      
    def all_ndofs(self):
        return np.array([node.ndofs for node in self.nodes])
    
    def in_part(self, nodelabels):
        return np.isin(nodelabels, self.get_node_labels())

    def in_list(self, nodelabels):
        return np.isin(self.get_node_labels(), nodelabels)

    def get_node_labels(self):
        return np.array([node.label for node in self.nodes])
    
    def get_node(self, nodelabel):
        return self.nodes[self.nodes.index(int(nodelabel))]

    def get_nodes(self, nodelabels):
        return [self.nodes[self.nodes.index(int(nodelabel))] for nodelabel in nodelabels]
    
    def all_dof_ixs(self):
        # Ready for elements with fewer dofs per node!
    
        n_dofs = 0
        
        for node_label in self.get_node_labels():
            n_dofs += max([el.dofs_per_node for el in self.elements_with_node(node_label, return_node_ix=False)])
            
        dof_ix = np.arange(0, int(n_dofs))
        return dof_ix  
    
    def get_node_subset(self, nodes):
        subset = copy(self)
        subset.nodes = [node for node in self.nodes if node in nodes]
        subset.discard_unused_elements()
        subset.assign_global_dofs()
        return subset
        
    def get_element_subset(self, elements):
        subset = copy(self)
        subset.elements = [element for element in self.elements if element in elements]
        subset.nodes = list(set([item for sublist in [el.nodes for el in subset.elements] for item in sublist]))
        subset.assign_node_dofcounts()
        subset.assign_global_dofs()
        return subset

    def get_elements_with_nodes(self, node_label_list, return_only_labels=False):
        els = []
        for element in self.elements:
            nodes_in_use = [node_label for node_label in element.get_nodelabels() if node_label in node_label_list]
            if len(nodes_in_use)>0:
                els.append(element)
        
        if return_only_labels:
            els = [el.label for el in els]
        
        return els
    
    
    def element_labels(self):
        element_labels = np.hstack([element.label for element in self.elements])
        return element_labels


    def get_element(self, element_label):
        if element_label in self.elements:
            ix = np.where(self.element_labels()==element_label)[0][0].astype(int)
            return self.elements[ix]
        else:
            return None
              
        
    def node_label_to_node_ix(self, node_label):
        node_ix = np.where(self.get_node_labels()==node_label)[0][0].astype(int)
        return node_ix
    
    
    def node_label_to_dof_ix(self, node_label): 
        node_ix = self.node_label_to_node_ix(node_label)   
        lower_dofs = sum(self.ndofs[:node_ix])
        dof_ix = lower_dofs + np.arange(0, self.ndofs[node_ix])
        return dof_ix

    # MODIFIERS
    def deform(self, u):
        for node in self.nodes:
            node.u = u[node.global_dofs]
            node.x = node.x0 + node.u

        for element in self.elements:
            element.update_geometry()
            element.update()

        self.update_tangent_stiffness()
        self.update_internal_forces()

    def deform_linear(self, u):
        for node in self.nodes:
            node.u = u[node.global_dofs]
            node.x = node.x0 + node.u

        for element in self.elements:
            element.update_linear()    

        self.update_internal_forces(u)      # on part level (element internal forces are dealt with intristicly by update function above)
    

    def update_all_geometry(self):
        for element in self.elements:
            element.update_geometry()

    def update_internal_forces(self, u=None):       # on part level
        if u is None:
            u = np.zeros([self.n_dofs])

        if hasattr(self, 'feature_mats'):
            self.q = self.feature_mats['k'] @ u   
        else:
            self.q = u*0

        for el in self.elements:
            ixs = np.hstack([el.nodes[0].global_dofs, el.nodes[1].global_dofs])
            self.q[ixs] += el.q

    def update_tangent_stiffness(self):
        self.k = self.feature_mats['k']*1          
        
        for el in self.elements:
            self.k[np.ix_(el.global_dofs, el.global_dofs)] += el.k

    
    def update_mass_matrix(self):
        self.m = self.feature_mats['m']*1   

        for el in self.elements:
            self.m[np.ix_(el.global_dofs, el.global_dofs)] += el.m
            

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
            return None

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
    
    
    def constraint_dof_count(self):
        count = 0
        for constraint in self.constraints:
            n_dofs = [len(nc.dof_ix) for nc in constraint.node_constraints]
            count = count + np.sum(n_dofs)
        
        return count
    
    
    def compatibility_matrix(self):
        dof_pairs = self.constraint_dof_ix()
        ndim = np.sum(self.ndofs)
        compat_mat = compmat(dof_pairs, ndim)
        
        return compat_mat   
    
    # TRANSFORMATION METHODS
    def local_node_csys(self, node_label, average_ok=True):
        elements, node_ix = self.elements_with_node(node_label, merge_parts=True)
        
        t_mats = [el.get_tmat(reps=1) for el in elements]
        
        if (len(elements)>1) and not average_ok:
            raise ValueError('More elements connected to node. Use average_ok=True if averaging of transformation mats is fine.')
        
        t_mat = blkdiag(np.mean(np.stack(t_mats, axis=2), axis=2), 2)
        
        return t_mat

    def get_kg(self, N=None):
        ndim = len(self.get_node_labels())*6
        kg_eldef = np.zeros([ndim, ndim])
        
        for el in self.elements:
            if el.nodes[1].global_dofs is None:
                print(el.nodes[1].global_dofs)
                print(el.nodes[1])
                
            glob_dofs = np.r_[el.nodes[0].global_dofs, el.nodes[1].global_dofs].astype(int)
            local_dofs = np.r_[0:len(el.nodes[0].global_dofs), 6:6+len(el.nodes[1].global_dofs)]    #added for cases where one node in element is not present in self.nodes, check speed effect later

            kg_eldef[np.ix_(glob_dofs, glob_dofs)] += el.get_kg(N=N)[np.ix_(local_dofs, local_dofs)]
            
            if np.any(np.isnan(el.get_kg()[np.ix_(local_dofs, local_dofs)])):
                print(el)

                        
        return kg_eldef


    # GENERATE OUTPUT FOR ANALYSIS    
    def global_element_matrices(self, constraint_type=None):
        ndim = len(self.all_dof_ixs())
        
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
        
        
    def export_matrices(self, part_ix=None):
        
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
    def __init__(self, node_matrix, element_matrix, sections=None, constraints=None, element_types=None, left_handed_csys=False, **kwargs):      
        if element_types == None:
            element_types = ['beam']*element_matrix.shape[0] # assume that all are beam

        nodes, elements = create_nodes_and_elements(node_matrix, element_matrix, sections=sections, left_handed_csys=left_handed_csys, element_types=element_types)
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

def create_nodes_and_elements(node_matrix, element_matrix, sections=None, left_handed_csys=False, element_types=None):
    nodes = create_nodes(node_matrix)
    node_labels = np.array([node.label for node in nodes])
    dim = node_matrix.shape[1]-1

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

        if element_types[el_ix] == 'beam':
            if dim==3:
                elements[el_ix] = BeamElement3d([nodes[ix1], nodes[ix2]], label=int(element_matrix[el_ix, 0]), section=sections[el_ix], left_handed_csys=left_handed_csys)
            else:
                elements[el_ix] = BeamElement2d([nodes[ix1], nodes[ix2]], label=int(element_matrix[el_ix, 0]), section=sections[el_ix])

        elif element_types[el_ix] == 'bar':
            # Currently only added to enable extraction of Kg for bars - not supported otherwise (assumed as beams otherwise)
            elements[el_ix] = BarElement3d([nodes[ix1], nodes[ix2]], label=int(element_matrix[el_ix, 0]), section=sections[el_ix], left_handed_csys=left_handed_csys)
        
    return nodes, elements