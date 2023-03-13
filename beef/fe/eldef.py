'''
FE objects submodule: element definitions
'''

import numpy as np
from .node import *
from .element import *
from .section import *
from scipy.linalg import null_space as null
from ..general import ensure_list, compatibility_matrix as compmat, lagrange_constrain, gdof_ix_from_nodelabels
from ..modal import statespace
from copy import deepcopy as copy

class ElDef:
    '''
    Element definition main class.

    Arguments
    ---------------------------
    nodes : obj
        list of node objects
    elements : obj 
        list of element objects
    constraints : obj
        list of constraints
    features : obj
        list of features (springs, dashpots, point masses)
    include_linear_kg : boolean
        whether or not to include linearized geometric stiffness from specified axial force (N0 in element objects)
    constraint_type : {'none', 'lagrange', or 'primal'}
        constraint to enforce
    domain : {'3d', '2d'}
    assemble : boolean, True
        whether or not to assemble structure automatically upon generation - useful to use False if no computation is used and invertable matrices is not strictly required

    '''

    def __init__(self, nodes, elements, constraints=None, features=None, 
                 include_linear_kg=False, constraint_type='none', domain='3d', 
                 assemble=True):
        self.nodes = nodes
        self.elements = elements
        self.assign_node_dofcounts()
        self.k, self.m, self.c, self.kg = None, None, None, None
        self.domain = domain
        self.dim = 2 if domain=='2d' else 3

        self.include_linear_kg = include_linear_kg

        if set([el.domain for el in self.elements]) != set([domain]):
            raise ValueError('Element domains has to match ElDef/Part/Assembly.')
        
        # Constraints
        self.constraints = constraints 
        self.constraint_type = constraint_type

        # TODO: Currently use this patch for inclusion of nlfe2d - fix.
        self.dof_pairs = self.constraint_dof_ix()    

        if len(set(self.get_node_labels()))!=len(self.get_node_labels()):
            raise ValueError('Non-unique node labels defined.')
        
        if constraints is not None:
            self.B = self.compatibility_matrix()
            self.L = null(self.B)
        else:
            self.B = None
            self.L = None   
        if self.dof_pairs is not None:
            self.constrained_dofs = self.dof_pairs[self.dof_pairs[:,1]==None, 0]
        else: 
            self.constrained_dofs = []
        # self.unconstrained_dofs = np.delete(np.arange(0, np.shape(self.B)[1]), self.constrained_dofs)
      
        if features is None:
            features = []

        # Update global matrices and vectors
        self.assign_node_dofcounts()
        self.assign_global_dofs()

        # Establish features
        self.features = features

        if assemble:
            self.assemble()
            self.update_tangent_stiffness()
            self.update_internal_forces()
            self.update_mass_matrix()

    # TODO: Currently use this patch for inclusion of nlfe2d - fix.
    def gdof_ix_from_nodelabels(self, node_labels, dof_ix):    
        return gdof_ix_from_nodelabels(self.get_node_labels(), node_labels, dof_ix=dof_ix)

    @property
    def ndofs(self):
        '''
        Number of nodes in `ElDef`.
        '''
        return np.sum([node.ndofs for node in self.nodes])


    def get_feature_mats(self, mats=['k', 'c', 'm']):
        '''
        Establish matrices for included features (springs, dashpots, point masses).

        Arguments 
        -------------
        mats : str
            list where the three values 'k', 'c', and 'm' are allowed, 
            specifying what matrices to establish

        Returns
        ----------
        feature_list 
            a list containing the specified feature mats (corresponding to order
            given in input)

        '''
        n_dofs = self.ndofs
        feature_mats = {key: np.zeros([n_dofs, n_dofs]) for key in mats}

        for feature in self.features:
            if feature.type in mats:
                for dof_ix in feature.dofs:
                    ixs = np.array([self.node_dof_lookup(nl, dof_ix=dof_ix) for nl in feature.node_labels]).flatten()
                    feature_mats[feature.type][np.ix_(ixs, ixs)] = feature.matrix

        feature_list = [feature_mats[key] for key in mats]  # return as list with same order as input

        if len(feature_list) == 1:
            feature_list = feature_list[0]

        return feature_list 


    def node_dof_lookup(self, node_label, dof_ix=slice(None)):
        '''
        Return global DOF of a node with specified node_label and dof_ix

        Arguments
        -----------
        node_label : int
            node label of queried node
        dof_ix : int
            index of local dof

        Returns
        -----------
        dof : int
            global dof index (as nodes are stacked in `ElDef`)

        '''
        return self.get_node(node_label).global_dofs[dof_ix]
    

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
        '''
        Method to assemble element definitions according to specified
        options from parent `ElDef` object.

        Arguments
        -------------
        constraint_type : {None, 'lagrange', 'primal'}
            how to constrain system
        
        '''

        if constraint_type is None:
            constraint_type = self.constraint_type

        self.assign_node_dofcounts() # ? 
        self.assign_global_dofs()
        self.ndim = np.sum(self.get_all_ndofs())
        self.m, self.c, self.k, self.kg = self.global_element_matrices(constraint_type=constraint_type)
        
        if self.include_linear_kg:
            self.k = self.k + self.kg
            
    def discard_unused_elements(self):
        '''
        Clear unused elements, such that only elements connected to two nodes are kept.
        '''
        
        discard_ix = []
        for element in self.elements:
            if (element.nodes[0] not in self.nodes) or (element.nodes[1] not in self.nodes):
                discard_ix.append(self.elements.index(element))

        self.elements = [self.elements[ix] for ix in range(len(self.elements)) if ix not in discard_ix]

    def arrange_nodes(self, node_labels):
        '''
        Arrange nodes according to specified list of labels.

        Arguments
        ------------
        node_labels : int
            list of node labels to order according to
        '''
        self.nodes = self.get_nodes(node_labels)

    def assign_global_dofs(self):
        '''
        Assign global dofs (`Node. global_dofs`) for each node based on current stacking in global system.
        '''
        for node in self.nodes:
            node.global_dofs = self.node_label_to_dof_ix(node.label)
        
    def assign_node_dofcounts(self):
        '''
        Assign node dof counts (`Node. ndofs`) for each node based on 
        current stacking in global system. Also update the field `all_ndofs`
        in the `ElDef` object.
        '''
        for node in self.nodes:
            els = self.elements_with_node(node.label, return_node_ix=False)
            if els != []:
                node.ndofs = els[0].dofs_per_node
            else:
                node.ndofs = 0
            
        self.all_ndofs = self.get_all_ndofs()
        
    
    # NODE/DOF LOOKUP AND BOOKKEEPING METHODS      
    def get_all_ndofs(self):
        return np.array([node.ndofs for node in self.nodes])
    
    def in_part(self, node_labels):
        '''
        Indicate whether node with specified node labels are in element definition or not.
        
        Arguments
        -----------
        node_labels : int
            list of node labels to check if is in `ElDef`
            
        Returns 
        ----------
        in_eldef : boolean
            array indicating if the queried node labels are present in `ElDef`

        '''

        return np.isin(node_labels, self.get_node_labels())

    def in_list(self, node_labels):
        '''
        Indicate whether `ElDef` node labels are input list of node labels. Inverse method of `in_part`.
        
        Arguments
        -----------
        node_labels : int
            list of node labels to check
            
        Returns 
        ----------
        in_list : boolean
            array indicating if the nodes (represented by labels) in `ElDef` are present in input node_labels

        '''
        return np.isin(self.get_node_labels(), node_labels)

    def get_node_labels(self):
        '''
        Get a list with all labels of the nodes in the `ElDef`.
        '''

        return np.array([node.label for node in self.nodes])
    
    def get_node(self, node_label):
        '''
        Get Node object from node label.

        Arguments
        -----------
        node_label : int
            label of node to get
        
        Returns
        -------------
        node : obj
            `Node` object corresponding to input label

        '''

        return self.nodes[self.nodes.index(int(node_label))]

    def get_nodes(self, node_labels):
        '''
        Get Node objects from node labels.

        Arguments
        -----------
        nodelabels : int
            list of labels of nodes to get
        
        Returns
        -------------
        nodes : obj
            list of `Node` objects corresponding to input labels

        '''

        return [self.get_node(node_label) for node_label in node_labels]
       
    def get_node_subset(self, nodes):
        '''
        Get a new `ElDef` as a subset from the current based on specified nodes.

        Arguments
        -----------
        nodes : Node obj
            list of `Node` objects to extract

        Returns
        -----------
        subset : ElDef obj
            new `ElDef` object corresponding to specified nodes

        '''

        subset = copy(self)
        subset.nodes = [node for node in self.nodes if node in nodes]
        subset.discard_unused_elements()
        subset.assign_global_dofs()
        return subset
        
    def get_element_subset(self, elements):
        '''
        Get a new `ElDef` as a subset from the current based on specified elements.

        Arguments
        -----------
        elements : Element obj
            list of `Element` objects to extract

        Returns
        -----------
        subset : ElDef obj
            new `ElDef` object corresponding to specified nodes

        '''

        subset = copy(self)
        subset.elements = [element for element in self.elements if element in elements]
        subset.nodes = list(set([item for sublist in [el.nodes for el in subset.elements] for item in sublist]))
        subset.assign_node_dofcounts()
        subset.assign_global_dofs()
        return subset

    def get_elements_with_nodes(self, node_labels, return_only_labels=False):
        '''
        Get elements corresponding to given list of node labels.

        Arguments
        --------------
        node_labels : int
            list of node labels
        return_only_labels : False
            whether or not to return only labels of elements (alternatively list of Element objects is returned)

        Returns
        ----------------
        els : obj
            list of `Element` objects (or element labels if specified)
        '''

        els = []
        for element in self.elements:
            nodes_in_use = [node_label for node_label in element.nodes if node_label in node_labels]
            if len(nodes_in_use)>0:
                els.append(element)
        
        if return_only_labels:
            els = [el.label for el in els]
        
        return els
    
    
    def element_labels(self):
        '''
        Establish list of all element labels as ordered in `ElDef`.
        '''
        element_labels = np.hstack([element.label for element in self.elements])
        return element_labels


    def get_element(self, element_label):
        '''
        Get `Element` object from element label.

        Arguments
        -----------
        element_label : int
            label of element to get
        
        Returns
        -------------
        element : obj
            Element object corresponding to input label

        '''
        if element_label in self.elements:
            ix = np.where(self.element_labels().astype(int)==int(element_label))[0][0]
            return self.elements[ix]
        else:
            return None
              
        
    def node_label_to_node_ix(self, node_label):
        '''
        Get node index from node label.

        Arguments
        -------------
        node_label : int
            node label to query
        
        Returns
        --------------
        node_ix : int
            node index as ordered in `ElDef`
        
        '''
        node_ix = np.where(self.get_node_labels()==node_label)
        if len(node_ix[0])>0:
            node_ix = node_ix[0][0].astype(int)
        else:
            raise ValueError(f'Invalid node requested. Node {node_label} not present.')

        return node_ix
    
    
    def node_label_to_dof_ix(self, node_label): 
        '''
        Get dof indices from node label.

        Arguments
        -------------
        node_label : int
            node label to query
        
        Returns
        --------------
        dof_ix : int
            array of global dof indices corresponding to node order in ElDef
        
        '''
        node_ix = self.node_label_to_node_ix(node_label)   
        lower_dofs = sum(self.all_ndofs[:node_ix])
        dof_ix = lower_dofs + np.arange(0, self.all_ndofs[node_ix])
        return dof_ix

    # MODIFIERS
    def update_sections(self, sections, element_labels=None, reassemble=True):
        '''
        Update section specifications for elements.

        Arguments
        -------------
        sections : Section obj
            list of `Section` objects (or single object to apply to all element labels), matching the order of element_labels
        element_labels : int
            list of element labels to apply sections to (if None, all are assumed)
        reasseble : True
            whether or not to reassemble `ElDef` after update (new sections cause new stiffness, mass, etc.)       
        
        '''

        if element_labels is None:  # all elements, assumed single section object in sections
            element_labels = self.element_labels()
            sections = [sections]*len(element_labels)

        for ix, el in enumerate(element_labels):
            element = self.get_element(el)
            element.section = sections[ix]

        if reassemble:
            self.assemble()

    def update_geometry_to_deformed(self):
        '''
        Propagate deformed state of `ElDef` to undeformed state (redefine geometry).
        '''
        for node in self.nodes:
            node.x0 = node.x*1  # make deformed structure new reference
    
    def deform(self, u, du=None, update_tangents=True):
        '''
        Deform `ElDef` specified u.

        Arguments
        ----------
        u : float
            array of displacements corresponding to global stacking of nodes and dofs
        update_tangents : True
            whether or not to update the tangent stiffness resulting from the deformation (relevant in corotational formulation)

        '''
        for node in self.nodes:
            if du is None:
                node.du = u[node.global_dofs] - node.u
            else:
                node.du = du[node.global_dofs]

            node.u = u[node.global_dofs]
            node.x = node.x0 + node.u
            node.increment_rotation_tensor()

        for element in self.elements:
            element.update()

        if update_tangents:
            self.update_tangent_stiffness()

        self.update_internal_forces()

    def deform_linear(self, u):
        '''
        Deform `ElDef` specified u linearly, i.e., without updating geometry or any internal forces.

        Arguments
        ----------
        u : float
            array of displacements corresponding to global stacking of nodes and dofs

        '''
        for node in self.nodes:
            node.du = u[node.global_dofs] - node.u
            node.u = u[node.global_dofs]
            node.x = node.x0 + node.u

        for element in self.elements:
            element.update_linear()    

        self.update_internal_forces(u)      # on part level (element internal forces are dealt with intristicly by update function above)
    

    def update_all_geometry(self):
        '''
        Update all geometry parameters (length, normal vector, 
        transformation matrix) based on specified deformation in current object.
        '''
        for element in self.elements:
            element.update_geometry()

    def update_internal_forces(self, u=None):       # on part level
        '''
        Update internal forces vector q in element definition from internal 
        forces in all elements.
        '''

        if u is None:
            u = np.zeros([self.ndofs])
        
        self.q = self.get_feature_mats(mats=['k']) @ u 

        for el in self.elements:
            self.q[el.global_dofs] += el.q

    def update_tangent_stiffness(self):
        '''
        Update tangent stiffness based on current state.
        '''
        self.k = self.get_feature_mats(mats=['k'])         
        
        for el in self.elements:
            self.k[np.ix_(el.global_dofs, el.global_dofs)] += el.k

    
    def update_mass_matrix(self):
        '''
        Update mass matrix based on current state.
        '''
        self.m = self.get_feature_mats(mats=['m'])  

        for el in self.elements:
            self.m[np.ix_(el.global_dofs, el.global_dofs)] += el.m
            

    # GET METHODS
    def get_sections(self):
        '''
        Get list of unique `Section` objects present in ElDef.
        '''
        sections = []
        
        for el in self.elements:
            if el.section not in sections: sections.append(el.section)
        
        return sections
    
    def get_tmats(self):
        '''
        Get list of tmat of all elements (in order of elements).
        '''
        return [el.tmat for el in self.elements]
        
    
    # CONSTRAINT METHODS   
    def constraint_dof_ix(self):   
        '''
        Create numpy array with constraint dof indices.

        Returns
        -----------
        c_dof_ix : int
            Numpy array with constraint dof pairs - each constraint is given a row.

        '''
        if self.constraints is None or len(self.constraints)==0:
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
        '''
        Output number of constrained dofs.

        TODO: needs refinement.

        Returns
        ----------
        count : int
            number of constrained dofs present
        '''
        count = 0
        for constraint in self.constraints:
            n_dofs = [len(nc.dof_ix) for nc in constraint.node_constraints]
            count = count + np.sum(n_dofs)
        
        return count
    
    
    def compatibility_matrix(self):
        '''
        Establish compatibility matrix from constraints.

        Returns
        --------
        compat_mat : float
            numpy array with compatibility matrix describing the constraints present
        '''
        dof_pairs = self.constraint_dof_ix()
        compat_mat = compmat(dof_pairs, self.ndofs)
        
        return compat_mat   
    
    # TRANSFORMATION METHODS
    def local_node_csys(self, node_label, average_ok=True):
        '''
        Establish CSYS (transformation matrix) of node.

        Arguments
        -----------
        node_label : int
            label of node
        average_ok : True
            whether or not averaging transformation matrices at node is okay or not - if multiple elements are connected to node,
            the T might not be unique and averaging can be conducted to overcome

        Returns
        -----------
        t_mat : float
            numpy 2d array with transformation matrix corresponding to node
        '''

        elements, node_ix = self.elements_with_node(node_label, merge_parts=True)
        
        t_mats = [el.get_tmat(reps=1) for el in elements]
        
        if (len(elements)>1) and not average_ok:
            raise ValueError('More elements connected to node. Use average_ok=True if averaging of transformation mats is fine.')
        
        t_mat = blkdiag(np.mean(np.stack(t_mats, axis=2), axis=2), 2)
        
        return t_mat

    def get_kg(self, N=None):
        '''
        Establish linearized geometric stiffness from axial forces.

        Arguments
        ------------
        N : None, float
            value of axial force applied; when defined as standard value None, `Node.N0` is used (`Node.N` is used if that is not specified).


        Returns
        --------------
        kg_eldef : float
            2d numpy array with geometric stiffness matrix (global FE format correpsonding to specified order of nodes)
        '''
        kg_eldef = np.zeros([self.ndofs, self.ndofs])
        
        for el in self.elements:
            if el.nodes[1].global_dofs is None:
                print(el.nodes[1].global_dofs)  # temporarily added for quick debugging purposes
                print(el.nodes[1])
                
            glob_dofs = np.r_[el.nodes[0].global_dofs, el.nodes[1].global_dofs].astype(int)
            local_dofs = np.r_[0:el.nodes[0].ndofs, (el.nodes[0].ndofs):(el.nodes[0].ndofs + el.nodes[1].ndofs )]    #added for cases where one node in element is not present in self.nodes, check speed effect later

            kg_eldef[np.ix_(glob_dofs, glob_dofs)] += el.get_kg(N=N)[np.ix_(local_dofs, local_dofs)]
            
            if np.any(np.isnan(el.get_kg()[np.ix_(local_dofs, local_dofs)])):
                print(el)   # temporarily added for quick debugging purposes
                        
        return kg_eldef


    # GENERATE OUTPUT FOR ANALYSIS    
    def global_element_matrices(self, constraint_type=None):   
        '''
        Get global mass, damping, stiffness and geometric stiffness matrices.

        Arguments
        -----------
        constraint_type : {None, 'lagrange', 'primal'}
            what constraint type to enforce

        Returns
        -----------
        mass : float
            global mass matrix
        damping : float
            global damping matrix (excluding rayleigh damping, which is considered part of analysis)
        stiffness : float
            global stiffness matrix
        geometric_stiffness : float
            global linearized stiffness matrix if N0 is specified on element level

        '''

        # Starting point is feature matrices
        stiffness, damping, mass = self.get_feature_mats()       
        geometric_stiffness = np.zeros([self.ndim, self.ndim])
        
        for el in self.elements:
            dof_ix1, dof_ix2 = el.nodes[0].global_dofs, el.nodes[1].global_dofs
            dof_range = np.r_[dof_ix1, dof_ix2]
            T = el.tmat

            mass[np.ix_(dof_range, dof_range)] += el.get_m()
            stiffness[np.ix_(dof_range, dof_range)] += el.get_k()
            geometric_stiffness[np.ix_(dof_range, dof_range)] += el.get_kg_axial()  #if N0 specified in 

        if self.constraints != None:  
            if constraint_type == 'lagrange':
                dof_pairs = self.constraint_dof_ix()
                mass = lagrange_constrain(mass, dof_pairs)
                damping = lagrange_constrain(damping, dof_pairs)
                stiffness = lagrange_constrain(stiffness, dof_pairs)
                geometric_stiffness = lagrange_constrain(geometric_stiffness, dof_pairs)
            
            elif constraint_type == 'primal':
                stiffness = self.L.T @ stiffness @ self.L
                damping = self.L.T @ damping @ self.L
                mass = self.L.T @ mass @ self.L
                geometric_stiffness = self.L.T @ geometric_stiffness @ self.L
                
        return mass, damping, stiffness, geometric_stiffness
    
    
    def get_state_matrix(self):
        '''
        Establish state space matrix based on global element matrices.

        Returns
        -------------
        A : float
            state matrix A
        '''
        m,c,k,kg = self.global_element_matrices(constraint_type='primal')
        return statespace(k+kg, c, m)

    # MISC METHODS
    def get_max_dim(self, max_of_max=True):
        '''
        Get the largest distance between two nodes in `ElDef`.

        Arguments
        -------------
        max_of_max : True
            whether or not to output the max of the maximum (single number)
        
        Returns
        -------------
        max_dim : float
            largest distance
        '''
        n,__ = self.export_eldef()
        coors = n[:,1:]
        
        max_dim = np.ptp(coors, axis=0)
        
        if max_of_max:
            max_dim = np.max(max_dim)
            
        return max_dim
    
    
    def elements_with_node(self, node_label, return_node_ix=True):
        '''
        Get elements that contain nodes with given node label.

        Arguments
        ------------
        node_label : int
            node label to find
        return_node_ix : True, optional
            whether or not to return a second variable with the indices of the nodes

        Returns
        -----------
        elements
        node_ix
            only returned if `return_node_ix` is True

        '''
        elements = [el for el in self.elements if node_label in el.nodes]  
        node_ix = [np.where(el.nodes==node_label)[0] for el in elements]
            
        if return_node_ix:
            return elements, node_ix
        else:
            return elements     
        
        
    def export_matrices(self, part_ix=None):
        '''
        Establish matrices describing nodes and elements in `ElDef`.

        Arguments
        -----------
        part_ix : int, optional
            index of part to export - if standard value None is used, all parts are included

        Returns
        ------------
        node_matrix : float
            node definition where each row represents a node as [node_label_i, x_i, y_i, z_i]
        element_matrix : int
            element definition where each row represents an element as [element_label_i, node_label_1, node_label_2]

        '''
        
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
    '''
    Assembly definition main class.

    Arguments
    ---------------------------
    parts : Part obj
        list of `Part` objects to assemble in `Assembly`
    constraints : Constraint obj
        list of `Constraint` objects to enforce
    features : Feature obj
        list of `Feature` objects (springs, dashpots, point masses)
    include_linear_kg : boolean
        whether or not to include linearized geometric stiffness from specified axial force (N0 in element objects)
    constraint_type : {'none', 'lagrange', or 'primal'}
        constraint to enforce
    domain : {'3d', '2d'}
    assemble : boolean, True
        whether or not to assemble structure automatically upon generation - useful to use False if no computation is used and invertable matrices is not strictly required

    '''


    def __init__(self, parts, **kwargs):
        domains = [part.domain for part in parts]
        if not all([domain==domains[0] for domain in domains]):
            raise ValueError('Cannot combine parts in different domains. Use either 2d or 3d for all parts!')

        super().__init__(self.all_nodes(parts), self.all_elements(parts), **kwargs)
        self.parts = parts        

    @staticmethod
    def all_nodes(parts, sort_by_label=True):
        '''
        Returns all nodes in all parts entered.

        Arguments
        ----------
        parts : obj
            list of `Part` objects to query

        Returns 
        ---------
        nodes : obj
            list of `Node` objects in parts
        '''
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
    
    @staticmethod
    def all_elements(parts):
        '''
        Returns all elements in all parts entered.

        Arguments
        ----------
        parts : obj
            list of `Part` objects to query

        Returns 
        ---------
        elements : obj
            list of `Element` objects in parts
        '''

        return [sublist for l in [part.elements for part in parts] for sublist in l]       
            
    
class Part(ElDef):
    '''
    Part definition main class. Mainly used as alternative constructor, basically identical to `ElDef`.

    Arguments
    ---------------------------
    node_matrix : float
        node definition where each row represents a node as [node_label_i, x_i, y_i, z_i]
    element_matrix : int
        element definition where each row represents an element as [element_label_i, node_label_1, node_label_2]
    sections : Section obj
        list of `Section` objects to match the order of the elements (rows) in `element_matrix`.
    element_types : {'beam', 'bar'}
        type of elements for each element (row) in `element_matrix`
    left_handed_csys : False
        whether or not to enforce left_handed_csys - right handed is standard
    constraints : obj
        list of `Constraint` objects
    features : obj
        list of `Feature` objects (springs, dashpots, point masses)
    include_linear_kg : boolean
        whether or not to include linearized geometric stiffness from specified axial force (N0 in element objects)
    constraint_type : {'none', 'lagrange', or 'primal'}
        constraint to enforce
    domain : {'3d', '2d'}
    assemble : boolean, True
        whether or not to assemble structure automatically upon generation - useful to use False if no computation is used and invertable matrices is not strictly required

    '''

    def __init__(self, node_matrix, element_matrix, sections=None, element_types=None, left_handed_csys=False, **kwargs):      
        if element_types == None:
            element_types = ['beam']*element_matrix.shape[0] # assume that all are beam

        nodes, elements = create_nodes_and_elements(node_matrix, element_matrix, sections=sections, left_handed_csys=left_handed_csys, element_types=element_types)
        if node_matrix.shape[1] == 3:
            domain = '2d'
        elif node_matrix.shape[1] == 4:
            domain = '3d'

        super().__init__(nodes, elements, domain=domain, **kwargs)


def create_nodes(node_matrix):
    '''
    Create node list from specified matrix.

    Arguments
    --------------
    node_matrix : float
        node definition where each row represents a node as [n_label_i, x_i, y_i, z_i]

    Returns
    -----------
    nodes : Node obj
        list of `Node` objects
    '''    

    n_nodes = node_matrix.shape[0]
    nodes = [None]*n_nodes

    for node_ix in range(0, n_nodes):
        nodes[node_ix] = Node(node_matrix[node_ix, 0], node_matrix[node_ix, 1:])
    
    return nodes

def create_nodes_and_elements(node_matrix, element_matrix, sections=None, left_handed_csys=False, element_types=None):
    '''
    Create node list and element list from specified matrices.

    Arguments
    --------------
    node_matrix : float
        node definition where each row represents a node as [n_label_i, x_i, y_i, z_i]
    element_matrix : int
        element definition where each row represents an element as [element_label_i, node_label_1, node_label_2]
    sections : Section obj
        list of `Section` objects to match the order of the elements (rows) in `element_matrix`.
    left_handed_csys : False
        whether or not to enforce left_handed_csys - right handed is standard
    element_types : {'beam', 'bar'}
        type of elements for each element (row) in `element_matrix`


    Returns
    -----------
    nodes : Node obj
        list of `Node` objects
    elements : Element obj
        list of `Element` objects

    '''    
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
                elements[el_ix] = BeamElement3d([nodes[ix1], nodes[ix2]], 
                                            label=int(element_matrix[el_ix, 0]), 
                                            section=sections[el_ix], 
                                            left_handed_csys=left_handed_csys)
            else:
                elements[el_ix] = BeamElement2d([nodes[ix1], nodes[ix2]], 
                                                label=int(element_matrix[el_ix, 0]), 
                                                section=sections[el_ix])

        elif element_types[el_ix] == 'bar':
            # Currently only added to enable extraction of Kg for bars - not supported otherwise (assumed as beams otherwise)
            elements[el_ix] = BarElement3d([nodes[ix1], nodes[ix2]], label=int(element_matrix[el_ix, 0]), section=sections[el_ix], left_handed_csys=left_handed_csys)
        
    return nodes, elements