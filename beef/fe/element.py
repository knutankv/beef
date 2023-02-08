'''
FE objects submodule: elements
'''

from ..fe.section import Section
import numpy as np
from ..general import transform_unit, blkdiag
from copy import deepcopy

class BeamElement:
    '''
    Beam element core class. Basic functionality (common for all children objects)
    will inherit these methods.
    '''

    # ------------- CORE FUNCTIONS -------------------------------------   
    def __eq__(self, other):
        if isinstance(other, BeamElement):
            return self.label == other.label
        elif isinstance(other, int):
            return self.label == other

    def __lt__(self, other):
        if isinstance(other, BeamElement):
            return self.label < other.label
        elif isinstance(other, int):
            return self.label < other

    def __repr__(self):
        return f'Element {self.label}'

    def __str__(self):
        return f'Element {self.label}'


    # ------------- GEOMETRY AND PROPERTIES ---------------------------
    def get_cog(self):
        '''
        Get center of gravity coordinates of element.

        Returns
        ---------
        cog : float
            numpy array indicating the center of gravity of the element (2 or 3
            components based on dimensionality of system) 
        '''
        return (self.nodes[0].coordinates + self.nodes[1].coordinates)/2

    def get_vec(self, undeformed=False):
        '''
        Get vector of element (length and direction).

        Arguments
        ------------
        undeformed : False
            whether the vector should represent the element 
            in its deformed or undeformed state

        Returns
        -------------
        vec : float
            vector with x,y (and z if 3d) components of element
        '''
        if undeformed:
            return self.nodes[1].coordinates - self.nodes[0].coordinates      
        else:
            return self.nodes[1].x[:self.dim] - self.nodes[0].x[:self.dim]
    
    def get_e(self):
        '''
        Get unit vector of element (direction with length 1)

        Returns
        ------------
        e : float
            vector with unit length describing the direction of the element 
            (from node 1 to node 2), with x, y (and z if 3d) components

        '''
        return self.get_vec()/self.get_length()

    def get_length(self):
        '''
        Get length of element.
        '''

        return np.sqrt(np.sum(self.get_vec()**2))

    def get_psi(self, return_phi=True):
        '''
        Calculates shear flexibility parameter.

        Arguments
        ------------
        return_phi : True
            whether or not to return phi () as a second return

        Returns
        -----------
        psi : float
            shear flexibility parameter (see Notes below)
        phi : float
            only returned if requested (return_phi = True)

        Notes
        -----------
        Calculates the following:
             $$\psi = \\frac{1}{1+\phi}, \quad \phi = \\frac{12 EI}{LGA}$$.
        
        References
        ------------
        [[1]](../#1) Krenk, 2009.
        '''

        if self.domain=='2d':
            I = self.section.I[0]
        else:
            I = self.section.I

        if (not hasattr(self, 'force_psi') or self.force_psi is None) and self.section.shear_deformation:
            denom = self.section.kappa*self.section.G*self.section.A*self.get_length()**2
            phi = 12*self.section.E*I/denom
        else:
            phi = I*0
        
        psi = 1/(1+phi)
        
        if return_phi:
            return psi, phi
        else:
            return psi
    # ------------- ELEMENT MATRIX -----------------------
    def get_kg(self, N=None):  # element level function (global DOFs) 
        '''
        Get global linearized geometric stiffness matrix of element.

        Arguments
        -------------
        N : float
            axial force used to establish stiffness (if standard value
            None is used, the actual N0 (first priority) or N (second priority)
            of the element is assumed)
        '''

        return self.tmat.T @ self.get_local_kg(N=N) @ self.tmat

    def get_m(self):
        '''
        Get global mass matrix of element.
        '''

        return self.tmat.T @ self.get_local_m() @ self.tmat
    
    def get_k(self):
        '''
        Get global stiffness matrix of element.
        '''

        return self.tmat.T @ self.get_local_k() @ self.tmat

    # ------------- FE UPDATING --------------------------
    def update_m(self):
        '''
        Update global mass matrix of element based on local current mass matrix.
        '''
        self.m = self.tmat.T @ self.get_local_m() @ self.tmat

    def update_k(self):
        '''
        Update global stiffness matrix of element based on local current mass matrix.
        '''
        self.k = self.tmat.T @ self.get_local_k() @ self.tmat

    def update_geometry(self):
        '''
        Run all update geometry methods of element.
        '''
        self.L = self.get_length()
        self.e = self.get_e()
        self.tmat = self.get_tmat()   
        self.psi = self.get_psi(return_phi=False)



    # ---------------- NODE-BASED PROPERTIES --------------
    @property
    def nodelabels(self):
        return [node.label for node in self.nodes]

    @property
    def global_dofs(self):
        return np.hstack([node.global_dofs for node in self.nodes])

    @property
    def u(self):
        return np.hstack([node.u for node in self.nodes])
        
    @property
    def ndofs(self):
        return self.nodes[0].ndofs + self.nodes[1].ndofs

    def subdivide(self, n):
        '''
        Divide element into n elements.

        Arguments
        -----------
        n : int
            number of divisions/resulting elements
            
        Returns
        -------------
        elements : obj
            list of new element objects
            
        '''
        
        elements = [None]*n
        x0 = self.nodes[0].coordinates
        x1 = self.nodes[1].coordinates
        v = x1-x0
        
        for el in range(n):
            elements[el] = deepcopy(self)
            elements[el].nodes[0].coordinates = x0+v*1/n*el
            elements[el].nodes[1].coordinates = x0+v*1/n*(el+1)
            
            elements[el].initiate_nodes()
        
        return elements
    

class BeamElement2d(BeamElement):
    '''
    Two-dimensional beam element class.

    Arguments
    -----------
    nodes : Node obj
        list of two node objects used to create element
    label : int
        integer label of element
    section : Section obj, optional
        section object to define element (standard value is standard initialized Section object)
    shear_flexible : False, optional
        whether or not to include shear flexibility
    mass_formulation : {'euler', 'timoshenko', 'lumped'}
        what mass formulation to apply (refers to shape functions used)
    nonlinear : True, optional
        whether or not to use nonlinear internal functions for element
    N0 : float, optional
        applied axial force (for linearized geometric stiffness calculation)
    '''
    def __init__(self, nodes, label, section=Section(), shear_flexible=False, 
                 mass_formulation='euler', nonlinear=True, N0=None):
        self.nodes = nodes
        self.label = int(label)
        self.section = section
        
        self.shear_flexible = shear_flexible
        self.nonlinear = nonlinear

        self.dofs_per_node = 3  
        self.dim = 2   
        self.domain = '2d'  

        self.v = np.zeros(3)
        self.N0 = N0

        # Assign mass matrix function
        if mass_formulation not in ['timoshenko', 'euler', 'lumped', 'euler_trans']:
            raise ValueError("{timoshenko', 'euler', 'lumped', 'euler_trans'} are allowed values for mass_formulation. Please correct input.")
        elif mass_formulation == 'timoshenko':
            self.get_local_m = self.local_m_timo
        elif mass_formulation == 'euler':
            self.get_local_m = self.local_m_euler
        elif mass_formulation  == 'euler_trans':
            self.get_local_m = self.local_m_euler_trans
        elif mass_formulation == 'lumped':
            self.get_local_m = self.local_m_lumped

        # Assign update functions
        if nonlinear:
            self.update = self.update_nonlinear
            self.get_local_k = self.local_k_nonlinear
        else:
            self.update = self.update_linear
            self.get_local_k = self.local_k_linear
        
        # Initialization function runs
        self.initiate_nodes()
        self.L0 = self.get_length()
        self.phi0 = self.get_element_angle()
        self.update_geometry()
        self.update_m()
        self.update()        

    # ------------- INITIALIZATION ----------------------
    def initiate_nodes(self):
        '''
        Initiate nodes of element.
        '''
        for node in self.nodes:
            node.ndofs = self.dofs_per_node 
            node.x0 = np.zeros(3)
            node.x0[:2] = node.coordinates
            node.x = node.x0*1
            node.u = np.zeros(3)

    # ------------- GEOMETRY -----------------------------
    def get_e2(self):
        '''
        Get normal vector of element. 
        '''

        return np.array([-self.e[1], self.e[0]])

    def get_element_angle(self):
        '''
        Get angle of element.

        Returns
        ----------
        el_ang : float
            angle of element in radians
        '''
        x_a = self.nodes[0].x
        x_b = self.nodes[1].x       

        dx = (x_b - x_a)[0:2]
        el_ang = np.arctan2(dx[1],dx[0])
        
        return el_ang

    def get_tmat(self):
        '''
        Get transformation matrix of element.

        Returns
        -----------
        T : float
            transformation matrix of element
        '''
        T = np.eye(6)
        T[0, :2] = self.e
        T[1, :2] = self.get_e2()

        T[3, 3:5] = T[0, :2]
        T[4, 3:5] = T[1, :2]

        return T

    # ------------- FE CORE -----------------------------
    def local_k_nonlinear(self):
        '''
        Get local nonlinear stiffness matrix.

        Returns
        -----------
        k_local : float
            2d numpy array describing local stiffness matrix of 2d beam (6x6)

        Notes
        -----------
        This method executes the summation of Eqs. 5.45--5.47 and Eqs. 5.48--5.50 in [[1]](../#1).
        The full element stiffness is established as follows:

        $$[k_{el}] = \\begin{bmatrix}
            [k_{11}] & [k_{12}]\\\\
            [k_{21}] & [k_{22}]
        \\end{bmatrix} $$
        
        where the submatrices are defined as:

        $$ [k_{11}] = \\frac{1}{L^3} 
        \\begin{bmatrix}
            EAL^2 & 0 & 0\\\\
            0 & 12 \psi EI & 6\psi EIL\\\\
            0 & 6 \psi EIL & (3\psi+1)EIL^2
        \\end{bmatrix} +
        \\frac{1}{L}
        \\begin{bmatrix}
            0 & -Q & 0\\\\
            -Q & \\frac{6}{5}N & \\frac{1}{10}NL\\\\
            0 & 6 \\frac{1}{10}NL & \\frac{2}{15}NL^2
        \\end{bmatrix}        
        $$

        $$ [k_{22}] = \\frac{1}{L^3} 
        \\begin{bmatrix}
            EAL^2 & 0 & 0\\\\
            0 & 12 \psi EI & -6\psi EIL\\\\
            0 & -6\psi EIL & (3\psi+1)EIL^2
        \\end{bmatrix} +
        \\frac{1}{L}
        \\begin{bmatrix}
            0 & -Q & 0\\\\
            -Q & \\frac{6}{5}N & -\\frac{1}{10}NL\\\\
            0 & 6 -\\frac{1}{10}NL & \\frac{2}{15}NL^2
        \\end{bmatrix}        
        $$

        $$ [k_{12}] = [k_{21}]^T = \\frac{1}{L^3} 
        \\begin{bmatrix}
            -EAL^2 & 0 & 0\\\\
            0 & -12 \psi EI & 6\psi EIL\\\\
            0 & -6 \psi EIL & (3\psi-1)EIL^2
        \\end{bmatrix} +
        \\frac{1}{L}
        \\begin{bmatrix}
            0 & Q & 0\\\\
            Q & -\\frac{6}{5}N & \\frac{1}{10}NL\\\\
            0 & 6 -\\frac{1}{10}NL & -\\frac{2}{15}NL^2
        \\end{bmatrix}        
        $$

        References
        ------------
        [[1]](../#1) Krenk, 2009.

        '''
        k_local = np.zeros([6,6])
        section = self.section


        k_local[:3, :3] = (1/self.L**3) * np.array([[section.E*section.A*self.L**2,-self.Q*self.L**2, 0],
                                  [-self.Q*self.L**2, 12*self.psi*section.E*section.I[0]+6/5*self.N*self.L**2, 6*self.psi*section.E*section.I[0]*self.L+1/10*self.N*self.L**3],
                                  [0, 6*self.psi*section.E*section.I[0]*self.L+1/10*self.N*self.L**3, (3*self.psi+1)*section.E*section.I[0]*self.L**2+2/15*self.N*self.L**4]])

        k_local[3:, 3:] = (1/self.L**3) * np.array([[section.E*section.A*self.L**2,-self.Q*self.L**2,0],
                                  [-self.Q*self.L**2, 12*self.psi*section.E*section.I[0]+6/5*self.N*self.L**2, -6*self.psi*section.E*section.I[0]*self.L-1/10*self.N*self.L**3],
                                  [0, -6*self.psi*section.E*section.I[0]*self.L-1/10*self.N*self.L**3, (3*self.psi+1)*section.E*section.I[0]*self.L**2+2/15*self.N*self.L**4]])
        
        k_local[:3, 3:] = (1/self.L**3) * np.array([[-section.E*section.A*self.L**2,self.Q*self.L**2,0],
                                  [self.Q*self.L**2, -12*self.psi*section.E*section.I[0]-6/5*self.N*self.L**2, 6*self.psi*section.E*section.I[0]*self.L+1/10*self.N*self.L**3],
                                  [0, -6*self.psi*section.E*section.I[0]*self.L-1/10*self.N*self.L**3, (3*self.psi-1)*section.E*section.I[0]*self.L**2-1/30*self.N*self.L**4]])
        
        k_local[3:, :3] = k_local[0:3,3:].T
        
        return k_local

    def local_k_linear(self):
        '''
        Get local linear stiffness matrix.

        Returns
        -----------
        k_local : float
            2d numpy array describing local stiffness matrix of 2d beam (6x6)

        Notes
        -----------
        The full element stiffness is established as follows:

        $$[k_{el}] = \\begin{bmatrix}
            [k_{11}] & [k_{12}]\\\\
            [k_{21}] & [k_{22}]
        \\end{bmatrix} $$
        
        where the submatrices are defined as:

        $$ [k_{11}] = \\frac{1}{L^3} 
        \\begin{bmatrix}
            EAL^2 & 0 & 0\\\\
            0 & 12 \psi EI & 6\psi EIL\\\\
            0 & 6 \psi EIL & (3\psi+1)EIL^2
        \\end{bmatrix}      
        $$

        $$ [k_{22}] = \\frac{1}{L^3} 
        \\begin{bmatrix}
            EAL^2 & 0 & 0\\\\
            0 & 12 \psi EI & -6\psi EIL\\\\
            0 & -6\psi EIL & (3\psi+1)EIL^2
        \\end{bmatrix}     
        $$

        $$ [k_{12}] = [k_{21}]^T = \\frac{1}{L^3} 
        \\begin{bmatrix}
            -EAL^2 & 0 & 0\\\\
            0 & -12 \psi EI & 6\psi EIL\\\\
            0 & -6 \psi EIL & (3\psi-1)EIL^2
        \\end{bmatrix}      
        $$


        '''
        k_local = np.zeros([6,6])
        section = self.section

        k_local[:3, :3] = (1/self.L0**3) * np.array([[section.E*section.A*self.L0**2,0, 0],
                                  [0, 12*self.psi*section.E*section.I[0], 6*self.psi*section.E*section.I[0]*self.L0],
                                  [0, 6*self.psi*section.E*section.I[0]*self.L0, (3*self.psi+1)*section.E*section.I[0]*self.L0**2]])

        k_local[3:, 3:] = (1/self.L0**3) * np.array([[section.E*section.A*self.L0**2,0,0],
                                  [0, 12*self.psi*section.E*section.I[0], -6*self.psi*section.E*section.I[0]*self.L0],
                                  [0, -6*self.psi*section.E*section.I[0]*self.L0, (3*self.psi+1)*section.E*section.I[0]*self.L0**2]])
        
        k_local[:3, 3:] = (1/self.L0**3) * np.array([[-section.E*section.A*self.L0**2,0,0],
                                  [0, -12*self.psi*section.E*section.I[0], 6*self.psi*section.E*section.I[0]*self.L0],
                                  [0, -6*self.psi*section.E*section.I[0]*self.L0, (3*self.psi-1)*section.E*section.I[0]*self.L0**2]])
        
        k_local[3:, :3] = k_local[0:3,3:].T
        
        return k_local

    def local_m_lumped(self):
        '''
        Local mass matrix of element based on lumped formulation.

        Returns
        ---------
        m_lumped : float
            local mass matrix, 6x6 numpy array 

        Notes 
        ---------
        The lumped mass matrix is established as follows:

        $$
        [m_{el}] = \\text{diag}
        \\begin{bmatrix}
        mL/2 & mL/2 & mL^2/4 & mL/2 & mL/2 & mL^2/4
        \\end{bmatrix}
        $$
        '''
        m = self.section.m
        L = self.L0
        m_lumped = np.diag([m*L/2, m*L/2, m*L**2/4, m*L/2, m*L/2, m*L**2/4])
        
        return m_lumped

    
    def local_m_euler(self):
        '''
        Local mass matrix of element based on consistent formulation (Euler stiffness).

        Returns
        ---------
        m : float
            local mass matrix, 6x6 numpy array 

        Notes 
        ---------
        The consistent (Euler) mass matrix is established as follows:

        $$
        [m_{el}] = \\frac{mL}{420} \\begin{bmatrix}
                               140 &          0 &          0 &          70 &         0 &          0        \\\\
                               0 &          156 &        22L &         0 &         54 &       -13L    \\\\
                               0 &          22L &         4L^2 &        0 &         13L &       -3L^2   \\\\          
                               70 &          0 &          0 &          140 &   0 &          0        \\\\
                               0 &          54 &       13L &       0 &         156 &    -22L      \\\\
                               0 &         -13L &       -3L^2 &      0 &        -22L &     4L^2                    
        \\end{bmatrix}
        $$
        '''
        m = self.section.m
        L = self.L0
        
        return m*L/420 * np.array([
                               [140,          0,          0,          70,         0,          0        ],
                               [0,          156,        22*L,         0,         54,       -13*L    ],
                               [0,          22*L,         4*L**2,        0,         13*L,       -3*L**2    ],          
                               [70,          0,          0,          140,   0,          0        ],
                               [0,          54,       13*L,       0,         156,    -22*L      ],
                               [0,         -13*L,       -3*L**2,      0,        -22*L,     4*L**2   ]
                               ])

    
    def local_m_euler_trans(self):
        '''
        Local mass matrix of element based on combination of
        consistent formulation (Euler stiffness) and lumped formulation.

        Returns
        ---------
        m : float
            local mass matrix, 6x6 numpy array 


        Notes
        ----------
        Rotational terms are established from lumped formulation (`BeamElement2d.local_m_lumped`),
        whereas translational terms are from the consistent 
        Euler formulation (`BeamElement2d.local_m_euler`)
        '''
        m_et = self.local_m_euler()
        m_et[np.ix_([2,5],[2,5])] = self.local_m_lumped()[np.ix_([2,5],[2,5])]
        
        return m_et


    def local_m_timo(self):
        '''
        Local mass matrix of element based on consistent formulation (Timoshenko stiffness).

        Returns
        ---------
        m : float
            local mass matrix, 6x6 numpy array 

        Notes 
        ---------
        *Experimental implementation - needs verification!*

        The consistent (Timoshenko) mass matrix is established as follows:

        $$
        [m_r] = \\frac{\\rho I}{(1+\psi^2)L} \\begin{bmatrix}
                        \\frac{6}{5} & (\\frac{1}{10}-\\frac{1}{2}\psi)L &-\\frac{6}{5} &(\\frac{1}{10}-\\frac{1}{2}\psi)L \\\\
                         \\dots& (\\frac{12}{15}+\\frac{1}{6}\psi+\\frac{1}{3}\psi^2)L^2 &(-\\frac{1}{10}+\\frac{1}{2}\psi)L &-(\\frac{1}{3}0+\\frac{1}{6}\psi-\\frac{1}{6}\psi^2)L^2 \\\\
                         \\dots& \\dots &\\frac{6}{5} &(-\\frac{1}{10}+\\frac{1}{2}\psi)L \\\\
                         sym.& \\dots& \\dots&(2/15+\\frac{1}{6}\psi+\\frac{1}{3}\psi^2)L^2                         
        \\end{bmatrix}
        $$

        $$
        [m_t] = \\frac{\\rho I}{1+\psi^2} \\begin{bmatrix}
                        \\frac{13}{35}+\\frac{7}{10}\psi+\\frac{1}{3}\psi^2 & (\\frac{11}{210}+\\frac{11}{210}\psi+\\frac{1}{24}\psi^2)L & \\frac{9}{70}+\\frac{3}{10}\psi+\\frac{1}{6}\psi^2 & -(\\frac{13}{420}+\\frac{3}{40}\psi+\\frac{1}{24}\psi^2)L \\\\
                        \\dots  & (\\frac{1}{105}+\\frac{1}{6}\psi+\\frac{1}{20}\psi^2)L^2 & (\\frac{13}{420}+\\frac{3}{40}\psi+\\frac{1}{24}\psi^2)L & -(\\frac{1}{140}+\\frac{1}{6}\psi+\\frac{1}{120}\psi^2)L^2 \\\\
                         \\dots &   \\dots& \\frac{13}{35}+\\frac{7}{10}\psi+\\frac{1}{3}\psi^2 & (\\frac{11}{210}+\\frac{11}{120}\psi+\\frac{1}{24}\psi^2)L \\\\
                        sym. & \\dots & \\dots &(\\frac{1}{105}+\\frac{1}{6}\psi+\\frac{1}{120}\psi^2)L^2                        
        \\end{bmatrix}
        $$
        '''
        rho = self.section.m/self.section.A
        I = self.section.I[0]
        L = self.L0
        A = self.section.A
        psi = self.psi
        
        m_t = rho*A*L/(1+psi)**2 * np.array([[13/35+7/10*psi+1/3*psi**2, (11/210+11/210*psi+1/24*psi**2)*L, 9/70+3/10*psi+1/6*psi**2, -(13/420+3/40*psi+1/24*psi**2)*L],
                                             [0, (1/105+1/60*psi+1/20*psi**2)*L**2, (13/420+3/40*psi+1/24*psi**2)*L, -(1/140+1/60*psi+1/120*psi**2)*L**2],
                                             [0, 0, 13/35+7/10*psi+1/3*psi**2, (11/210+11/120*psi+1/24*psi**2)*L],
                                             [0,0,0,(1/105+1/60*psi+1/120*psi**2)*L**2]])
        m_t = m_t + m_t.T - np.diag(np.diag(m_t))   # copy values above diagonal to below diagonal
        
        m_r = rho*I/((1+psi**2)*L) * np.array([[6/5, (1/10-1/2*psi)*L,-6/5,(1/10-1/2*psi)*L],
                                               [0, (12/15+1/6*psi+1/3*psi**2)*L**2,(-1/10+1/2*psi)*L,-(1/30+1/6*psi-1/6*psi**2)*L**2],
                                               [0,0,6/5,(-1/10+1/2*psi)*L],
                                               [0,0,0,(2/15+1/6*psi+1/3*psi**2)*L**2]])
        
        m_r = m_r + m_r.T - np.diag(np.diag(m_r))   # copy values above diagonal to below diagonal
        
        m = np.zeros([6,6])
        m[np.ix_([1,2,4,5],[1,2,4,5])] = m_r + m_t
        m[np.ix_([0,3],[0,3])] = np.array([[2, 1], [1, 2]])*rho*A*L/6
        
        return m

    def get_local_kg(self, N=None):
        '''
        Get local linearized geometric stiffness of element due to axial force N.

        Arguments
        ------------
        N : float
            axial force to apply to element (standard value None enforces the use of self.N0 (priority 1) or self.N (priority 2)
            of the current object)

        Returns
        -----------
        kg : float
            6x6 numpy array describing the local geometric stiffness

        Notes
        -----------
        The linearized geometric stiffness is calculated as follows:
        $$
        [k_g] = 
        \\frac{N}{30L} \\begin{bmatrix}
                    0 & 0 & 0 & 0 & 0 & 0 \\\\
                    0 & 36 & 3L & 0 & -36 & 3L \\\\
                    0 & 3L & 4L^2 & 0 & -3L & -L^2 \\\\
                    0 & 0 & 0 & 0 & 0 & 0 \\\\
                    0 & -36 & -3L & 0 & 36 & -3L \\\\
                    0 & 3L & -L^2 & 0 & -3L & 4L^2 \\\\
        \end{bmatrix}
        $$
        '''
        if N is None:
            if self.N0 is None:
                N = self.N
            else:
                N = self.N0
            

        L = self.L0
        return np.array([
                    [0, 0, 0, 0, 0, 0],
                    [0, 36, 3*L, 0, -36, 3*L],
                    [0, 3*L, 4*L**2, 0, -3*L, -L**2],
                    [0, 0, 0, 0, 0, 0],
                    [0, -36, -3*L, 0, 36, -3*L],
                    [0, 3*L, -L**2, 0, -3*L, 4*L**2]
                ]) * N/(30*L)     

    # -------------- UPDATE METHODS ---------------------
    def update_linear(self):
        '''
        Update element forces from linear stiffness assumption.
        '''
        self.q_loc = self.local_k_linear() @ self.tmat @ np.hstack([self.nodes[0].u, self.nodes[1].u])

        self.N = (self.q_loc[3] - self.q_loc[0])/2             # update internal force N from t
        self.M = (self.q_loc[5] - self.q_loc[2])/2
        self.Q = (self.q_loc[4] - self.q_loc[1])/2

        self.t = np.array([self.N, self.M, -self.Q*self.L0/2])
        self.q = self.tmat.T @ self.get_S() @ self.t  # calculate internal forces in global format

    def update_nonlinear(self):
        '''
        Update element forces from nonlinear stiffness assumption.
        '''
        self.update_geometry()              # update all node positions and element geometry     
        self.update_corot()                 # --> new internal forces (corotational)
        self.update_k()                     # --> new tangent stiffness (consider mass as well? need to adjust density.)

    def update_v(self):
        '''
        Update deformation modes of element.

        Notes 
        ----------
        {v} describes the deformation modes of the element, which are given by:
            
        1. Elongation of element
        2. Angle of symmetric deformation mode
        3. Angle of the anti-symmetric deformation mode

        For more information, refer to Chapter 5 of [[1]](../#1).

        '''
        el_angle = self.get_element_angle()

        self.v[0] = self.L - self.L0
        self.v[1] = self.nodes[1].x[2] - self.nodes[0].x[2]

        phi_a = self.nodes[0].x[2] + self.nodes[1].x[2] - 2*(el_angle - self.phi0)  #asymmetric bending
        self.v[2] = ((phi_a + np.pi) % (2*np.pi)) - np.pi # % is the modulus operator, this ensures 0<phi_a<2pi

    def update_corot(self, linear=False):
        '''
        Update all corotational parameters of element based on current state.

        Conducts the following steps:

        * Update the deformation modes \(\{v\}\): `BeamElement2d.update_v`
        * Set internal forces from deformation modes \(\{t\} = [K_d] \{v\}\)
        * Assign values for moment M, axial force N and shear force V from internal force \(\{v\})\)
        **TODO: consider simply define as dynamic properties**
        * Calculate internal forces in global format with \(\{q\} = [T]^T [S] [T]\)
        
        '''
        self.update_v()        # compute displacement mode
        self.t = self.get_Kd_c() @ self.v              # new internal forces (element forces) based on the two above  

        self.N = self.t[0]                  # update internal force N from t
        self.M = self.t[1]
        self.Q = -2*self.t[2]/self.L        # update internal force Q from t   
        
        self.q = self.tmat.T @ self.get_S() @ self.t  # calculate internal forces in global format
    
    def get_S(self):
        '''
        Get matrix transforming from reduced (deformation modes) to full format.

        Returns
        ---------
        S : float
            6x3 numpy array describing S

        Notes
        ---------
        \([S]\) is described in Eq. 5.13 of [[1]](../#1)
        '''
        return np.array([[-1,0,0], 
                         [0,0,2/self.L], 
                         [0,-1,1], 
                         [1,0,0], 
                         [0,0,-2/self.L], 
                         [0, 1, 1]])    

    def get_Kd_c(self):
        '''
        Get constitutive part of stiffness matrix for deformation modes.

        Returns
        --------
        Kd_c : float
            3x3 numpy array describing constitutive part of stiffness matrix for the deformation modes

        Notes
        ---------
        See Eq. 5.33 in [[1]](../#1).

        '''
        section = self.section
        Kd_c = 1/self.L * np.array([
            [section.E*section.A, 0, 0], 
            [0, section.E*section.I[0], 0],
            [0, 0, 3*self.psi*section.E*section.I[0]]])
        
        return Kd_c

    def get_Kd_g(self):
        '''
        Get geometric part of stiffness matrix for deformation modes.

        Returns
        -----------
        k_dg : float
            3x3 numpy array (matrix) describing the geometric stiffness related to the three deformation modes given by v

        Notes
        -----------
        \(k_{d,g}\) is given in Eq. 5.42 in [[1]](../#1).
        '''
        return self.L*self.N*np.array([[0,0,0], [0,1/12,0], [0,0, 1/20]])

    # --------------- POST PROCESSING ------------------------------
    def extract_load_effect(self, load_effect):
        '''
        Postprocessing method to extract bending moment, shear force or axial force at given deformation state.

        Arguments
        ----------
        load_effect : {'M', 'V', 'N'}
            load effect to extract

        Returns
        ----------
        val : float
            float number describing the queried load effect
        '''

        if load_effect == 'M':
            return (self.q[5] - self.q[2])/2
        elif load_effect == 'V':
            return (self.q[4] - self.q[1])/2
        elif load_effect == 'N':
            return self.N

    # --------------- MISC ------------------------------
    def get_kg_nonlinear(self):  # element level function (global DOFs)
        '''
        Extract geometric stiffness from corotational formulation.

        Notes
        ---------
        The stiffness is established as follows:
        $$
        [k_g] = [T]^T ([S] [K_{d,g}] [S]^T) [T]
        $$
        '''

        return self.tmat.T @ self.get_S() @ self.get_Kd_g() @ self.get_S().T @ self.tmat #from corotated formulation

class BeamElement3d(BeamElement):
    '''
    Three-dimensional beam element class.

    Arguments
    -------------
    nodes : Node obj
        list of Node objects
    label : int, optional
        integer label of element
    section : Section obj
        section describing element
    mass_formulation : {'consistent', 'lumped'}
        selector for mass formulation
    shear_flexible : False
        whether or not to include shear flexibility in establishment of element matrices
    nonlinear : False
        whether or not to use nonlinear formulation (corotational)
    e2 : float, optional
        3x1 numpy array describing second perpendicular vector (if not given, automatically generated)
    N0 : 0, optional    
        axial force to use for computation of linearized geometric stiffness of element;
        if not given N from current state is assumed
    left_handed_csys : False
        whether or not to create transformation matrices such that the results are expressed
        in a left-handed csys (*experimental*)
    '''
    def __init__(self, nodes, label=None, section=Section(), mass_formulation='consistent', 
                 shear_flexible=False, nonlinear=False, e2=None, N0=0, left_handed_csys=False):
        self.nodes = nodes
        self.label = label
        self.section = section
        self.shear_flexible = shear_flexible
        self.nonlinear = nonlinear
        
        self.dim = 3
        self.dofs_per_node = 6     
        self.domain = '3d'  

        self.e2 = e2
        self.N0 = N0
        self.left_handed_csys = left_handed_csys
        
        if left_handed_csys:
            self.get_tmat = self.get_tmat_lhs
        else:
            self.get_tmat = self.get_tmat_rhs            

        if mass_formulation not in ['lumped', 'consistent']:
            raise ValueError("{'lumped', 'consistent'} are allowed values for mass_formulation. Please correct input.")
        elif mass_formulation == 'lumped':
            self.get_local_m = self.local_m_lumped
        elif mass_formulation == 'consistent':
            self.get_local_m = self.local_m_consistent

        if nonlinear:
            raise ValueError('Only linear 3d elements are currently supported.')
        else:
            self.get_local_k = self.local_k_linear
            self.update = self.update_linear

        self.initiate_nodes()
        self.L0 = self.get_length()   
        self.update_geometry()
        self.update_m()
        self.update()        

    # ------------- INITIALIZATION ----------------------
    def initiate_nodes(self):
        '''
        Initiate nodes of element.
        '''
        for node in self.nodes:
            node.ndofs = self.dofs_per_node 
            node.x0 = np.zeros(6)
            node.x0[:3] = node.coordinates
            node.x = node.x0*1
            node.u = np.zeros(6)

    # ------------- GEOMETRY -----------------------------
    def get_tmat_rhs(self, reps=4):
        '''
        Get transformation matrix of element assuming a right-handed csys.

        Arguments
        ----------
        reps : 4
            number of repititions of core transformation matrix (used to transform a 3d vector)

        Returns
        -----------
        T : float
            transformation matrix of element
        '''

        T0 = transform_unit(self.get_e(), self.get_e2())
        return blkdiag(T0, reps)
    
    def get_tmat_lhs(self):
        '''
        Get transformation matrix of element assuming a left-handed csys. This is only
        relevant if `BeamElement3d.left_handed_csys` is defined as `True`.

        Arguments
        ----------
        reps : 4
            number of repititions of core transformation matrix (used to transform a 3d vector)
            
        Returns
        -----------
        T : float
            transformation matrix of element
        '''

        T0 = transform_unit(self.get_e(), self.get_e2())
        T_r2l = np.array([[1,0,0,0,0,0], 
                          [0,0,1,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,0,-1,0,0],
                          [0,0,0,0,0,-1],
                          [0,0,0,0,-1,0]])
        
        return blkdiag(T_r2l, 2) @ blkdiag(T0, 4)
        

    def get_e2(self):
        '''
        Get second perpendicular element vector. Uses specified e2 if available, and otherwise constructs it. 
        '''
        if self.e2 is None:
            smallest_ix = np.argmin(abs(self.e))
            return np.eye(3)[smallest_ix, :]
        else:
            return self.e2
    
    # ------------- FE CORE -------------------------------
    def local_k_linear(self):
        '''
        Get local linear stiffness matrix.

        Returns
        ----------
        ke : float
            12x12 numpy array characterizing the local element stiffness 

        Notes
        ----------
        The local stiffness matrix is computed as follows:

        $$
            \\begin{bmatrix}
                EA/L &  &  &  &  &  &  &  &  &  &  &  \\\\
                0 & 12\\mu_zEI_z/L^3 &  &  &  &  &  &  &  &  &  &  \\\\
                0 & 0 & 12\\mu_yEI_y/L^3 &  &  &  &  &  & sym. &  &  & \\\\
                0 & 0 & 0 & GJ/L &  &  & &  & &  & & \\\\
                0 & 0 & -6\\mu_yEI_y/L^2 & 0 & (4+\phi_y)\\mu_yEI_y/L & & & & & & & \\\\
                0 & 6\\mu_zEI_z/L^2 & 0 & 0 & 0 & (4+\phi_z)\\mu_zEI_z/L & & & & & & \\\\
                -EA/L & 0 & 0 & 0 & 0 & 0 & EA/L & & & & & \\\\
                0 & -12\\mu_zEI_z/L^3 & 0 & 0 & 0 & -6\\mu_zEI_z/L^2 & 0 & 12\\mu_zEI_z/L^3 & & & & \\\\
                0 & 0 & -12\\mu_yEI_y/L^3 & 0 & 6\\mu_yEI_y/L^2 & 0 & 0 & 0 & 12\\mu_yEI_y/L^3 & & & \\\\
                0 & 0 & 0 & -GJ/L & 0 & 0 & 0 & 0 & 0 & GJ/L & & \\\\
                0 & 0 & -6\\mu_yEI_y/L^2 & 0 & (2-\phi_y)\\mu_yEI_y/L & 0 & 0 & 0 & 6\\mu_yEI_y/L^2 & 0 & (4+\phi_y)\\mu_yEI_y/L &\\\\
                0 & 6\\mu_zEI_z/L^2 & 0 & 0 & 0 & (2-\phi_z)\\mu_zEI_z/L & 0 & -6\\mu_zEI_z/L^2 & 0 & 0 & 0 & (4+\phi_z)\\mu_zEI_z/L
            \\end{bmatrix}
        $$
        
        References 
        ----------
        [[2]](../#2) W. Fang, EN234: Three-dimentional Timoshenko beam element undergoing axial, torsional and bending deformations, 2015
        '''

        A = self.section.A
        G = self.section.G
        L = self.L
        E = self.section.E
        I_y, I_z = self.section.I

        J = self.section.J       
        k_axial = E*A/L

        mu, phi = self.get_psi(return_phi=True)
        
        mu_y, mu_z = mu 
        phi_y, phi_z = phi
        
        ke = np.array([
            [k_axial, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 12*mu_z*E*I_z/L**3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 12*mu_y*E*I_y/L**3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, G*J/L, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -6*mu_y*E*I_y/L**2, 0, (4+phi_y)*mu_y*E*I_y/L, 0, 0, 0, 0, 0, 0, 0],
            [0, 6*mu_z*E*I_z/L**2, 0, 0, 0, (4+phi_z)*mu_z*E*I_z/L, 0, 0, 0, 0, 0, 0],
            [-k_axial, 0, 0, 0, 0, 0, k_axial, 0, 0, 0, 0, 0],
            [0, -12*mu_z*E*I_z/L**3, 0, 0, 0, -6*mu_z*E*I_z/L**2, 0, 12*mu_z*E*I_z/L**3, 0, 0, 0, 0],
            [0, 0, -12*mu_y*E*I_y/L**3, 0, 6*mu_y*E*I_y/L**2, 0, 0, 0, 12*mu_y*E*I_y/L**3, 0, 0, 0],
            [0, 0, 0, -G*J/L, 0, 0, 0, 0, 0, G*J/L, 0, 0],
            [0, 0, -6*mu_y*E*I_y/L**2, 0, (2-phi_y)*mu_y*E*I_y/L, 0, 0, 0, 6*mu_y*E*I_y/L**2, 0, (4+phi_y)*mu_y*E*I_y/L, 0],
            [0, 6*mu_z*E*I_z/L**2, 0, 0, 0, (2-phi_z)*mu_z*E*I_z/L, 0, -6*mu_z*E*I_z/L**2, 0, 0, 0, (4+phi_z)*mu_z*E*I_z/L]
        ])
        
        ke = ke + ke.T - np.diag(np.diag(ke))   #copy symmetric parts (& avoid doubling diagonal)
        return ke        
   
    def local_m_lumped(self):
        '''
        Local mass matrix of element based on lumped formulation.

        Returns
        ---------
        me : float
            local mass matrix, 12x12 numpy array 

        Notes 
        ---------
        The lumped mass matrix is established as follows:

        $$
        [m_{el}] = mL \cdot \\text{diag}  
        \\begin{bmatrix}
         0.5 \\\\
        0.5 \\\\
        0.5 \\\\
        Ip/(2A) \\\\
        (L^2\mu_z^2*(1+I_y(42+210\\phi_z^2))/(AL^2))/420 \\\\
        (L^2\mu_y^2*(1+I_z(42+210\\phi_y^2))/(AL^2))/420 \\\\
        0.5 \\\\
        0.5 \\\\
        0.5 \\\\
        Ip/(2*A) \\\\
        (L^2\\mu_z^2(1+I_y(42+210\phi_z^2))/(AL^2))/420 \\\\
        (L^2\\mu_y^2(1+I_z(42+210\phi_y^2))/(AL^2))/420
        \\end{bmatrix}
        $$

        where
        $$\phi_i = \\frac{12 EI_i}{LGA}$$.

        '''
        I_z = self.section.I[1]
        I_y = self.section.I[0]

        mu, phi = self.get_psi(return_phi=True)
        
        mu_y, mu_z = mu 
        phi_y, phi_z = phi

        m = self.section.m
        L = self.L
        
        A = self.section.A
        Ip = self.section.J
        me = m*L * np.diag([0.5,
                             0.5,
                             0.5,
                             Ip/(2*A),
                             (L**2*mu_z**2*(1+I_y*(42+210*phi_z**2))/(A*L**2))/420,
                             (L**2*mu_y**2*(1+I_z*(42+210*phi_y**2))/(A*L**2))/420,
                             0.5,
                             0.5,
                             0.5,
                             Ip/(2*A),
                             (L**2*mu_z**2*(1+I_y*(42+210*phi_z**2))/(A*L**2))/420,
                             (L**2*mu_y**2*(1+I_z*(42+210*phi_y**2))/(A*L**2))/420,
                             ])
            
        me = me + me.T - np.diag(np.diag(me)) #copy symmetric parts (& avoid doubling diagonal)
    
        return me

        
    def local_m_consistent(self):
        '''
        Local mass matrix of element based on consistent formulation (Euler stiffness).

        Returns
        ---------
        m : float
            local mass matrix, 12x12 numpy array 

        Notes 
        ---------
        See Kardeniz et al. [[3]](../#3) for details.
        '''
          
        I_y, I_z = self.section.I

        mu, phi = self.get_psi(return_phi=True)
        
        mu_y, mu_z = mu 
        phi_y, phi_z = phi
        
        m = self.section.m
        L = self.L
        
        A = self.section.A
        Ip = self.section.J
        m22 = mu_y**2 * (13/35 + 7/10*phi_y + phi_y**2/3+6/5*I_z/(A*L**2))
        m26 = mu_y**2*L * (11/210 + 11/120 * phi_y + phi_y**2/24 + I_z/(A*L**2)*(1/10-3/2*phi_y-phi_y**2))
        m28 = mu_y**2 * (9/70 + 3/10*phi_y + phi_y**2/6-6/5*I_z/(A*L**2))
        m212 = mu_y**2*L*(13/420 + 3/40*phi_y+phi_y**2/24-I_z/(A*L**2)*(1/10-phi_y/2))
        m33 = mu_z**2*(13/35 + 7/10*phi_z+phi_z**2/3+6/5*I_y/(A*L**2))
        m35 = mu_z**2*L*(11/210 + 11/120*phi_z+phi_z**2/24+I_y/(A*L**2)*(1/10-3/2*phi_z-phi_z**2))
        m39 = mu_z**2*(11/210+11/120*phi_z+phi_z**2/24+I_y/(A*L**2)*(1/10-3/2*phi_z-phi_z**2))
        m311 = mu_z**2*L*(13/420+3/40*phi_z+phi_z**2/24-I_y/(A*L)*(1/10-phi_z/2))
        m44 = Ip/(3*A)
        m410 = Ip/(6*A)
        
        m55 = mu_z**2*L**2*(1/105 + phi_z/60 + phi_z**2/120 + I_y/(A*L**2)*(2/15+phi_z/6+phi_z**2/3))
        m59 = mu_z**2*L*(13/420 + 3/40*phi_z+phi_z**2/24-I_y/(A*L**2)*(1/10-3/2*phi_z-phi_z**2))
        m511 = mu_z**2*L**2*(1/140+phi_z/60 + phi_z**2/120 + I_y/(A*L**2)*(1/30+phi_z/6-phi_z**2/6))
        
        m66 = mu_y**2*L**2*(1/105+phi_y/60+phi_y**2/120+I_z/(A*L**2)*(2/15+phi_y/6+phi_y**2/3))
        m68 = mu_y**2*L*(13/420+3/40*phi_y+phi_y**2/24-I_z/(A*L**2)*(1/10-3/2*phi_y-phi_y**2))
                
        m612 = mu_y**2*L**2*(1/140+1/60*phi_y+1/120*phi_y**2+I_z/(A*L**2)*(1/30+phi_y/6-phi_y**2/6))
        m812 = mu_y**2*L*(11/210+11/120*phi_y+1/24*phi_y**2+I_z/(A*L**2)*(1/10-phi_y/2))
        m911 = mu_z**2*L*(11/210+11/120*phi_z+phi_z**2/24+I_y/(A*L**2)*(1/10-phi_z/2))

        m_11 = np.array([[1/3, 0 ,0],[0, m22, 0],[0, 0, m33]])
        m_12 = np.array([[0, 0 ,0],[0, 0, m26],[0, -m35, 0]])
        
        m_13 = np.array([[1/6, 0 ,0],[0, m28, 0],[0, 0, m39]])
        m_14 = np.array([[0, 0 ,0],[0, 0, -m212],[0, m311, 0]])

        m_22 = np.array([[m44, 0 ,0],[0, m55, 0],[0, 0, m66]])
        m_23 = np.array([[0, 0 ,0],[0, 0, -m59],[0, m68, 0]])

        m_24 = np.array([[m410, 0 ,0],[0, -m511, 0],[0, 0, -m612]])
        m_34 = np.array([[0, 0 ,0],[0, 0, -m812],[0, m911, 0]])
        O = m_11*0
        
        me = m*L*np.vstack([
                np.hstack([m_11, m_12, m_13, m_14]),
                np.hstack([O, m_22, m_23, m_24]),
                np.hstack([O, O, m_11, m_34]),
                np.hstack([O, O, O, m_22])
                ]
            )
        
        me = me + me.T - np.diag(np.diag(me)) #copy symmetric parts (& avoid doubling diagonal)
        return me
    

    def get_local_kg(self, N=None):
        '''
        Get local linearized geometric stiffness of element due to axial force N.

        Arguments
        ------------
        N : float
            axial force to apply to element (standard value None enforces the use of self.N0 (priority 1) or self.N (priority 2)
            of the current object)

        Returns
        -----------
        kg : float
            12x12 numpy array describing the local geometric stiffness

        '''

        if N is None:
            N = self.N0
        
        L = self.L
    
        if self.section.shear_deformation:
            print('Timoshenko formulation (shear deformation) not implemented for geometric stiffness. Using Euler-Bernoulli.')

        return np.array([
                [0,         0,          0,          0,          0,              0,              0,              0,              0,              0,          0,              0],
                [0,         6/5,        0,          0,          0,              L/10,           0,              -6/5,           0,              0,          0,               L/10],
                [0,         0,          6/5,        0,          -L/10,          0,              0,              0,              -6/5,           0,          -L/10,          0],
                [0,         0,          0,          0,          0,              0,              0,              0,              0,              0,          0,              0],
                [0,         0,          -L/10,      0,          2*L**2/15,      0,              0,              0,              L/10,           0,          -L**2/30,       0],
                [0,         L/10,       0,          0,          0,              2*L**2/15,      0,              -L/10,          0,              0,          0,              -L**2/30],
                [0,         0,          0,          0,          0,              0,              0,              0,              0,              0,          0,              0],
                [0,         -6/5,       0,          0,          0,              -L/10,          0,              6/5,            0,              0,          0,              -L/10],
                [0,         0,          -6/5,       0,          L/10,           0,              0,              0,              6/5,            0,          L/10,           0],
                [0,         0,          0,          0,          0,              0,              0,              0,              0,              0,          0,              0],
                [0,         0,          -L/10,      0,          -L**2/30,       0,              0,              0,              L/10,           0,          2*L**2/15,      0],
                [0,         L/10,       0,          0,          0,              -L**2/30,       0,              -L/10,          0,              0,          0,              2*L**2/15],
            ]) * N/L
    
    


    # --------------- FE UPDATING ---------------------------------
    def update_linear(self):
        '''
        Update element forces from linear stiffness assumption.
        '''
        self.update_k()
        self.update_m()
        self.q = np.zeros(12)

    # --------------- POST PROCESSING ------------------------------
    def extract_load_effect(self, load_effect):
        '''
        Postprocessing method to extract bending moment, shear force or axial force at given deformation state.

        Arguments
        ----------
        load_effect : {'M', 'V', 'N'}
            load effect to extract

        Returns
        ----------
        val : float
            float number describing the queried load effect
        '''

        if load_effect == 'M':
            return (self.q[5] - self.q[2])/2
        elif load_effect == 'V':
            return (self.q[4] - self.q[1])/2
        elif load_effect == 'N':
            return self.N

    

class BarElement3d(BeamElement3d):
    '''
    Three-dimensional bar element class, inheriting methods from the three-dimensional beam element class.
    '''

    def get_local_kg(self, N=None):
        '''
        Get local linearized geometric stiffness of bar element due to axial force N (assuming three translational
        and three rotational DOFs on each node).

        Arguments
        ------------
        N : float
            axial force to apply to element (standard value None enforces the use of self.N0 (priority 1) or self.N (priority 2)
            of the current object)

        Returns
        -----------
        kg : float
            12x12 numpy array describing the local geometric stiffness
        '''

        if N is None:
            N = self.N0
        
        L = self.L
        return np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])*N/L