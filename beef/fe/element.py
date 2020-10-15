from ..fe.section import Section
import numpy as np
from ..general import transform_unit, blkdiag

class BeamElement:
    # CORE FUNCTIONS    
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
        return (self.nodes[0].coordinates + self.nodes[1].coordinates)/2

    def get_vec(self, undeformed=False):
        if undeformed:
            return self.nodes[1].coordinates - self.nodes[0].coordinates      
        else:
            return self.nodes[1].x[:self.dim] - self.nodes[0].x[:self.dim]
    
    def get_e(self):
        return self.get_vec()/self.get_length()

    def get_length(self):
        return np.sqrt(np.sum(self.get_vec()**2))

    def get_psi(self, return_phi=True):
        if (not hasattr(self, 'force_psi') or self.force_psi is None) and self.section.shear_deformation:
            denom = self.section.kappa*self.section.G*self.section.A*self.get_length()**2
            phi = 12*self.section.E*self.section.I/denom
        else:
            phi = self.section.I*0
        
        psi = 1/(1+phi)
        
        if return_phi:
            return psi, phi
        else:
            return psi

    # ------------- FE UPDATING --------------------------
    def update_m(self):
        self.m = self.tmat.T @ self.get_local_m() @ self.tmat

    def update_geometry(self):
        self.L = self.get_length()
        self.e = self.get_e()
        self.tmat = self.get_tmat()   
        self.psi = self.get_psi(return_phi=False)

    def update_k(self):
        self.k = self.tmat.T @ self.get_local_k() @ self.tmat

    # --------------- MISC --------------------------------
    def get_nodelabels(self):
        return [node.label for node in self.nodes]


class BeamElement2d(BeamElement):
    def __init__(self, nodes, label, section=Section(), shear_flexible=False, mass_formulation='timoshenko', nonlinear=True, N0=0):
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
        
        self.initiate_nodes()
        self.L0 = self.get_length()
        self.phi0 = self.get_element_angle()
        self.update_geometry()
        self.update()        

    # ------------- INITIALIZATION ----------------------
    def initiate_nodes(self):
        for node in self.nodes:
            node.ndofs = self.dofs_per_node 
            node.x0 = np.zeros(6)
            node.x0[:2] = node.coordinates
            node.x = node.x0*1
            node.u = np.zeros(6)

    # ------------- GEOMETRY -----------------------------
    def get_e2(self):
        return np.array([-self.e[1], self.e[0]])

    def get_element_angle(self):
        x_a = self.nodes[0].x
        x_b = self.nodes[1].x       

        dx = (x_b - x_a)[0:2]
        el_ang = np.arctan2(dx[1],dx[0])
        
        return el_ang

    def get_tmat(self):
        T = np.eye(6)
        T[0, :2] = self.e
        T[1, :2] = self.e2

        T[3, 3:5] = T[0, :2]
        T[4, 3:5] = T[1, :2]

        return T

    # ------------- FE CORE -----------------------------
    def local_k_nonlinear(self):
        k_local = np.zeros([6,6])
        section = self.section

        k_local[:3, :3] = (1/self.L**3) * np.array([[section.E*section.A*self.L**2,-self.Q*self.L**2, 0],
                                  [-self.Q*self.L**2, 12*self.psi*section.E*section.I+6/5*self.N*self.L**2, 6*self.psi*section.E*section.I*self.L+1/10*self.N*self.L**3],
                                  [0, 6*self.psi*section.E*section.I*self.L+1/10*self.N*self.L**3, (3*self.psi+1)*section.E*section.I*self.L**2+2/15*self.N*self.L**4]])

        k_local[3:, 3:] = (1/self.L**3) * np.array([[section.E*section.A*self.L**2,-self.Q*self.L**2,0],
                                  [-self.Q*self.L**2, 12*self.psi*section.E*section.I+6/5*self.N*self.L**2, -6*self.psi*section.E*section.I*self.L-1/10*self.N*self.L**3],
                                  [0, -6*self.psi*section.E*section.I*self.L-1/10*self.N*self.L**3, (3*self.psi+1)*section.E*section.I*self.L**2+2/15*self.N*self.L**4]])
        
        k_local[:3, 3:] = (1/self.L**3) * np.array([[-section.E*section.A*self.L**2,self.Q*self.L**2,0],
                                  [self.Q*self.L**2, -12*self.psi*section.E*section.I-6/5*self.N*self.L**2, 6*self.psi*section.E*section.I*self.L+1/10*self.N*self.L**3],
                                  [0, -6*self.psi*section.E*section.I*self.L-1/10*self.N*self.L**3, (3*self.psi-1)*section.E*section.I*self.L**2-1/30*self.N*self.L**4]])
        
        k_local[3:, :3] = k_local[0:3,3:].T
        
        return k_local

    def local_k_linear(self):
        # Original version
        k_local = np.zeros([6,6])
        section = self.section

        k_local[:3, :3] = (1/self.L0**3) * np.array([[section.E*section.A*self.L0**2,0, 0],
                                  [0, 12*self.psi*section.E*section.I, 6*self.psi*section.E*section.I*self.L0],
                                  [0, 6*self.psi*section.E*section.I*self.L0, (3*self.psi+1)*section.E*section.I*self.L0**2]])

        k_local[3:, 3:] = (1/self.L0**3) * np.array([[section.E*section.A*self.L0**2,0,0],
                                  [0, 12*self.psi*section.E*section.I, -6*self.psi*section.E*section.I*self.L0],
                                  [0, -6*self.psi*section.E*section.I*self.L0, (3*self.psi+1)*section.E*section.I*self.L0**2]])
        
        k_local[:3, 3:] = (1/self.L0**3) * np.array([[-section.E*section.A*self.L0**2,0,0],
                                  [0, -12*self.psi*section.E*section.I, 6*self.psi*section.E*section.I*self.L0],
                                  [0, -6*self.psi*section.E*section.I*self.L0, (3*self.psi-1)*section.E*section.I*self.L0**2]])
        
        k_local[3:, :3] = k_local[0:3,3:].T
        
        return k_local

    def local_m_lumped(self):
        m = self.section.m
        L = self.L0
        m_lumped = np.diag([m*L/2, m*L/2, m*L**2/4, m*L/2, m*L/2, m*L**2/4])
        
        return m_lumped
    
    def local_m_euler_trans(self):
        m_et = self.local_m_euler()
        m_et[np.ix_([2,5],[2,5])] = self.local_m_lumped()[np.ix_([2,5],[2,5])]
        
        return m_et
    
    def local_m_euler(self):
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

    def local_m_timo(self):
        rho = self.section.m/self.section.A
        I = self.section.I
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

    def get_local_kg(self, N):
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
        self.q_loc = self.local_k_linear() @ self.tmat @ np.hstack([self.nodes[0].u, self.nodes[1].u])

        self.N = (self.q_loc[3] - self.q_loc[0])/2             # update internal force N from t
        self.M = (self.q_loc[5] - self.q_loc[2])/2
        self.Q = (self.q_loc[4] - self.q_loc[1])/2

        self.t = np.array([self.N, self.M, -self.Q*self.L0/2])
        self.q = self.tmat.T @ self.get_S() @ self.t  # calculate internal forces in global format

    def update_nonlinear(self):
        self.update_geometry()              # update all node positions and element geometry     
        self.update_corot()                 # --> new internal forces (corotational)
        self.update_k()                     # --> new tangent stiffness (consider mass as well? need to adjust density.)

    def update_v(self):
        el_angle = self.get_element_angle()

        self.v[0] = self.L - self.L0
        self.v[1] = self.nodes[1].x[2] - self.nodes[0].x[2]

        phi_a = self.nodes[0].x[2] + self.nodes[1].x[2] - 2*(el_angle - self.phi0)  #asymmetric bending
        self.v[2] = ((phi_a + np.pi) % (2*np.pi)) - np.pi # % is the modulus operator, this ensures 0<phi_a<2pi

    def update_corot(self, linear=False):
        self.update_v()        # compute displacement mode
        self.t = self.get_Kd_c() @ self.v              # new internal forces (element forces) based on the two above  

        self.N = self.t[0]                  # update internal force N from t
        self.M = self.t[1]
        self.Q = -2*self.t[2]/self.L        # update internal force Q from t   
        
        self.q = self.tmat.T @ self.get_S() @ self.t  # calculate internal forces in global format
    
    def get_S(self):
        return np.array([[-1,0,0], 
                         [0,0,2/self.L], 
                         [0,-1,1], 
                         [1,0,0], 
                         [0,0,-2/self.L], 
                         [0, 1, 1]])    

    def get_Kd_c(self):
        section = self.section
        Kd_c = 1/self.L * np.array([
            [section.E*section.A, 0, 0], 
            [0, section.E*section.I, 0],
            [0, 0, 3*self.psi*section.E*section.I]])
        
        return Kd_c

    def get_Kd_g(self):
        return self.L*self.N*np.array([[0,0,0], [0,1/12,0], [0,0, 1/20]])

    # --------------- POST PROCESSING ------------------------------
    def extract_load_effect(self, load_effect):
        if load_effect == 'M':
            return (self.q[5] - self.q[2])/2
        elif load_effect == 'V':
            return (self.q[4] - self.q[1])/2
        elif load_effect == 'N':
            return self.N

    # --------------- POST PROCESSING ------------------------------
    def get_kg(self, N=None, nonlinear=True):  # element level function (global DOFs)
        if N is None:
            N = self.N

        if nonlinear:
            return self.tmat.T @ self.get_S() @ self.get_Kd_g() @ self.get_S().T @ self.tmat #from corotated formulation
        else:
            return self.tmat.T @ self.get_local_kg(N) @ self.tmat

class BeamElement3d(BeamElement):
    def __init__(self, nodes, label=None, section=Section(), mass_formulation='consistent', shear_flexible=False, nonlinear=False, e2=None, N0=0, left_handed_csys=False):
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
        self.update()        

    # ------------- INITIALIZATION ----------------------
    def initiate_nodes(self):
        for node in self.nodes:
            node.ndofs = self.dofs_per_node 
            node.x0 = np.zeros(6)
            node.x0[:3] = node.coordinates
            node.x = node.x0*1
            node.u = np.zeros(6)

    # ------------- GEOMETRY -----------------------------
    def get_tmat_rhs(self):
        T0 = transform_unit(self.get_e(), self.get_e2())
        return blkdiag(T0, 4)
    
    def get_tmat_lhs(self):
        T0 = transform_unit(self.get_e(), self.get_e2())
        T_r2l = np.array([[1,0,0,0,0,0], 
                          [0,0,1,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,0,-1,0,0],
                          [0,0,0,0,0,-1],
                          [0,0,0,0,-1,0]])
        
        return blkdiag(T_r2l, 2) @ blkdiag(T0, 4)
        
    
    
    def get_e2(self):
        if self.e2 is None:
            smallest_ix = np.argmin(abs(self.e))
            return np.eye(3)[smallest_ix, :]
        else:
            return self.e2
    
    # ------------- FE CORE -------------------------------
    def local_k_linear(self):
        # EN234: Three-dimentional Timoshenko beam element undergoing axial, torsional and bending deformations
        # Fang, 2015 (Wengqiang_Fan.pdf)

        A = self.section.A
        G = self.section.G
        L = self.L
        E = self.section.E
        I_y, I_z = self.section.I

        J = self.section.J       
        k_axial = E*A/L

        mu_y, mu_z, phi_y, phi_z = self.get_psi(return_phi=True)
        
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
        I_z = self.section.I[1]
        I_y = self.section.I[0]

        mu_y, mu_z, phi_y, phi_z = self.get_psi(return_phi=True)

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
        # https://link.springer.com/content/pdf/10.1007%2F978-1-84996-190-5_1.pdf    

        I_y, I_z = self.section.I
        mu_y, mu_z, phi_y, phi_z = self.get_psi(return_phi=True)

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
    
    
    
    def get_kg(self, N=None):  # element level function (global DOFs)
        return self.tmat.T @ self.get_local_kg(N=N) @ self.tmat

    def get_m(self):
        return self.tmat.T @ self.get_local_m() @ self.tmat

    # --------------- FE UPDATING ---------------------------------
    def update_linear(self):
        pass

    # --------------- POST PROCESSING ------------------------------
    def extract_load_effect(self, load_effect):
        if load_effect == 'M':
            return (self.q[5] - self.q[2])/2
        elif load_effect == 'V':
            return (self.q[4] - self.q[1])/2
        elif load_effect == 'N':
            return self.N