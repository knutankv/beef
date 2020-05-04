from ._section import *
from ._base import *
from . import *
import numpy as np

class Element:
    def __init__(self, nodes, label=None, section=Section(), dofs_per_node=None):
        self.section = section
        self.nodes = nodes
        self.label = int(label)
        self.dofs_per_node = dofs_per_node        

    def tmat(self, reps=None):
        if reps is None:
            reps = int(self.dofs_per_node/3 * len(self.nodes))
            
        e1 = self.vec()/self.length()
        
        if self.section.e2 is None:
            smallest_ix = np.argmax(abs(e1))
            e2p = np.eye(3)[smallest_ix, :]
        else:
            e2p = self.section.e2
        
        element_tmat = transform_unit(e1, e2p)
        element_tmat = blkdiag(element_tmat, reps)

        return element_tmat

    def cog(self):
        return (self.nodes[0].coordinates + self.nodes[1].coordinates)/2

    def vec(self):
        return self.nodes[1].coordinates - self.nodes[0].coordinates
    
    def e(self):
        return self.vec()/self.length()

    def length(self):
        return np.sqrt(np.sum(self.vec()**2))
    
    def node_labels(self):
        return  [node.label for node in self.nodes]

    def get_local_matrix(self, matrix_type='K'):
        if matrix_type.lower() == 'k':
            matrix = self.stiffness()
        elif matrix_type.lower() == 'm':
            matrix = self.mass()
        elif matrix_type.lower() == 'kg':            
            matrix = self.geometric_stiffness()
        else:

            raise ValueError('Missing or non-matching matrix type defined. Matrix type defined as %s. Supported: "m", "k", "kg".' % matrix_type.lower())

        return matrix
    
    
class BarElement(Element):
    # NOTE! 6 DOFS per node is sort of hard coded. Should be possible to fix,
    # but all methods and functions should be checked.
    pass

class BeamElement(Element):
    def __init__(self, nodes, label=None, section=Section()):
        super().__init__(nodes, label=label, section=section, dofs_per_node=6)
        self.phi_y, self.phi_z = self.get_phi()
        
    def get_phi(self):
        if self.section.shear_deformation:
            denom = self.section.kappa*self.section.G*self.section.A*self.length()**2
            phi_y = 12*self.section.E*self.section.I_y/denom
            phi_z = 12*self.section.E*self.section.I_z/denom
        else:
            phi_y = 0
            phi_z = 0
        
        return phi_y, phi_z  
    

    def stiffness(self):
        # EN234: Three-dimentional Timoshenko beam element undergoing axial, torsional and bending deformations
        # Fang, 2015 (Wengqiang_Fan.pdf)

        A = self.section.A
        G = self.section.get_G()
        L = self.length()
        E = self.section.E
        I_z = self.section.I_z
        I_y = self.section.I_y
        J = self.section.J       
        k_axial = E*A/L

        phi_y, phi_z = self.phi_y, self.phi_z
        
        mu_z = 1/(1+phi_z)
        mu_y = 1/(1+phi_y)
        
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

        
    def mass(self):
        # https://link.springer.com/content/pdf/10.1007%2F978-1-84996-190-5_1.pdf    

        I_z = self.section.I_z
        I_y = self.section.I_y
        
        phi_y, phi_z = self.phi_y, self.phi_z
        mu_z = 1/(1+phi_z)
        mu_y = 1/(1+phi_y)

        m = self.section.m
        L = self.length()
        
        A = self.section.A
        Ip = self.section.J
        
        if self.section.lumped_mass:
            #old from http://what-when-how.com/the-finite-element-method/fem-for-frames-finite-element-method-part-1/   
            # a = L*0.5
            # me = m * a/105.0 * np.array([
            #         [70, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0],
            #         [0, 78, 0, 0, 0, 22*a, 0, 27, 0, 0, 0, -13*a],
            #         [0, 0, 78, 0, -22*a, 0, 0, 0, 27, 0, 13*a, 0],
            #         [0, 0, 0, 70*r_x**2, 0, 0, 0, 0, 0, -35*r_x**2, 0, 0],
            #         [0, 0, 0, 0, 8*a**2, 0, 0, 0, -13*a, 0, -6*a**2, 0],
            #         [0, 0, 0, 0, 0, 8*a**2, 0, 13*a, 0, 0, 0, -6*a**2],
            #         [0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0],
            #         [0, 0, 0, 0, 0, 0, 0, 78, 0, 0, 0, -22*a],
            #         [0, 0, 0, 0, 0, 0, 0, 0, 78, 0, 22*a, 0],
            #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 70*r_x**2, 0, 0],
            #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8*a**2, 0],
            #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8*a**2]
            # ])
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

        else:
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
    
    
    def geometric_stiffness(self):
        
        L = self.length()
        N = self.section.N0
        # kappa = self.kappa
    
        if self.section.shear_deformation:
            print('Timoshenko formulation (shear deformation) not implemented for geometric stiffness. Using Euler-Bernoulli.')
        
        #Copied from paajthon/brutils
        kg = np.array([
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
    
        return kg
