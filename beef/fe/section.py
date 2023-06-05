'''
FE objects submodule: section definition
'''

import numpy as np

class Section:
    '''
    Section definition class.

    Arguments
    ------------
    E : float, optional
        Youngs modulus (0.0 is standard)
    rho : float, optional
        mass density (0.0 is standard)
    A : float, optional
        area of beam cross section (infinity is standard)
    I_y : float, optional
        second moment of area about y-axis (0.0 is standard)
    I_z : float, optional
        second moment of area about z-axis (None is standard), only applicable in 3d systems
    poisson : 0.3, optional
        Poisson ratio
    m : float, optional
        mass per unit length (None is standard) - if not defined, this is computed from mass density and area
    G : float, optional
        shear modulus (None is standard) - if not defined, computed from the Poisson ratio and the Youngs modulus
    kappa : float, optional
        Timoshenko shear coefficient (1.0 is standard)
    J : float, optional
        polar moment of area, relevant for torsional response
    e2 : float, optional
        vector (3x1 in 3d, 2x1 in 2d) describing the second perpendicular vector of the element - if not given, it is generated automatically
    shear_deformation : False, optional
        whether or not to include shear deformation in the creation of element stiffness matrices
    mass_formulation : {'euler', 'timoshenko', 'lumped'}
        type of mass formulation; 'euler' and 'timoshenko' refers to the type of interpolation of the stiffness matrix
    name : str, optional   
        name of section
    '''

    def __init__(self, E=0.0, rho=0.0, A=np.inf, I_y=0.0, I_z=None, 
                 poisson=0.3, m=None, G=None, kappa=1.0, J=0.0,
                 e2=None, shear_deformation=False, mass_formulation='euler', name=None):

        self.name = name

        # Other
        self.mass_formulation = mass_formulation
        
        # Material
        self.E = E
        self.G = G
        self.poisson = poisson
    
        self.m = m
        self.rho = rho
        
        # Cross-section
        self.A = A
        if I_z == None:
            I_z = np.nan
        
        self.I = np.hstack([I_y, I_z])
        self.J = J
        self.kappa = kappa
                
        # Beam direction  
        self.shear_deformation = shear_deformation
        
        # Compute stuff
        self.G = self.get_G()
        self.m = self.get_m()

    
    # CORE METHODS
    def __str__(self):
        return f'BEEF Section: {self.name}'

    def __repr__(self):
        return f'BEEF Section: {self.name}'

    # USEFUL
    def get_m(self):
       '''
       Return mass per unit length. 

       Returns
       ---------
       m : float
            if `self.m` is not None, this is output; otherwise, the value `self.rho*self.A` is output

       '''
       if self.m is None:
           return self.rho*self.A
       else:
           return self.m
       
    def get_G(self):
        '''
        Return shear modulus G.

        Returns
        ---------
        G : float
                if `self.G` is not None, this is output; otherwise, the value is established based on the Poisson ratio and the Youngs modulus

        '''        
        if self.G is None:
            return self.E/(2*(self.poisson+1))
        else:
            return self.G
