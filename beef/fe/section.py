import numpy as np

#%% Section class method
class Section:

    def __init__(self, E=0, rho=0, A=0, I_y=0, I_z=None, 
                 poisson=0.3, m=None, G=None, kappa=1, J=0,
                 e2=None, shear_deformation=False, lumped_mass=False, name=None):

        self.name = name

        # Other
        self.lumped_mass = lumped_mass
        
        # Material
        self.E = E
        self.G = G
        self.poisson = poisson
    
        self.m = m
        self.rho = rho
        
        # Cross-section
        self.A = A
        if I_z == None:
            I_z = []
        
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
       if self.m is None:
           return self.rho*self.A
       else:
           return self.m
       
    def get_G(self):
        if self.G is None:
            return self.E/(2*self.poisson+1)
        else:
            return self.G
