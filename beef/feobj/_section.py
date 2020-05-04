from . import *

#%% Section class method
class Section:
    def __init__(self, E=0, rho=0, A=0, I_y=0, I_z=0, 
                 poisson=0.3, m=None, G=None, kappa=1, J=0, 
                 N0=0, e2=None, shear_deformation=False, lumped_mass=False):
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
        self.I_y = I_y
        self.I_z = I_z
        self.J = J
        self.kappa = kappa

        # Load        
        self.N0 = N0
        
        # Beam direction
        if e2 is not None:
            e2 = np.array(e2)
            
        self.e2 = e2        
        self.shear_deformation = shear_deformation
        
        # Compute stuff
        self.G = self.get_G()
        self.m = self.get_m()


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
