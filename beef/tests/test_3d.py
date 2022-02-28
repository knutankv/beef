from beef import fe
import beef
from beef.newmark import factors as newmark_factors, factors_from_alpha

import numpy as np

from vispy import app, gloo
app.use_app('pyqt5')
import matplotlib.pyplot as plt

# Define sections
# Deck
deck_section_params = dict(
    A=1,
    m=5000,
    I_z=1e-2,
    I_y=1e-2,
    J=1e-12,
    E=210e9,
    poisson=0.3
    )

deck_section = fe.Section(**deck_section_params, name='Deck beam section')

# Tower
tower_section_params = dict(
    A=1,
    m=5000,
    I_z=1e-2,
    I_y=1e-2,
    J=1e-12,
    E=210e9,
    poisson=0.3
    )

tower_section = fe.Section(**tower_section_params, name='Tower beam section')

rayleigh=dict(mass=1e-3, stiffness=1e-3)

#%% Define geometry and mesh
node_matrix1 = np.array([[1, 0, 0, 0],
                         [2, 10, 10, 0],
                         [3, 20, 20, 0],
                         [4, 30, 30, 0],
                         [5, 40, 40, 0],
                         [6, 50, 50, 0],
                         [7, 60, 60, 0]
                        ])

node_matrix2 = np.array([[101, 20, 20, -30],
                         [102, 20, 20, 0],
                         [103, 40, 40, -30],
                         [104, 40, 40, 0]
                        ])

element_matrix1 = np.array([[1, 1, 2],
                            [2, 2, 3],
                            [3, 3, 4],
                            [4, 4, 5],
                            [5, 5, 6],
                            [6, 6, 7]]) 

element_matrix2 = np.array([[7, 101, 102],
                            [8, 103, 104]]) 

sections1 = [deck_section]*element_matrix1.shape[0]
sections2 = [tower_section]*element_matrix2.shape[0]
sections = sections1 + sections2 

#%%
# Define constraints and assembly
tie_nodes = [[3, 102], [5, 104]]
constraints_tie = [fe.Constraint(nodes, dofs='all', node_type='beam3d') for nodes in tie_nodes] 
constraints_fix = [fe.Constraint([1, 7, 101, 103], dofs='all', node_type='beam3d')]

constraints = constraints_tie + constraints_fix

part1 = fe.Part(node_matrix1, element_matrix1, sections1)
part2 = fe.Part(node_matrix2, element_matrix2, sections2)

assembly = fe.Assembly([part1, part2], constraints=constraints)
assembly.plot(node_labels=True, element_labels=True)

#%%
# Create analysis
forces = [fe.Force([4], [0], [-500e3])] # 500 kN downwards in center node
analysis = fe.Analysis(assembly, forces=forces)
analysis.run_static()

analysis.eldef.plot(overlay_deformed=True, plot_nodes=True)