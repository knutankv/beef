'''
Finite Element (FE) library
========================
This module consists of all classes to create and run FE analyses. 

The following convention is used throughout to stack DOFs from multiple nodes:
$$
\\begin{Bmatrix}
\{u_{1}\}\\\\
\{u_{2}\}\\\\
\\vdots \\\\
\{u_{n}\} \\\\
\\vdots \\\\  
\{u_{N}\}
\\end{Bmatrix}
$$

where the translational DOFs (subscript t) and rotational DOFs (subscript r) 
from node i are stacked as follows:

$$
\\begin{Bmatrix}
\{u_{t}\}\\\\
\{u_{r}\}
\\end{Bmatrix}
$$
'''

# External functions
import numpy as np
from scipy.linalg import null_space, eig
from datetime import datetime

# Import all submodules (splitted for tidyness)
from .node import *
from .element import *

from .section import *
from .constraint import *
from .features import *
from .eldef import *

from .force import *
from .step import *

from .analysis import *
from .results import *

__all__ = ['node', 'element', 'section', 'constraint',
          'features', 'eldef', 'force', 'step', 'analysis', 'results']

import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm