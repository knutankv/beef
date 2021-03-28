# External functions
import numpy as np
from scipy.sparse.linalg import eigsh
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

import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
   from tqdm import tqdm