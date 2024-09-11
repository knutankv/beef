![BEEF logo](https://raw.githubusercontent.com/knutankv/beef/master/beef-logo.svg)
=======================

What is beef?
=======================
BEEF is a BEam Elements Framework package for Python. The main features are:

* 2D and 3D beam finite element analyses
* Solves static and dynamic problems
* Linear and co-rotated formulations, allowing for large-displacement problems (3D is experimental)
* Newmark, Alpha-HHT, static solvers, eigenvalue solution, eigenbuckling solution
* Postprocessing tools for visualization
* Custom function inputs for post-processing


Installation 
========================
Either download the repository to your computer and install, e.g. by **pip**

```
pip install .
```

or install directly from github:

```
pip install git+https://www.github.com/knutankv/beef.git@master
```


Quick start
=======================
Import the relevant package modules, exemplified for the `newmark` module, as follows:
    
```python
from beef import newmark
```

To access the classes used to construct FE objects, the `fe` module has to be imported (usage is exemplified by creation of part objects and placement in assembly):

```python
from beef import fe
first_part = fe.Part(node_matrix_1, element_matrix_1, sections_1, constraints=constraints_1)
second_part = fe.Part(node_matrix_2, element_matrix_2, sections_2, constraints=constraints_2)
assembly = fe.Assembly([first_part, second_part])
```    

For details on how to set up a full model, please refer to the examples. For code reference visit [knutankv.github.io/beef](https://knutankv.github.io/beef/).

Examples
=======================
Examples are provided as Jupyter Notebooks in the [examples folder](https://github.com/knutankv/beef/tree/master/examples).

References
=======================
<a id="1">[1]</a> 
S. Krenk, Non-linear modeling and analysis of solids and structures. Cambridge University Press, 2009.

<a id="2">[2]</a>
W. Fang, EN234: Three-dimentional Timoshenko beam element undergoing axial, torsional and bending deformations, 2015. https://www.brown.edu/Departments/Engineering/Courses/En2340/Projects/Projects_2015/Wenqiang_Fan.pdf

<a id="3">[3]</a>
H. Karadeniz, M. P. Saka, and V. Togan, “Finite Element Analysis of Space Frame Structures BT  - Stochastic Analysis of Offshore Steel Structures: An Analytical Appraisal,” H. Karadeniz, Ed. London: Springer London, 2013, pp. 1–119. https://link.springer.com/content/pdf/10.1007%2F978-1-84996-190-5_1

<a id="4">[4]</a> P.I. Bruheim, Development and validation of a finite element software facilitating large-displacement aeroelastic analysis of wind turbines, Norwegian University of Science and Technology, 2012. https://core.ac.uk/download/pdf/30855172.pdf


Citation
=======================
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8090990.svg)](https://doi.org/10.5281/zenodo.8090990)

Support
=======================
Please [open an issue](https://github.com/knutankv/beef/issues/new) for support.

