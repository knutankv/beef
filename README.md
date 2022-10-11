![BEEF logo](https://raw.githubusercontent.com/knutankv/beef/master/beef-logo.png)
=======================

What is beef?
=======================
BEEF is a BEam Elements Framework package for Python. The main features are:

* 2D and 3D beam finite element analyses
* Solves static and dynamic problems
* Linear and co-rotated formulations, allowing for large-displacement problems (only 2D currently, 3D is work in progress)
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


Citation
=======================
ZENODO citation element will be created.


Support
=======================
Please [open an issue](https://github.com/knutankv/beef/issues/new) for support.

