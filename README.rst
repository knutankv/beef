.. image:: beef-logo.svg
    :width: 300px

BEEF
=======================

BEEF is a BEam Elements Framework package for Python, which includes support for 3D beam finite element analyses -- both for static and dynamic problems. Linear and co-rotated, allowing for large-displacement problems, formulations are implemented.
Please refer to the `documentation <https://beef.readthedocs.io/>`_ for more details.


Installation and usage
-----------------------

Either download the repository to your computer and install, e.g. by **pip**

.. code-block::

   pip install .


or install directly from the python package index.

.. code-block::

   pip install git+https://www.github.com/knutankv/beef.git@master


Thereafter, import the package modules, exemplified for the `newmark´ module, as follows:
    
.. code-block:: python

    import beef.newmark

To access the classes used to construct FE objects, the `feeobj´ package has to be imported (usage is exemplified by creation of part objects and placement in assembly):

.. code-block:: python

    import beef.feobj as fe
    first_part = fe.Part(node_matrix_1, element_matrix_1, sections_1, constraints=constraints_1)
    second_part = fe.Part(node_matrix_2, element_matrix_2, sections_2, constraints=constraints_2)
    assembly = fe.Assembly([first_part, second_part])
    


Citation
=======================
...

Support
=======================
Please `open an issue <https://github.com/knutankv/beef/issues/new>`_ for support.
