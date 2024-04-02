import setuptools

long_description = '''
BEEF is a BEam Elements Framework package for Python. The main features are:

* 2D and 3D beam finite element analyses
* Solves static and dynamic problems
* Linear and co-rotated formulations, allowing for large-displacement problems (only 2D currently, 3D is work in progress)
* Newmark, Alpha-HHT, static solvers, eigenvalue solution, eigenbuckling solution
* Postprocessing tools for visualization
* Custom function inputs for post-processing
'''

setuptools.setup(
    name="beef-knutankv",
    version="0.5.2",
    author="Knut Andreas Kvåle",
    author_email="knut.a.kvale@gmail.com",
    description="BEam Elements Framework for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knutankv/beef",
    packages=setuptools.find_packages(),
    install_requires=['scipy', 'numpy', 'matplotlib', 'tqdm', 'trame', 'pyvista', 'dill'],    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)