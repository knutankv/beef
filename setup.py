import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beef-knutankv",
    version="0.1.0",
    author="Knut Andreas Kvåle",
    author_email="knut.a.kvale@gmail.com",
    description="BEam Elements Framework for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knutankv/beef",
    packages=setuptools.find_packages(),
    install_requires=['scipy', 'numpy', 'matplotlib', 'vispy', 'tqdm', 'PyQt5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)