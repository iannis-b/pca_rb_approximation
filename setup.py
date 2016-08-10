#!/usr/bin/env python

from setuptools import setup

# Read the long description from the readmefile
with open("README.rst", "rb") as f:
    long_descr = f.read().decode("utf-8")

# Run setup
setup(name='pca_approximation',
      packages=['pca_approximation', ],
      version='0.1',
      description='Approximation of Rayleigh Brillouin scattering using PCA analysis.',
      long_description=long_descr,
      author='Ioannis Binietoglou',
      author_email='ioannis@inoe.ro',
      install_requires=[
          "sphinx",
          "numpydoc",
          "unittest2",
          "numpy",
          "sklearn",
          "cycler",
          "matplotlib",
      ],
      )
