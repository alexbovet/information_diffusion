#!/usr/bin/env python3
# Author: Alexandre Bovet <alexandre.bovet (at) maths.ox.ac.uk>, 2020
#
# License: GNU General Public License v3.0

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


setup(
    ext_modules = cythonize("CIcython.pyx"),
    include_dirs=[numpy.get_include()]
)


