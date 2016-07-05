#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

setup(
    name='Jensen3D',
    version='0.0.0',
    description='several variants of the Jensen wake model',
    install_requires=['openmdao>=1.6.3'],
    package_dir={'': 'src'},
    packages=['jensen3d'],
    license='Apache License, Version 2.0',
)
