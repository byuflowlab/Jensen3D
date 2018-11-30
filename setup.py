#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

module1 = Extension('_jensen', sources=['src/jensen3d/jensen.f90'], extra_compile_args=['-O2', '-c'])
module2 = Extension('_jensen2', sources=['src/jensen3d/jensencosineWEC.f90'], extra_compile_args=['-O2', '-c'])


setup(
    name='Jensen3D',
    version='0.0.0',
    description='several variants of the Jensen wake model',
    install_requires=['openmdao>=1.6.3'],
    package_dir={'': 'src'},
    ext_modules=[module1, module2],
    packages=['jensen3d'],
    license='Apache License, Version 2.0',
)
