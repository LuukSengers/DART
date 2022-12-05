#!/usr/bin/env python

from distutils.core import setup

setup(name='dart',
      version='0.1',
      description='The code from the paper wes-2021-43',
      author='Balthazar Sengers',
      author_email='balthazar.sengers@uni-oldenburg.de', 
      install_requires=['dart','matplotlib', 'numpy', 'pandas', 'scikit-learn', 'pytest']
)
