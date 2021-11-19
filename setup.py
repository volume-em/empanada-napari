#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


install_deps = [
    'napari',
    'napari-plugin-engine>=0.1.4',
    'imagecodecs',
    'git+https://github.com/volume-em/MitoNet.git#egg=my-git-package'
]

setup(
    name='empanada-napari',
    author='Ryan Conrad',
    author_email='conradrw@nih.gov',
    license='BSD-3',
    url='https://github.com/volume-em/empanada-napari',
    description='Algorithms for Panoptic Segmentation of organelles in EM',
    packages=find_packages(),
    python_requires='>=3.7',
    use_scm_version=True,
    install_requires=install_deps,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Framework :: napari',
    ],
    entry_points={
        'napari.plugin': [
            'empanada-napari = empanada_napari',
        ],
    },
)
