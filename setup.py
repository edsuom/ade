#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018 by Edwin A. Suominen,
# http://edsuom.com/ade
#
# See edsuom.com for API documentation as well as information about
# Ed's background and other projects, software and otherwise.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS
# IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language
# governing permissions and limitations under the License.


NAME = "ade"


### Imports and support
from setuptools import setup

### Define requirements
required = ['Twisted', 'numpy', 'scipy', 'matplotlib', 'pydoe', 'AsynQueue']


### Define setup options
kw = {'version':'0.9',
      'license':'Apache License (2.0)',
      'platforms':'OS Independent',

      'url':"http://edsuom.com/{}.html".format(NAME),
      'author':"Edwin A. Suominen",
      'author_email':"foss@edsuom.com",
      'maintainer':'Edwin A. Suominen',
      'maintainer_email':"foss@edsuom.com",
      
      'install_requires':required,
      'packages':['ade', 'ade.test'],
      
      'zip_safe':False,
}

kw['keywords'] = [
    'Twisted', 'asynchronous',
    'differential evolution', 'de', 'genetic algorithm', 'evolution',
]


kw['classifiers'] = [
    'Development Status :: 4 - Beta',

    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Framework :: Twisted',

    'Topic :: Software Development :: Libraries :: Python Modules',
]


kw['description'] = " ".join("""
Asynchronous Differential Evolution, with efficient multiprocessing.
""".split("\n"))

kw['long_description'] = """
TODO
"""

### Finally, run the setup
setup(name=NAME, **kw)

