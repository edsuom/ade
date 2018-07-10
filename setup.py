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
kw = {'version':'0.8.1',
      'license':'Apache License (2.0)',
      'platforms':'OS Independent',

      'url':"http://edsuom.com/{}.html".format(NAME),
      'author':"Edwin A. Suominen",
      'author_email':"foss@edsuom.com",
      'maintainer':'Edwin A. Suominen',
      'maintainer_email':"foss@edsuom.com",
      
      'install_requires':required,
      'packages':['ade', 'ade.test'],
      'package_data':        {
          'ade': ['examples/*'],
      },
      'entry_points':      {
          'console_scripts': [
              'ade-examples = ade:extract_examples',
          ],
      },
      'zip_safe':True,
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
Performs the Differential Evolution (DE) algorithm
asynchronously. With a multiprocess evaluation function running on a
multicore CPU or cluster, ade can get the DE processing done several
times faster than standard single-threaded DE. It does this without
departing in any way from the numeric operations performed by the
classic Storn and Price algorithm with either a randomly chosen
candidate or the best available candidate.

You get a substantial multiprocessing speed-up and the
well-understood, time-tested behavior of the classic DE/rand/1/bin or
DE/best/1/bin algorithm. (You can pick which one to use.) The
underlying numeric recipe is not altered at all, but everything runs a
lot faster.
"""

### Finally, run the setup
setup(name=NAME, **kw)
print("\n" + '-'*79)
print("To create a subdirectory 'ade-examples' of example files")
print("in the current directory, you may run the command 'ade-examples'.")
print("It's not required to use the ade package, but you might find")
print("it instructive.")
