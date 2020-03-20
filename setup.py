#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018-19 by Edwin A. Suominen,
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
required = [
    'Twisted', 'numpy', 'scipy', 'matplotlib', 'pydoe',
    # Other EAS projects
    'AsynQueue>=0.9.8', 'yampex>=0.9.5',
]


### Define setup options
kw = {'version': '1.3.3',
      'license': 'Apache License (2.0)',
      'platforms': 'OS Independent',

      'url': "http://edsuom.com/{}.html".format(NAME),
      'project_urls': {
          'GitHub': "https://github.com/edsuom/{}".format(NAME),
          'API': "http://edsuom.com/{}/{}.html".format(
              NAME, NAME.lower()),
          },
      'author': "Edwin A. Suominen",
      'author_email': "foss@edsuom.com",
      'maintainer': 'Edwin A. Suominen',
      'maintainer_email': "foss@edsuom.com",
      
      'install_requires': required,
      'packages': ['ade', 'ade.test', 'ade.scripts', 'ade.examples'],
      'package_data': {
          'ade.examples': ['*.c'],
      },
      'entry_points': {
          'console_scripts': [
              'ade-examples = ade.scripts.examples:extract',
              "lgg = ade.scripts.lgg:main",
              "pv = ade.scripts.pv:main",
          ],
      },
      'zip_safe': True,
      'long_description_content_type': "text/markdown",
}

kw['keywords'] = [
    'Twisted', 'asynchronous',
    'differential evolution', 'de', 'genetic algorithm', 'evolution',
]


kw['classifiers'] = [
    'Development Status :: 5 - Production/Stable',

    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Framework :: Twisted',

    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

# You get 77 characters. Use them wisely.
#----------------------------------------------------------------------------
#        10        20        30        40        50        60        70
#2345678901234567890123456789012345678901234567890123456789012345678901234567
kw['description'] = " ".join("""
Asynchronous Differential Evolution, with efficient multiprocessing.
""".split("\n"))

kw['long_description'] = """
Performs the Differential Evolution (DE) algorithm
asynchronously. With a multiprocess evaluation function running on a
multicore CPU or cluster, *ade* can get the DE processing done several
times faster than standard single-threaded DE. It does this without
departing in any way from the numeric operations performed by the
classic Storn and Price algorithm. You can use either a randomly
chosen candidate or the best available candidate.

You get a substantial multiprocessing speed-up and the
well-understood, time-tested behavior of the classic DE/rand/1/bin or
DE/best/1/bin algorithm. (You can pick which one to use, or, thanks to
a special *ade* feature, pick a probabilistic third version that
effectively operates at a selected midpoint between the extremes of
"random" and "best.") The underlying numeric recipe is not altered at
all, but everything runs a lot faster.

The *ade* package also does simple and smart population initialization,
informative progress reporting, adaptation of the vector differential
scaling factor *F* based on how much each generation is improving, and
automatic termination after a reasonable level of convergence to the
best solution.

Comes with a couple of small and informative [example
files](http://edsuom.com/ade/ade.examples.html), which you can install
to an *ade-examples* subdirectory of your home directory by typing
`ade-examples` as a shell command.

For a tutorial and more usage examples, see the [project
page](http://edsuom.com/ade.html) at **edsuom.com**.

"""

### Finally, run the setup
setup(name=NAME, **kw)
print("\n" + '-'*79)
print("To create a subdirectory 'ade-examples' of example files")
print("in the current directory, you may run the command 'ade-examples'.")
print("It's not required to use the ade package, but you might find")
print("it instructive.\n")
