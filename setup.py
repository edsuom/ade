#!/usr/bin/python

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

