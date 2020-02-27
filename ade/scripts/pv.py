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


"""
C{pv}: Parses a data file containing a pickled L{History} object,
which gets produced when you call L{Population.save} with the file
path.

This is very handy for seeing whether you have to redefine bounds for
any of your parameters. It is often useful to tighten bounds for
particular problems. Or you might need to loosen things up if you see
the values for a given parameter crowding onto one bound or the other.

Specify the data file for the pickled Population object with the first
argument.
"""

from ade.population import Population
from ade.util import *


args = Args(
    """
    ADE Parameter Viewer.

    Provides two different types of plots for visualizing the
    parameters in your Population: (1) the parameter values vs SSE,
    and (2) most highly correlated parameter values vs each other,
    with greater emphasis on parameter value pairs with lower SSE.
    
    The first and possibly only positional argument is the path of a
    data file containing a pickled History object, produced from
    calling Population.save with the file path.

    Any additional positional arguments must be names of parameters to
    limit the plotting to. If none are supplied, all parameters will
    be plotted. This is usually what you'll wind up doing.

    The -r and -i options only apply to parameter value vs SSE plots,
    not to correlation plots. If -r is not specified, no such plots
    will be included.

    The -N option only applies to parameter correlation plots, which
    will not be included if -N is not specified. A recommended value
    of N is 4. (If you have more highly correlated parameters than
    that, you should rethink your model.
    
    """
)
args('-r', '--max-ratio', 0.0,
     "Show parameter vs SSE plots, limited to this max/min SSE ratio")
args('-i', '--in-pop',
     "Only individuals currently in the population")
args('-N', '--N-correlates', 0,
     "Show the N most correlated pairs of all parameters")
args('-v', '--verbose', "Print info about parameters to STDOUT")
# Positional argument
args("<pickle file> [param1 param2 ...]")


def main():
    """
    The C{pv} script entry point.
    """
    if not len(args):
        raise RuntimeError("No pickle file specified!")
    filePath = args[0]
    p = Population.load(os.path.expanduser(filePath))
    analyzer = p.history.a
    names = args[1:] if len(args) > 1 else []
    pt = None
    if args.r:
        pt = analyzer.plot(names, maxRatio=args.r, inPop=args.i, noShow=True)
    if args.N:
        if names:
            for name in names:
                pt = analyzer.plotCorrelated(
                    name=name, N=args.N, noShow=True, verbose=args.v)
        else:
            pt = analyzer.plotCorrelated(N=args.N, noShow=True, verbose=args.v)
    if pt is None: raise RuntimeError("No analysis done!")
    pt.showAll()


if __name__ == '__main__' and not args.h:
    main()
