#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018-20 by Edwin A. Suominen,
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
The L{Constraints} base class makes it easier for you to enforce
parameter constraints. If you have any highly correlated parameters,
an instance of L{RelationsChecker} may be helpful, too.
"""

import numpy as np
from ade.util import *


class RelationsChecker(object):
    """
    Checks that the linear relation between two parameters is within
    the limits imposed by the 'relations' dict-of-dicts specification.

    Each entry of the dict, keyed by a first parameter name, is
    another dict with its entries keyed by a second parameter
    name. Each of those entries is a 3-tuple with the slope I{m},
    y-intercept I{b}, and maximum deviation I{yMaxErr} in value of the
    second parameter from its nominal value as determined by
    M{y2(y1)=m*y1+b}.
    """
    def __init__(self, relations):
        self.relations = relations

    def __call__(self, params):
        for xName in self.relations:
            x = params[xName]
            ys = self.relations[xName]
            for yName in ys:
                y = params[yName]
                m, b, yMaxErr = ys[yName]
                yr = m*x + b
                if abs(yr-y) > yMaxErr:
                    return False
        return True


class Constraints(object):
    """
    Subclass me and define one or more constraint-checking
    methods.

    Register the methods to be used with a given instance of that
    subclass by defining a I{registry} dict in your subclass, keyed by
    method name. Each entry must have a 2-sequence, with the first
    item being linear parameter names (or C{None}) and log-space
    parameter names (or C{None}) for the constraint method.

    You can define instance attributes via constructor keywords. Any
    constructor arguments are supplied to the L{setup} method you can
    override in your subclass, which gets called during construction
    right after instance attributes get set by any constructor
    keywords.

    To just add a raw constraint function that gets called without any
    parameter transformations, use L{append}.

    Log-space parameter values are used in the author's ongoing
    circuit simulation project and are supported in this module, but
    not yet implemented at the level of the I{ade} package otherwise.

    @cvar debug: Set C{True} to have failing constraints shown with
        parameters. (Debugging only.)
    """
    debug = False
    
    def __init__(self, *args, **kw):
        for name in kw:
            setattr(self, name, kw[name])
        self.cList = []
        self.setup(*args)
        for methodName in self.registry:
            func = getattr(self, methodName)
            self.cList.append([func]+list(self.registry[methodName]))
        self.fList = []

    def setup(self, *args):
        """
        Override this to do setup with any constructor arguments and with
        instance attributes set via any constructor keywords.
        """
        pass
        
    def __len__(self):
        return len(self.cList) + len(self.fList)

    def __bool__(self):
        return bool(self.cList) or bool(self.fList)
    
    def __iter__(self):
        """
        Iterating over an instance of me yields wrappers of my class-wide
        constraint-checking functions registered in my I{cList}, plus
        any functions registered after setup in my I{fList}.

        You can register a function in I{fList} after setup by calling
        L{append} with the callable as the sole argument. It will be
        treated exactly the same except called after class-wide
        functions.

        Each wrapper automatically transforms any log parameters into
        their linear values before calling the wrapped
        constraint-checking function with a revised parameter
        dict. Also, if a parameter is not present, forces a C{True}
        "constraint satisfied" result so that setting a parameter to
        known doesn't cause bogus constraint checking.

        You can treat an instance of me sort of like a list, using
        iteration and appending. But you can't get or set individual
        items by index.
        """
        def wrapper(params):
            if linearParams:
                if not linearParams.issubset(params):
                    return True
                for name in linearParams:
                    if name not in newParams:
                        newParams[name] = params[name]
            if logParams:
                if not logParams.issubset(params):
                    return True
                for name in logParams:
                    if name not in newParams:
                        newParams[name] = np.power(10.0, params[name])
            result = func(newParams)
            if self.debug:
                print(sub("BOGUS: {}, {}", func.__name__, newParams))
            return result
        
        newParams = {}
        for func, linearParams, logParams in self.cList:
            yield wrapper
        for func in self.fList:
            yield func

    def append(self, func):
        self.fList.append(func)
