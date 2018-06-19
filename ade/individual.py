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


"""
An Individual class for parameter combinations that occupy a
Population and get evaluated.
"""

import random

import numpy as np
from twisted.internet import defer, task

from util import sub


class Individual(object):
    """
    Construct me with a C{Population} object. You can set my values
    with a 1-D Numpy array of initial I{values}.

    I act like a sequence of values. You can iterate my values in
    sequence. You can access them (read/write) as items. You can even
    replace the whole 1-D Numpy array of them at once with another
    array of like dimensions, although the safest way to do that is
    supplying a list or 1-D array to L{update}.

    @ivar values: A 1-D Numpy array of parameter values.
    """
    __slots__ = ['values', 'SSE', 'partial_SSE', 'p']

    def __init__(self, p, values=None):
        self.p = p
        if values is None:
            self.values = np.empty(p.Nd)
        else: self.update(values)
        self.SSE = None
        self.partial_SSE = False

    def __repr__(self):
        prelude = "" if self.SSE is None else sub("SSE={:.4f}", self.SSE)
        return sub("<{}>", self.p.pm.prettyValues(self.values, prelude))

    def spawn(self, values):
        return Individual(self.p, values)
    
    def __getitem__(self, k):
        return self.values[k]

    def __setitem__(self, k, value):
        self.values[k] = value

    def __iter__(self):
        for value in self.values:
            yield value

    def __len__(self):
        return self.Nd

    def __nonzero__(self):
        return bool(self.SSE)
    
    def __eq__(self, other):
        return self.SSE == other.SSE

    def equals(self, other):
        if hasattr(other, 'values'):
            other = other.values
        if not hasattr(other, 'shape'):
            other = np.array(other)
        return np.all(self.values == other)

    def __lt__(self, other):
        if self.SSE is None:
            return False
        if other.SSE is None:
            return True
        return self.SSE < other.SSE

    def __gt__(self, other):
        if self.SSE is None:
            return True
        if other.SSE is None:
            return False
        return self.SSE > other.SSE
    
    def __sub__(self, other):
        return self.spawn(self.values - other.values)

    def __add__(self, other):
        return self.spawn(self.values + other.values)

    def __mul__(self, F):
        """
        Scales each of my values by the constant I{F}.
        """
        return self.spawn(self.values * F)

    def update(self, values):
        if len(values) != self.p.Nd:
            raise ValueError(sub(
                "Expected {:d} values, not {:d}", self.p.Nd, len(values)))
        if isinstance(values, (list, tuple)):
            values = np.array(values)
        self.values = values
    
    def evaluate(self, xSSE=None):
        """
        Computes the sum of squared errors (SSE) from my evaluation
        function using my current I{values}. Stores the result in my
        I{SSE} attribute and returns a reference to me for
        convenience.
        """
        def done(SSE):
            self.p.counter += 1
            self.SSE = SSE
            self.partial_SSE = xSSE is not None
            return self
        return self.p.evalFunc(self.values, xSSE).addCallback(done)

