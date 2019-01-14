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

import random, pickle

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
    __slots__ = ['values', '_SSE', 'p']

    def __init__(self, p, values=None):
        """Constructor"""
        self.p = p
        if values is None:
            self.values = np.empty(p.Nd)
        else: self.update(values)
        self.SSE = None

    def __repr__(self):
        """
        Informative string representation, with SSE and pretty-printed values.
        """
        prelude = "" if self.SSE is None else sub(
            "SSE={:.4f}", float(self.SSE))
        return sub("<{}>", self.p.pm.prettyValues(self.values, prelude))

    @property
    def SSE(self):
        """
        My SSE value as a float. Infinite if no SSE computed yet or was
        set to C{None}.
        """
        if self._SSE is None:
            return np.inf
        return float(self._SSE)
    @SSE.setter
    def SSE(self, x):
        self._SSE = x
    
    def spawn(self, values):
        """
        Returns another instance of my class with the same population and
        values.
        """
        return Individual(self.p, values)
    
    def copy(self):
        """
        Returns a complete copy of me, another instance of my class with
        the same population, values, and SSE.
        """
        i = Individual(self.p, list(self.values))
        i.SSE = self.SSE
        return i

    def __float__(self):
        """
        I can be evaluated as a float using my SSE.
        """
        return self.SSE
    
    def __getitem__(self, k):
        """
        Sequence-like read access to values.
        """
        return self.values[k]

    def __setitem__(self, k, value):
        """
        Sequence-like write access to values.
        """
        self.values[k] = value

    def __iter__(self):
        """
        Sequence-like iteration over values.
        """
        for value in self.values:
            yield value

    def __eq__(self, other):
        """
        I am equal to another C{Individual} if we have the same SSE and
        values.
        """
        return self.SSE == other.SSE and self.equals(other)

    def equals(self, other):
        if hasattr(other, 'values'):
            other = other.values
        if not hasattr(other, 'shape'):
            other = np.array(other)
        return np.all(self.values == other)

    def __lt__(self, other):
        """
        My SSE less than other C{Individual} or float?
        """
        return float(self) < float(other)

    def __gt__(self, other):
        """
        My SSE greater than other C{Individual} or float?
        """
        return float(self) > float(other)
    
    def __sub__(self, other):
        """
        Subtract the other C{Individual}'s values from mine and return a
        new C{Individual} whose values are the differences.
        """
        return self.spawn(self.values - other.values)

    def __add__(self, other):
        return self.spawn(self.values + other.values)

    def __mul__(self, F):
        """
        Scales each of my values by the constant I{F}.
        """
        return self.spawn(self.values * F)

    def blacklist(self):
        """
        Sets my SSE to the worst possible value and forces my population
        to update its sorting to account for my degraded status.
        """
        self.SSE = np.inf
        del self.p.iSorted
    
    def update(self, values):
        if len(values) != self.p.Nd:
            raise ValueError(sub(
                "Expected {:d} values, not {:d}", self.p.Nd, len(values)))
        if isinstance(values, (list, tuple)):
            values = np.array(values)
        self.values = values
    
    def evaluate(self):
        """
        Computes the sum of squared errors (SSE) from my evaluation
        function using my current I{values}. Stores the result in my
        I{SSE} attribute and returns a reference to me for
        convenience.
        """
        def done(SSE):
            self.p.counter += 1
            self.SSE = SSE
            return self
        return self.p.evalFunc(self.values).addCallback(done)

    @property
    def params(self):
        pd = {}
        for k, value in enumerate(self):
            name = self.p.pm.names[k]
            pd[name] = value
        return pd
    
    def save(self, filePath):
        """
        Saves my parameters to I{filePath} as a pickled dict.
        """
        with open(filePath, 'wb') as fh:
            pickle.dump(self.params, fh)
