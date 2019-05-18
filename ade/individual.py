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
An L{Individual} class for parameter combinations to be evaluated.

You won't need to construct individuals directly. Just let
L{Population} set up a population full of them, and have
L{DifferentialEvolution} create challengers as it does its thing.
"""

import random, pickle, time

import numpy as np
from twisted.internet import defer, task

from util import sub, msg


class Individual(object):
    """
    I act like a sequence of parameter values, but with other stuff
    like an SSE value, too.
    
    Construct me with a L{Population} object. You can set my values
    with a 1-D Numpy array of initial I{values}.

    You can iterate my values in sequence. You can access them
    (read/write) as items. You can even replace the whole 1-D Numpy
    array of them at once with another array of like dimensions,
    although the safest way to do that is supplying a list or 1-D
    array to L{update}.

    @ivar values: A 1-D Numpy array of parameter values.

    @ivar p: The L{Population} I am part of.

    @ivar dt: The time difference between start and end of my last
        evaluation.

    @keyword values: Set to a sequence of initial parameter values if
        you're not going to set them with a call to L{update}.
    """
    __slots__ = ['values', '_SSE', 'p', 'dt']

    def __init__(self, p, values=None):
        """Individual(p, values=None)"""
        self.p = p
        if values is None:
            self.values = np.empty(p.Nd)
        else: self.update(values)
        self.SSE = None
        self.dt = None

    def __getstate__(self):
        """
        For pickling. Does not include the L{Population} object I{p}.
        """
        state = {}
        for name in {'values', '_SSE', 'dt'}:
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """
        For unpickling. You have to set the I{p} attribute of the
        unpickled version of me to a L{Population} object.
        """
        for name in state:
            setattr(self, name, state[name])
        
    def __repr__(self):
        """
        Informative string representation, with SSE and pretty-printed values.
        """
        prelude = "" if self.SSE is None else sub(
            "SSE={:.4f}", float(self.SSE))
        return sub("<{}>", self.p.pm.prettyValues(self.values, prelude))

    @property
    def params(self):
        """
        Property: A dict of my parameter values, keyed by name.
        """
        pd = {}
        for k, value in enumerate(self):
            name = self.p.pm.names[k]
            pd[name] = value
        return pd
    
    @property
    def SSE(self):
        """
        Property: My SSE value, which must at least behave like a
        float. Infinite if no SSE computed yet, a fatal error occurred
        during evaluation, or was set to C{None}.
        """
        if self._SSE is None or self._SSE < 0:
            return float('+inf')
        return self._SSE #float(self._SSE)
    @SSE.setter
    def SSE(self, x):
        """
        Property setter: Sets my SSE value.
        """
        self._SSE = x
    
    def spawn(self, values):
        """
        Returns another instance of my class with the same population and
        with the supplied I{values}.
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
        I can be evaluated as a float using my SSE. A negative SSE
        translates to C{inf} because it indicates a fatal error.
        """
        return float('+inf' if self.SSE < 0 else self.SSE)
    
    def __getitem__(self, k):
        """
        Sequence-like read access to values when I{k} is an integer,
        dict-like access otherwise.
        """
        if not isinstance(k, int):
            k = self.p.pm.names.index(k)
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

    def __nonzero__(self):
        """
        I am C{True} if there were no fatal errors during my last
        evaluation, which would be indicated by an evaluation SSE
        result of less than zero.

        I will evaluate as C{True} even if my SSE is C{None},
        infinite, or C{NaN}, so long as it is not negative.
        """
        if self._SSE is None: return True
        SSE = float(self._SSE)
        if np.isnan(SSE): return True
        return SSE >= 0

    def __hash__(self):
        return hash(bytes(self.SSE) + self.values.tobytes())
    
    def __eq__(self, other):
        """
        I am equal to another C{Individual} if we have the same SSE and
        values.
        """
        return self.SSE == other.SSE and self.equals(other)

    def equals(self, other):
        """
        Returns C{True} if my values equal the I{other} individual's
        values, or the other values themselves if supplied as a
        sequence or 1-D array.
        """
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
        """
        Add the other C{Individual}'s values to mine and return a new
        C{Individual} whose values are the sums.
        """
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
        self.SSE = float('+inf')
        del self.p.KS
    
    def update(self, values):
        """
        Updates my I{values} as an array form of the supplied list or
        tuple.

        Raises an exception if there's a different number of values
        than my values length I{Nd}.
        """
        if len(values) != self.p.Nd:
            raise ValueError(sub(
                "Expected {:d} values, not {:d}", self.p.Nd, len(values)))
        if isinstance(values, (list, tuple)):
            values = np.array(values)
        self.values = values
    
    def evaluate(self, xSSE=None):
        """
        Computes the sum of squared errors (SSE) from my evaluation
        function using my current I{values}.

        Stores the result in my I{SSE} attribute and returns a
        reference to me for convenience.

        If the SSE value is less than zero, or results in a Twisted
        failure, I{ade} will abort operations. Use this feature to
        provide your evaluator with a simple way to stop everything if
        something goes terribly wrong.

        Updates my I{dt} attribute with the elapsed time for this
        evaluation.

        Returns a C{Deferred} that fires with a reference to my
        instance when the evaluation is done, pass or fail.
        """
        def done(SSE):
            self.dt = time.time() - t0
            self.p.counter += 1
            self.SSE = SSE
            return self
        def failed(failureObj):
            info = failureObj.getTraceback()
            msg(0, "FATAL ERROR in evaluation:\n{}\n{}\n", '-'*40, info)
            self.SSE = -1
            return self
        t0 = time.time()
        if xSSE is None:
            d = self.p.evalFunc(self.values)
        else: d = self.p.evalFunc(self.values, xSSE=xSSE)
        d.addCallbacks(done, failed)
        return d

    def save(self, filePath):
        """
        Saves my parameters to I{filePath} as a pickled dict.
        """
        with open(filePath, 'wb') as fh:
            pickle.dump(self.params, fh)
