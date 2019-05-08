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
Unit tests for L{ade.individual}.
"""

import time, os.path

import numpy as np

from twisted.internet import defer, reactor

from ade.util import *
from ade import individual

from ade.test import testbase as tb


# If verbose
msg(True)

        
class TestIndividual(tb.TestCase):
    verbose = False

    def setUp(self):
        bounds = [(-5, 5), (-5, 5)]
        self.p = tb.MockPopulation(tb.ackley, bounds, ["x", "y"])

    def spawn(self, values=None):
        return individual.Individual(self.p, values)
        
    def test_init_with_values(self):
        values = np.array([0.5, 0.5])
        i = self.spawn(values)
        i.SSE = 123.456
        self.assertItemsAlmostEqual(list(i), [0.5, 0.5])
        text = repr(i)
        self.assertIn('SSE=123.456', text)
        if self.verbose:
            print text
        
    def test_getAndSet(self):
        i = self.spawn()
        i[1] = 3.14
        self.assertEqual(i[1], 3.14)

    def test_iterate(self):
        values = [10, 20]
        i = self.spawn()
        for k, x in enumerate(values):
            i[k] = x
        count = 0
        for xi, xe in zip(i, values):
            self.assertEqual(xi, xe)
            count += 1
        self.assertEqual(count, 2)
        
    def test_subtract(self):
        i = self.spawn(0.5*np.ones(2))
        iDiff = i - i
        self.assertEqual(iDiff, self.spawn(np.zeros(2)))
        iDiff = i - iDiff
        self.assertEqual(iDiff, i)
        
    def test_add(self):
        i = self.spawn(0.5*np.ones(2))
        iSum = i + i
        self.assertEqual(iSum, self.spawn(np.ones(2)))
        
    def test_multiply(self):
        i = self.spawn(0.5*np.ones(2))
        self.assertEqual(i * 0, self.spawn([0, 0]))
        i = self.spawn(0.5*np.ones(2))
        self.assertEqual(i * 1, self.spawn([0.5, 0.5]))
        i = self.spawn(0.5*np.ones(2))
        self.assertEqual(i * 20, self.spawn([10, 10]))

    @defer.inlineCallbacks
    def test_evaluate(self):
        i = self.spawn(np.zeros(2))
        iSame = yield i.evaluate()
        self.assertEqual(i.SSE, 0)
        self.assertTrue(iSame is i)
        self.assertTrue(i.p.counter, 1)

    @defer.inlineCallbacks
    def test_nonzero(self):
        i = self.spawn(np.zeros(2))
        yield i.evaluate()
        self.assertTrue(i)
        i.SSE = np.inf
        self.assertTrue(i)
        i.SSE = -1
        self.assertFalse(i)

    @defer.inlineCallbacks
    def test_eq_and_hash(self):
        iList = []
        for k in range(2):
            i = self.spawn([1.1, 2.2])
            yield i.evaluate()
            iList.append(i)
        hashList = [hash(i) for i in iList]
        self.assertEqual(iList[0], iList[1])
        self.assertEqual(hashList[0], hashList[1])
        
        
        
