#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pingspice:
# Object-oriented circuit construction and efficient asynchronous
# simulation with Ngspice and twisted.
#
# Copyright (C) 2017 by Edwin A. Suominen,
# http://edsuom.com/pingspice
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
Unit testing for the B{pingspice} (easier to type out than
"PyNgspice") package by Edwin A. Suominen.

References with "§" are to subsections of the Ngspice manual, Version
26, by Paolo Nenzi and Holger Vogt.
"""

import time, os.path

import numpy as np

from twisted.internet import defer, reactor

from pingspice.ade.util import *
from pingspice.ade import individual

from pingspice.test import testbase as tb


# If verbose
msg(True)

        
class TestIndividual(tb.TestCase):
    verbose = False

    def setUp(self):
        bounds = [(-5, 5), (-5, 5)]
        self.p = tb.MockPopulation(tb.ackley, bounds, ["x", "y"])

    def test_init_with_values(self):
        values = np.array([0.5, 0.5])
        i = individual.Individual(self.p, values)
        i.SSE = 123.456
        self.assertItemsAlmostEqual(list(i), [0.5, 0.5])
        text = repr(i)
        self.assertIn('SSE=123.46', text)
        if self.verbose:
            print text
        
    def test_getAndSet(self):
        i = individual.Individual(self.p)
        i[1] = 3.14
        self.assertEqual(i[1], 3.14)

    def test_iterate(self):
        values = [10, 20]
        i = individual.Individual(self.p)
        for k, x in enumerate(values):
            i[k] = x
        count = 0
        for xi, xe in zip(i, values):
            self.assertEqual(xi, xe)
            count += 1
        self.assertEqual(count, 2)
        
    def test_subtract(self):
        i = individual.Individual(self.p, 0.5*np.ones(2))
        iDiff = i - i
        self.assertEqual(iDiff, np.zeros(2))
        iDiff = i - iDiff
        self.assertEqual(iDiff, i)
        
    def test_add(self):
        i = individual.Individual(self.p, 0.5*np.ones(2))
        iSum = i + i
        self.assertEqual(iSum, np.ones(2))
        
    def test_multiply(self):
        i = individual.Individual(self.p, 0.5*np.ones(2))
        self.assertEqual(i * 0, [0, 0])
        i = individual.Individual(self.p, 0.5*np.ones(2))
        self.assertEqual(i * 1, [0.5, 0.5])
        i = individual.Individual(self.p, 0.5*np.ones(2))
        self.assertEqual(i * 20, [10, 10])

    def test_pv(self):
        i = individual.Individual(self.p, np.zeros(2))
        for k in range(100):
            self.assertTrue(i.parameterLottery())
        i = individual.Individual(self.p, 10*np.ones(2))
        for k in range(100):
            self.assertFalse(i.parameterLottery())
        
    @defer.inlineCallbacks
    def test_evaluate(self):
        i = individual.Individual(self.p, np.zeros(2))
        iSame = yield i.evaluate()
        self.assertEqual(i.SSE, 0)
        self.assertTrue(iSame is i)
        self.assertTrue(i.p.counter, 1)

