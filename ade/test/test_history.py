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
Unit tests for L{ade.report}.
"""

import pickle

import numpy as np

from ade.util import *
from ade import history
from ade.test import testbase as tb


class TestHistory(tb.TestCase):
    def setUp(self):
        self.names = ['foo', 'bar', 'zebra']
        self.h = history.History(self.names, N_max=10)

    def test_add(self):
        i = tb.MockIndividual(values=[1,2,3])
        i.SSE = 101.0
        self.h.add(i)
        self.assertEqual(len(self.h), 1)
        self.assertTrue(np.all(self.h[0] == np.array([1,2,3])))
        i = tb.MockIndividual(values=[4,5,6])
        i.SSE = 102.0
        self.h.add(i)
        self.assertEqual(len(self.h), 2)
        self.assertTrue(np.all(self.h[1] == np.array([4,5,6])))

    def test_add_limitSize_improving(self):
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0-k
            self.h.add(i)
        self.assertEqual(len(self.h), 10)
        self.assertEqual(self.h[0][0], 14)
        self.assertEqual(self.h[9][0], 5)

    def test_add_limitSize_worsening(self):
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0+k
            self.h.add(i)
        self.assertEqual(len(self.h), 10)
        self.assertEqual(self.h[0][0], 0)
        self.assertEqual(self.h[9][0], 9)

    def test_valueVsSSE(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            self.h.add(i)
        XY = self.h.value_vs_SSE('bar')
        self.assertEqual(len(XY), 2)
        self.assertTrue(np.all(XY[0] == np.linspace(10.0, 19.0, 10)))
        self.assertTrue(np.all(XY[1] == np.linspace(1.0, 10.0, 10)))

    def test_pickle(self):
        def values(k):
            return [k, np.exp(-0.1*k), np.exp(-0.5*k)]
        
        for k in range(15):
            i = tb.MockIndividual(values=values(k))
            i.SSE = 1000.0+k
            self.h.add(i)
        s = pickle.dumps(self.h)
        h = pickle.loads(s)
        self.assertEqual(len(h), 10)
        for k, x in enumerate(h):
            sdiff = np.sum(np.square(x-values(k)))
            self.assertLess(sdiff, 1E-6)
        
        
