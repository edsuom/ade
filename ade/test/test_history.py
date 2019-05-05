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


class Test_Analysis(tb.TestCase):
    def setUp(self):
        self.names = ['foo', 'bar', 'zebra']
        self.X = np.array([
            [110.0, 1, 2, 5], #0
            [810.0, 2, 3, 4], #1
            [270.0, 3, 4, 3], #2
            [580.0, 4, 5, 2], #3
            [999.0, 5, 6, 1], #4
        ])
        self.K = [0, 3, 2, 1, 4]
        self.a = history.Analysis(self.names, self.X, self.K)

    def test_valueVsSSE(self):
        XY = self.a.value_vs_SSE('bar')
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], [110.0, 270.0, 580.0, 810.0, 999.0])
        self.assertItemsEqual(XY[1], [2, 4, 5, 3, 6])

    def test_corr(self):
        self.assertAlmostEqual(self.a.corr(1, 2), +1)
        self.assertAlmostEqual(self.a.corr(1, 3), -1)
        
    def test_Kf12(self):
        # 0.0           1.0
        K = self.a.Kf12(0.0, 1.0)
        self.assertItemsEqual(K, [0, 3, 2, 1])
        # 0.0   0.6     1.01
        K = self.a.Kf12(0.0, 0.6)
        self.assertItemsEqual(K, [0, 3, 2])
        K = self.a.Kf12(0.6, 1.01)
        self.assertItemsEqual(K, [1, 4])
        # 0.0   0.3     1.01
        K = self.a.Kf12(0.0, 0.3)
        self.assertItemsEqual(K, [0, 2])
        K = self.a.Kf12(0.3, 1.01)
        self.assertItemsEqual(K, [3, 1, 4])

    def test_Kp12(self):
        K = self.a.Kp12(0.0, 0.5)
        self.assertItemsEqual(K, [0, 3, 2])
        K = self.a.Kp12(0.2, 0.7)
        self.assertItemsEqual(K, [3, 2, 1])
        K = self.a.Kp12(0.5, 1.01)
        self.assertItemsEqual(K, [2, 1, 4])


class Test_History(tb.TestCase):
    def setUp(self):
        self.names = ['foo', 'bar', 'zebra']
        self.h = history.History(self.names, N_max=10)
        self.kr = {}

    def test_add_worsening(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 + k
            self.h.add(i)
            self.assertEqual(len(self.h), k+1)
            self.assertItemsEqual(self.h[k], i.values)
        for k, values in enumerate(self.h):
            self.assertItemsEqual(values, [k,k+1,k+2])

    def test_add_improving(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 - k
            self.h.add(i)
            self.assertEqual(len(self.h), k+1)
        for k, values in enumerate(self.h):
            self.assertItemsEqual(values, [4-k,4-k+1,4-k+2])
            
    def test_add_limitSize_worsening(self):
        krPopped = set()
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 + k
            self.kr[i] = self.h.add(i)
            if len(self.kr) == 10:
                i, kr = self.kr.popitem()
                self.h.notInPop(kr)
                krPopped.add(kr)
        self.assertEqual(len(self.h), 10)
        valuesPrev = None
        for values in self.h:
            if valuesPrev is not None:
                for v, vp in zip(values, valuesPrev):
                    self.assertGreater(v, vp)
            valuesPrev = values

    def test_add_limitSize_improving(self):
        krPopped = set()
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            self.kr[i] = self.h.add(i)
            if len(self.kr) == 10:
                i, kr = self.kr.popitem()
                self.h.notInPop(kr)
                krPopped.add(kr)
        self.assertEqual(len(self.h), 10)
        valuesPrev = None
        for values in self.h:
            if valuesPrev is not None:
                for v, vp in zip(values, valuesPrev):
                    self.assertLess(v, vp)
            valuesPrev = values

    def test_value_vs_SSE(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            self.h.add(i)
        XY = self.h.a.value_vs_SSE('bar')
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(10.0, 19.0, 10))
        self.assertItemsEqual(XY[1], np.linspace(1.0, 10.0, 10))

    def test_value_vs_SSE_maxRatio(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            self.h.add(i)
        XY = self.h.a.value_vs_SSE('bar', maxRatio=1.5)
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(10.0, 15.0, 6))
        self.assertItemsEqual(XY[1], np.linspace(1.0, 6.0, 6))

    def test_value_vs_SSE_inPop(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            kr = self.h.add(i)
        self.h.notInPop(kr)
        XY = self.h.a.value_vs_SSE('bar', inPop=True)
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(10.0, 18.0, 9))
        self.assertItemsEqual(XY[1], np.linspace(1.0, 9.0, 9))

    def test_value_vs_SSE_notInPop(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            kr = self.h.add(i)
            if k > 5: self.h.notInPop(kr)
        XY = self.h.a.value_vs_SSE('bar', notInPop=True)
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(16.0, 19.0, 4))
        self.assertItemsEqual(XY[1], np.linspace(7.0, 10.0, 4))

    def test_pickle(self):
        def values(k):
            return [k, np.exp(-0.1*k), np.exp(-0.5*k)]
        
        for k in range(10):
            i = tb.MockIndividual(values=values(k))
            i.SSE = 1000.0+k
            self.h.add(i)
        s = pickle.dumps(self.h)
        h = pickle.loads(s)
        self.assertEqual(len(h), 10)
        for k, x in enumerate(h):
            sdiff = np.sum(np.square(x-values(k)))
            self.assertLess(sdiff, 1E-6)
        
        
