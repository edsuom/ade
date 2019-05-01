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


class TestAnalysis(tb.TestCase):
    def setUp(self):
        self.names = ['foo', 'bar', 'zebra']
        self.X = np.array([
            [1, 2, 5],
            [2, 3, 4],
            [3, 4, 3],
            [4, 5, 2],
            [5, 6, 1]
        ])
        self.SSEs = [
            110.0,      # 0
            120.0,      # 1
            129.0,      # 2
            140.0,      # 4
            150.0,      # 3
        ]
        self.K = [0, 1, 2, 4, 3]
        self.a = history.Analysis(self.names, self.X, self.K, self.SSEs)

    def test_valueVsSSE(self):
        XY = self.a.value_vs_SSE('bar')
        self.assertEqual(len(XY), 2)
        self.assertTrue(np.all(XY[0] == self.SSEs))
        self.assertTrue(np.all(XY[1] == [2, 3, 4, 6, 5]))

    def test_corr(self):
        self.assertAlmostEqual(self.a.corr(0, 1), +1)
        self.assertAlmostEqual(self.a.corr(0, 2), -1)
        
    def test_Kf12(self):
        K = self.a.Kf12(0.0, 0.5)
        self.assertTrue(np.all(K == [0, 1, 2]))

    def test_Kp12(self):
        K = self.a.Kp12(0.0, 0.5)
        self.assertTrue(np.all(K == [0, 1, 2]))
        K = self.a.Kp12(0.2, 0.7)
        self.assertTrue(np.all(K == [1, 2, 4]))
        K = self.a.Kp12(0.5, 1.0)
        self.assertTrue(np.all(K == [2, 4, 3]))


class TestHistory(tb.TestCase):
    def setUp(self):
        self.names = ['foo', 'bar', 'zebra']
        self.h = history.History(self.names, N_max=10)

    def test_isDuplicative_ofBest(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            self.h.add(i)
        SSE = i.SSE * 1.00001
        values = [x*1.00001 for x in i.values]
        self.assertTrue(self.h.isDuplicative(0, SSE, values))

    def test_isDuplicative_twoNeighbors(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            self.h.add(i)
            if k == 5: i5 = i
        SSE = i5.SSE * 1.00001
        values = [x*1.00001 for x in i5.values]
        for k in range(10):
            yes = self.h.isDuplicative(k, SSE, values)
            if k in (4, 5):
                self.assertTrue(yes)
            else: self.assertFalse(yes)
                
    def test_isDuplicative_ofWorst(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 + k
            self.h.add(i)
        SSE = i.SSE * 1.00001
        values = [x*1.00001 for x in i.values]
        self.assertFalse(self.h.isDuplicative(8, SSE, values))
        self.assertTrue(self.h.isDuplicative(9, SSE, values))

    def test_add(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 + k
            self.h.add(i)
            self.assertEqual(len(self.h), k+1)
            self.assertTrue(np.all(self.h[k] == i.values))

    def test_add_limitSize_improving(self):
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            self.h.add(i)
        self.assertEqual(len(self.h), 10)
        self.assertEqual(self.h[0][0], 14)
        self.assertEqual(self.h[1][0], 13)
        self.assertEqual(self.h[9][0], 5)

    def test_add_limitSize_worsening(self):
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 + k
            self.h.add(i)
        self.assertEqual(len(self.h), 10)
        self.assertEqual(self.h[0][0], 0)
        self.assertEqual(self.h[9][0], 9)

    def test_add_limitSize_duplicative(self):
        for k in range(21):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            self.h.add(i)
        self.assertEqual(self.h[0][0], 20)
        self.assertEqual(self.h[1][0], 19)
        i.SSE += 0.00001
        value = i.values[0]
        i.values[0] = value*1.00001
        self.h.add(i)
        self.assertEqual(self.h[0][0], 20)
        self.assertEqual(self.h[1][0], 19)
        i.values[0] = value*1.01
        self.h.add(i)
        self.assertEqual(self.h[0][0], 20.0)
        self.assertEqual(self.h[1][0], 19)
        i.values[0] = value*1.05
        self.h.add(i)
        self.assertAlmostEqual(self.h[1][0], value*1.05, 5)

    def test_add_limitSize_duplicativeButBest(self):
        for k in range(21):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            self.h.add(i)
        self.assertEqual(self.h[0][0], 20)
        self.assertEqual(self.h[1][0], 19)
        i.SSE -= 0.00001
        value = i.values[0]
        i.values[0] = value*1.00001
        self.h.add(i)
        self.assertAlmostEqual(self.h[0][0], value*1.00001, 5)
        self.assertEqual(self.h[1][0], 19)
        
    def test_valueVsSSE(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            self.h.add(i)
        XY = self.h.a.value_vs_SSE('bar')
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
        
        
