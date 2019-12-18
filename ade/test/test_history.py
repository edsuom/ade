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
from twisted.internet import defer

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

    def test_name2k_k2name(self):
        for k, name in enumerate(self.names):
            self.assertEqual(self.a.name2k(name), k+1)
            self.assertEqual(self.a.k2name(k+1), name)
    
    def test_valueVsSSE(self):
        XY = self.a.value_vs_SSE(['bar'])
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


class Test_ClosestPairFinder(tb.TestCase):
    def setUp(self):
        self.cpf = history.ClosestPairFinder(10, 4)

    def test_setRow(self):
        self.cpf.S = True # Just not None, for testing
        for k in range(10):
            Z = [10.0+k] + [k,k+1,k+2]
            self.cpf.setRow(k, Z)
        self.assertEqual(self.cpf.S, None)
        self.assertItemsEqual(self.cpf.X[0,:], [10.0, 0, 1, 2])

    def test_clearRow(self):
        self.cpf.setRow(0, [100.0, 2, 3, 4])
        self.cpf.S = True # Just not None, for testing
        self.cpf.clearRow(0)
        self.assertEqual(self.cpf.S, None)
        self.assertEqual(len(self.cpf.K), 0)

    def test_pairs_sampled(self):
        self.cpf.K = {3, 1, 4, 5, 9, 2}
        for N in (2, 3, 4):
            KP = self.cpf.pairs_sampled(N)
            self.assertEqual(KP.shape, (N, 2))
            for k1, k2 in KP:
                self.assertGreater(k2, k1)
                self.assertGreater(k1, 0)
                self.assertGreater(k2, 0)
                self.assertLess(k1, 10)
                self.assertLess(k2, 10)
        
    def test_pairs_all(self):
        self.cpf.K = {3, 1, 4, 5, 9, 2}
        N = len(self.cpf.K)
        Np = N*(N-1)/2
        KP = self.cpf.pairs_all()
        self.assertEqual(KP.shape, (Np, 2))
        for k1, k2 in KP:
            self.assertGreater(k2, k1)
            self.assertGreater(k1, 0)
            self.assertGreater(k2, 0)
            self.assertLess(k1, 10)
            self.assertLess(k2, 10)

    @defer.inlineCallbacks
    def test_diffs(self):
        self.cpf.setRow(0, [ 90.0, 0.11, 0.2, 0.3])
        self.cpf.setRow(1, [ 90.0, 0.09, 0.2, 0.3])
        self.cpf.setRow(2, [100.0, 0.09, 0.2, 0.3])
        self.cpf.setRow(3, [110.0, 0.11, 0.2, 0.3])
        self.cpf.setRow(4, [110.0, 0.10, 0.2, 0.3])
        self.assertEqual(self.cpf.S, None)
        K = np.array([[0, 1], [0, 2], [0, 3], [2, 3], [3, 4]])
        D = yield self.cpf(K=K)
        self.assertEqual(self.cpf.S.shape, (4,))
        s0 = 1.0/np.var([90., 90., 100., 110., 110.])
        self.assertAlmostEqual(self.cpf.S[0], s0)
        s1 = 1.0/np.var([0.11, 0.09, 0.09, 0.11, 0.10])
        self.assertAlmostEqual(self.cpf.S[1], s1)
        #        0-1   0-2    0-3    2-3    3-4
        SSEs = [90.0, 95.0, 100.0, 105.0, 110.0]
        for k, de in enumerate(
                [s1*0.02**2,                    # 0, 1
                 s0*10.0**2 + s1*0.02**2,       # 0, 2
                 s0*20.0**2,                    # 0, 3
                 s0*10.0**2 + s1*0.02**2,       # 2, 3
                 s1*0.01**2                     # 3, 4
                ]):
            #print k, D[k], de/np.sqrt(SSEs[k]),
            self.assertWithinOnePercent(D[k], de/SSEs[k])

    @defer.inlineCallbacks
    def test_diffs_someNeverpops(self):
        self.cpf.setRow(0, [100.0, 0.1130, 0.10, 0.100], 1)
        self.cpf.setRow(1, [100.0, 0.1010, 0.11, 0.100], 1)
        self.cpf.setRow(2, [100.0, 0.0940, 0.10, 0.100], 0)
        self.cpf.setRow(3, [100.0, 0.0957, 0.10, 0.099], 1)
        self.cpf.setRow(4, [100.0, 0.1100, 0.11, 0.100], 0)
        self.cpf.setRow(5, [100.0, 0.1100, 0.11, 0.110], 1)
        K = np.array([[0, 1], [0, 2], [2, 3]])
        D = yield self.cpf(K=K)
        # Kn = 4, N = 6
        Kn_penalty = 1 + np.exp(12*(4.0/6 - 0.4))
        penalty = [Kn_penalty if x else 1.0 for x in (1, 1, 0)]
        for k, p in enumerate(penalty):
            self.assertWithinTenPercent(D[k], 0.00120/p)
            
    @defer.inlineCallbacks
    def test_call(self):
        self.cpf.setRow(0, [ 90.0, 0.11, 0.2, 0.30])
        self.cpf.setRow(1, [ 90.0, 0.09, 0.2, 0.30])
        self.cpf.setRow(2, [100.0, 0.09, 0.2, 0.30])
        self.cpf.setRow(3, [110.0, 0.11, 0.2, 0.30])
        self.cpf.setRow(4, [110.0, 0.10, 0.2, 0.30])
        self.cpf.setRow(5, [140.0, 0.10, 0.2, 0.30])
        self.cpf.setRow(6, [140.0, 0.10, 0.2, 0.31])
        self.cpf.setRow(7, [140.1, 0.10, 0.2, 0.31])
        kr = yield self.cpf()
        self.assertEqual(kr, 6)
        self.cpf.clearRow(6)
        kr = yield self.cpf()
        self.assertEqual(kr, 1)
        self.cpf.clearRow(1)
        kr = yield self.cpf()
        self.assertEqual(kr, 0)


class Test_History(tb.TestCase):
    def setUp(self):
        self.names = ['foo', 'bar', 'zebra']
        self.h = history.History(self.names, N_max=10)

    def tearDown(self):
        return self.h.shutdown()
        
    @defer.inlineCallbacks
    def test_add_worsening(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 + k
            yield self.h.add(i)
            self.assertEqual(len(self.h), k+1)
            self.assertItemsEqual(self.h[k], [i.SSE] + i.values)
        for k, values in enumerate(self.h):
            self.assertItemsEqual(values, [k,k+1,k+2])

    @defer.inlineCallbacks
    def test_add_ignoreInfSSE(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 + k if k < 3 else float('+inf')
            kr = yield self.h.add(i)
            if k < 3:
                self.assertLess(kr, 10)
                self.assertEqual(len(self.h), k+1)
                self.assertItemsEqual(self.h[k], [i.SSE] + i.values)
            else:
                self.assertIs(kr, None)
                self.assertEqual(len(self.h), 3)
        for k, values in enumerate(self.h):
            self.assertItemsEqual(values, [k,k+1,k+2])
            
    @defer.inlineCallbacks
    def test_add_improving(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 - k
            yield self.h.add(i)
            self.assertEqual(len(self.h), k+1)
        for k, values in enumerate(self.h):
            self.assertItemsEqual(values, [4-k,4-k+1,4-k+2])
            
    @defer.inlineCallbacks
    def test_add_limitSize_worsening(self):
        krPopped = set()
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 + k
            yield self.h.add(i)
            if len(self.h.kr) == 10:
                iHash, kr = self.h.kr.popitem()
                self.h.notInPop(kr)
                krPopped.add(kr)
        self.assertEqual(len(self.h), 10)
        valuesPrev = None
        for values in self.h:
            if valuesPrev is not None:
                for v, vp in zip(values, valuesPrev):
                    self.assertGreater(v, vp)
            valuesPrev = values

    @defer.inlineCallbacks
    def test_add_limitSize_improving(self):
        krPopped = set()
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            yield self.h.add(i)
            if len(self.h.kr) == 10:
                iHash, kr = self.h.kr.popitem()
                self.h.notInPop(kr)
                krPopped.add(kr)
        self.assertEqual(len(self.h), 10)
        valuesPrev = None
        for values in self.h:
            if valuesPrev is not None:
                for v, vp in zip(values, valuesPrev):
                    self.assertLess(v, vp)
            valuesPrev = values

    @defer.inlineCallbacks
    def test_add_limitSize_improving_neverInPop(self):
        for k in range(15):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 1000.0 - k
            yield self.h.add(i, neverInPop=True)
        self.assertEqual(len(self.h), 10)
        self.assertEqual(len(self.h.Kp), 0)
        self.assertEqual(len(self.h.Kn), 10)
        valuesPrev = None
        for values in self.h:
            if valuesPrev is not None:
                for v, vp in zip(values, valuesPrev):
                    self.assertLess(v, vp)
            valuesPrev = values

    @defer.inlineCallbacks
    def test_add_then_purge(self):
        for k in range(5):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 100.0 - k
            yield self.h.add(i)
            self.assertEqual(len(self.h), k+1)
        self.assertEqual(len(self.h), 5)
        self.assertEqual(len(self.h.Kp), 5)
        self.h.purgePop()
        self.assertEqual(len(self.h), 0)
        self.assertEqual(len(self.h.Kp), 0)
            
    @defer.inlineCallbacks
    def test_value_vs_SSE(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            yield self.h.add(i)
        XY = yield self.h.value_vs_SSE(['bar'])
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(10.0, 19.0, 10))
        self.assertItemsEqual(XY[1], np.linspace(1.0, 10.0, 10))

    @defer.inlineCallbacks
    def test_value_vs_SSE_maxRatio(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            yield self.h.add(i)
        XY = yield self.h.value_vs_SSE(['bar'], maxRatio=1.5)
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(10.0, 15.0, 6))
        self.assertItemsEqual(XY[1], np.linspace(1.0, 6.0, 6))

    @defer.inlineCallbacks
    def test_value_vs_SSE_inPop(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            kr = yield self.h.add(i)
        self.h.notInPop(kr)
        XY = yield self.h.value_vs_SSE(['bar'], inPop=True)
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(10.0, 18.0, 9))
        self.assertItemsEqual(XY[1], np.linspace(1.0, 9.0, 9))

    @defer.inlineCallbacks
    def test_value_vs_SSE_notInPop(self):
        for k in range(10):
            i = tb.MockIndividual(values=[k,k+1,k+2])
            i.SSE = 10.0 + k
            kr = yield self.h.add(i)
            if k > 5: self.h.notInPop(kr)
        XY = yield self.h.value_vs_SSE(['bar'], notInPop=True)
        self.assertEqual(len(XY), 2)
        self.assertItemsEqual(XY[0], np.linspace(16.0, 19.0, 4))
        self.assertItemsEqual(XY[1], np.linspace(7.0, 10.0, 4))

    @defer.inlineCallbacks
    def test_pickle(self):
        def values(k):
            return [k, np.exp(-0.1*k), np.exp(-0.5*k)]
        
        for k in range(10):
            i = tb.MockIndividual(values=values(k))
            i.SSE = 1000.0+k
            yield self.h.add(i)
        s = pickle.dumps(self.h)
        h = pickle.loads(s)
        self.assertEqual(len(h), 10)
        for k, x in enumerate(h):
            sdiff = np.sum(np.square(x-values(k)))
            self.assertLess(sdiff, 1E-6)
                
            
        
