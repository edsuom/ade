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

import time, os.path, random
from StringIO import StringIO

import numpy as np
from matplotlib import pyplot as plt

from twisted.internet import defer, reactor

from ade.util import *
from ade import population
from ade.individual import Individual

from ade.test import testbase as tb


class TestConstraintChecking(tb.TestCase):
    def setUp(self):
        self.pm = population.ParameterManager(
            ['a', 'b'], [(-5, +5), (-5, +5)], )
    
    def unitySumList(self, params):
        return sum(params.values()) == 1.0

    def proportionalDict(self, params):
        return params['a'] == 2*params['b']
    
    def test_call_no_constraints(self):
        self.assertTrue(self.pm.passesConstraints([0, 0]))
        self.assertTrue(self.pm.passesConstraints([1E20, 1E20]))
        
    def test_call(self):
        self.pm.constraints = [self.unitySumList]
        self.assertTrue(self.pm.passesConstraints([0.4, 0.6]))
        self.assertFalse(self.pm.passesConstraints([0.5, 0.6]))

    def test_call_with_names(self):
        self.pm.constraints = [self.proportionalDict]
        self.assertTrue(self.pm.passesConstraints([3.0, 1.5]))
        self.assertFalse(self.pm.passesConstraints([4.0, 1.5]))


class TestParameterManager(tb.TestCase):
    N_trials = 100000

    verbose = False

    def setUp(self):
        self.pm = population.ParameterManager([], [])
    
    def checkZeroPortion(self, N_zeros, expectedProportion):
        self.assertAlmostEqual(
            float(self.N_trials)/N_zeros, 1.0/expectedProportion, 1)

    def test_limit_distribution(self):
        N = 100000
        pm = population.ParameterManager(['x'], [(1, 2)])
        X = np.random.uniform(0, 3, N)
        Y = []
        for x in X:
            x = [x]
            y = pm.limit(x)[0]
            Y.append(y)
        plt.figure()
        bins = np.linspace(0, 3, 100)
        counts = plt.hist(Y, bins)[0]
        plt.title("Univariate limited i(1,2) distribution")
        #plt.xticks(range(0, 3, 12), fontsize=14)
        plt.grid()
        for k, count in enumerate(counts):
            if self.verbose:
                print sub(
                    "{:3d}  {:4.2f}-{:4.2f}  {:f}",
                    k, bins[k], bins[k+1], count)
                if bins[k+1] < 1 or bins[k] > 2:
                    self.assertEqual(count, 0)
                else:
                    self.assertGreater(count, 2900)
                    self.assertLess(count, 3100)
        if self.verbose:
            plt.show()


class TestReporter(tb.TestCase):
    def setUp(self):
        self.calls = []
        self.p = tb.MockPopulation(tb.ackley, [(0,1)]*2, ['x', 'y'], popsize=1)
        self.r = population.Reporter(self.p)
        self.r.addCallback(self.cbImmediate, 3.14, bar=9.87)

    def cbImmediate(self, values, counter, foo, bar=None):
        self.calls.append(['cbi', values, counter, foo, bar])

    def cbDeferred(self, values, counter):
        def doNow(null):
            self.calls.append(['cbd', values, counter])
        return self.deferToDelay(0.5).addCallback(doNow)
    
    def test_runCallbacks_basic(self):
        self.r.runCallbacks([1, 2, 3])
        self.assertEqual(
            self.calls,
            [['cbi', [1, 2, 3], 0, 3.14, 9.87]])
        return self.r.waitForCallbacks()
    
    @defer.inlineCallbacks
    def test_runCallbacks_stacked(self):
        self.r.addCallback(self.cbDeferred)
        self.r.runCallbacks([1, 2, 3])
        self.r.cbrScheduled([4, 5, 6])
        self.assertEqual(self.calls, [
            ['cbi', [1, 2, 3], 0, 3.14, 9.87],
        ])
        yield self.r.waitForCallbacks()
        self.assertEqual(self.calls, [
            ['cbi', [1, 2, 3], 0, 3.14, 9.87],
            ['cbd', [1, 2, 3], 0],
            ['cbi', [4, 5, 6], 0, 3.14, 9.87],
            ['cbd', [4, 5, 6], 0],
        ])

    @defer.inlineCallbacks
    def test_newBest_basic(self):
        self.r.addCallback(self.cbDeferred)
        yield self.p.setup(uniform=True)
        self.r.newBest(self.p.best())
        self.assertEqual(len(self.calls), 1)
        yield self.deferToDelay(0.6)
        self.assertEqual(len(self.calls), 2)

    @defer.inlineCallbacks
    def test_newBest_stacked(self):
        self.r.addCallback(self.cbDeferred)
        yield self.p.setup(uniform=True)
        self.r.newBest(self.p.best())
        for x in (1E-3, 1E-4, 1E-5):
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            self.r.newBest(i)
        # cbi0
        self.assertEqual(len(self.calls), 1)
        yield self.deferToDelay(0.6)
        # cbi0, cbd0, cbi1
        self.assertEqual(
            [x[0] for x in self.calls], ['cbi', 'cbd', 'cbi'])
        yield self.deferToDelay(0.5)
        # cbi0, cbd0, cbi1, cbd2
        self.assertEqual(
            [x[0] for x in self.calls], ['cbi', 'cbd', 'cbi', 'cbd'])
        yield self.deferToDelay(1.0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[-1][1][0], 1E-5)

    @defer.inlineCallbacks
    def test_msgRatio(self):
        fh = StringIO()
        msg(fh)
        iPrev = yield self.p.spawn(np.array([1.001E-3, 1.001E-3])).evaluate()
        expectedRatios = [0, 2, 17, 3]
        for k, x in enumerate((1E-3, 5E-4, 3E-5, 1E-5)):
            # Closer to (0,0), thus lower SSE
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            ratio = self.r.msgRatio(iPrev, i)
            self.assertAlmostEqual(ratio, expectedRatios[k])
            ratio = self.r.msgRatio(i, iPrev)
            self.assertEqual(ratio, 0)
            iPrev = i
        text = fh.getvalue()
        msg(None)
        self.assertEqual(text, "0X2X9X3X")

    @defer.inlineCallbacks
    def test_call(self):
        fh = StringIO()
        msg(fh)
        iPrev = self.p.spawn(np.array([1.005, 1.005]))
        mr = self.r(iPrev)
        self.assertEqual(mr, 0)
        yield iPrev.evaluate()
        mr = self.r(iPrev)
        self.assertEqual(mr, 0)
        mrExpected = [
            (0, "0"),
            (4, "4"),
            (11, "9"),
            (36, "9"),
        ]
        for mre, x in zip(mrExpected, (1.0, 0.1, 0.05, 0.02)):
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            mr = self.r(i)
            self.assertEqual(mr, mre[0])
            self.assertEqual(fh.getvalue()[-1], str(mre[1]))
        msg(None)

        
class TestPopulation(tb.TestCase):
    Np = 20

    verbose = False
    
    def setUp(self):
        self.p = population.Population(
            tb.ackley, ["x", "y"], [(-5, 5), (-5, 5)], popsize=self.Np)

    def positiveOnly(self, XY):
        return min(XY.values()) > 0

    def test_setup(self):
        def done(null):
            self.assertEqual(len(self.p), 2*self.Np)
            if self.verbose:
                self.plot(self)
            for k in (0, 1):
                values = [i.values[k] for i in self.p]
                self.assertGreater(min(values), -5)
                self.assertLess(max(values), 5)
        return self.p.setup().addCallback(done)
        
    def test_setup_constrained(self):
        def done(null):
            self.assertEqual(len(self.p), 2*self.Np)
            for i in self.p:
                self.assertTrue(np.all(i.values > 0))
            if self.verbose:
                self.plot(self)
        self.p = population.Population(
            tb.ackley, ["x", "y"], [(-5, 5), (-5, 5)],
            constraints=self.positiveOnly, popsize=self.Np)
        return self.p.setup().addCallback(done)

    def test_setup_constrained_uniform(self):
        def done(null):
            self.assertEqual(len(self.p), 2*self.Np)
            for i in self.p:
                self.assertTrue(np.all(i.values > 0))
            if self.verbose:
                self.plot(self)
        self.p = population.Population(
            tb.ackley, ["x", "y"], [(-5, 5), (-5, 5)],
            constraints=self.positiveOnly, popsize=self.Np)
        return self.p.setup(uniform=True).addCallback(done)

    def test_replacement(self):
        self.assertTrue(self.p.replacement())
        self.assertEqual(self.p.replacementScore, 0)
        self.assertFalse(self.p.replacement())
        self.assertAlmostEqual(self.p.statusQuoScore, self.p.Np*5.0/100)
        self.p.replacement(0)
        self.assertEqual(self.p.replacementScore, 0)
        self.assertFalse(self.p.replacement())
        self.p.replacement(1)
        self.assertEqual(self.p.replacementScore, 0.25)
        self.assertFalse(self.p.replacement())
        self.p.replacement(2)
        self.assertEqual(self.p.replacementScore, 1.25)
        self.assertFalse(self.p.replacement())
        self.p.replacement(2)
        self.p.replacement(2)
        self.assertEqual(self.p.replacementScore, 2.5)
        self.assertTrue(self.p.replacement())
        
    @defer.inlineCallbacks
    def test_push(self):
        yield self.p.setup()
        iList = list(self.p)
        iWorst = iList[np.argmax([i.SSE for i in iList])]
        iNew = Individual(self.p, np.array([1.23, 4.56]))
        iNew.SSE = 0
        self.p.push(iNew)
        self.assertIn(iNew, list(self.p))
        self.assertNotIn(iWorst, list(self.p))

    @defer.inlineCallbacks
    def test_push_and_best(self):
        SSE_prev = np.inf
        yield self.p.setup()
        for i in self.p:
            i.SSE = np.inf
        for k in range(100000):
            i = Individual(self.p, [0, 0])
            i.SSE = 1000*random.random()
            iBest = self.p.best()
            if i < iBest:
                self.p.push(i)
                if self.verbose:
                    print k, i.SSE
            self.assertTrue(i.SSE <= self.p.iSorted[-1].SSE)
            self.assertTrue(iBest.SSE <= SSE_prev)
            SSE_prev = i.SSE
    
    @defer.inlineCallbacks
    def test_lock(self):
        def gotLock(null):
            stuff.append(sum(stuff))

        stuff = [1]
        yield self.p.setup()
        yield self.p.lock(4, 5)
        d = self.p.lock(5).addCallback(gotLock)
        stuff.append(2)
        self.p.release(4, 5)
        yield d
        self.assertEqual(stuff, [1, 2, 3])
