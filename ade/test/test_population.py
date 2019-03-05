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
Unit tests for L{ade.population}.
"""

import time, os.path, random
from StringIO import StringIO

import numpy as np
from matplotlib import pyplot as plt

from twisted.internet import defer, reactor

from ade.util import *
from ade import abort, population
from ade.individual import Individual

from ade.test import testbase as tb

#import twisted.internet.base
#twisted.internet.base.DelayedCall.debug = True


class TestConstraintChecking(tb.TestCase):
    def setUp(self):
        self.pm = population.ParameterManager(
            ['a', 'b'], [(-5, +5), (-5, +5)], )

    def tearDown(self):
        abort.shutdown()
        
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
        self.p = tb.MockPopulation(tb.ackley, ['x', 'y'], [(0,1)]*2, popsize=1)
        self.r = population.Reporter(self.p, self.processComplaint)
        self.r.addCallback(self.cbImmediate, 3.14, bar=9.87)
        return self.p.setup()

    def cbImmediate(self, values, counter, SSE, foo, bar=None):
        self.calls.append(['cbi', values, counter, SSE, foo, bar])
        if SSE == 0:
            # Complaint
            return 123
    
    def cbDeferred(self, values, counter, SSE):
        def doNow(null):
            self.calls.append(['cbd', values, counter, SSE])
        return self.deferToDelay(0.5).addCallback(doNow)

    def processComplaint(self, i, result):
        self.calls.append(['processComplaint', i, result])
    
    def test_runCallbacks_basic(self):
        i = self.p.spawn([1.0, 2.0])
        i.SSE = 0.876
        self.r.runCallbacks(i)
        self.assertEqual(
            self.calls,
            [['cbi', [1.0, 2.0], 0, 0.876, 3.14, 9.87]])
        return self.r.waitForCallbacks()

    def test_runCallbacks_complaint(self):
        i = self.p.spawn([0.0, 0.0])
        i.SSE = 0.0
        self.r.runCallbacks(i)
        self.assertEqual(
            self.calls,
            [['cbi', [0.0, 0.0], 0, 0.0, 3.14, 9.87],
             ['processComplaint', i, 123]])
        return self.r.waitForCallbacks()
    
    @defer.inlineCallbacks
    def test_runCallbacks_stacked(self):
        i1 = self.p.spawn([1, 2])
        i1.SSE = 0.876
        i2 = self.p.spawn([3, 4])
        i2.SSE = 0.543
        self.r.addCallback(self.cbDeferred)
        self.r.runCallbacks(i1)
        self.r.cbrScheduled(i2)
        self.assertEqual(self.calls, [
            ['cbi', [1, 2], 0, 0.876, 3.14, 9.87],
        ])
        yield self.r.waitForCallbacks()
        self.assertEqual(self.calls, [
            ['cbi', [1, 2], 0, 0.876, 3.14, 9.87],
            ['cbd', [1, 2], 0, 0.876],
            ['cbi', [3, 4], 0, 0.543, 3.14, 9.87],
            ['cbd', [3, 4], 0, 0.543],
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
    def test_msgRatio_nans(self):
        def reciprocal(x):
            if x[0] == 0: return np.nan
            return 1.0 / x[0]
        self.p = tb.MockPopulation(reciprocal, ['x'], [(-5, 5)], popsize=1)
        self.r = population.Reporter(self.p)
        fh = StringIO()
        msg(fh)
        iPrev = yield self.p.spawn([1.0]).evaluate()
        expectedRatios = [0, 0, 1, 2, 3]
        for k, x in enumerate([0, 1, 1.4, 2, 3.1]):
            # Closer to (0), thus higher SSE
            i = yield self.p.spawn(np.array([x])).evaluate()
            ratio = self.r.msgRatio(iPrev, i)
            self.assertAlmostEqual(ratio, expectedRatios[k])
        text = fh.getvalue()
        msg(None)
        self.assertEqual(text, "X0123")

    @defer.inlineCallbacks
    def test_fileReport_otherNone(self):
        fh = StringIO()
        msg(fh)
        # First -> best
        i1 = yield self.p.spawn(np.array([0.1, 0.1])).evaluate()
        ratio = yield self.r._fileReport(i1, None)
        self.assertEqual(self.r.iBest, i1)
        self.assertEqual(ratio, 0)
        self.assertEqual(fh.getvalue()[-1], "*")
        # Second, 4x worse
        i2 = yield self.p.spawn(np.array([1.005, 1.005])).evaluate()
        ratio = yield self.r._fileReport(i2, None)
        self.assertEqual(self.r.iBest, i1)
        self.assertEqual(ratio, 4)
        self.assertEqual(fh.getvalue()[-1], "4")
        # Force it to be best, for further testing
        self.r.iBest = i2
        # Third, slightly better than second
        i3 = yield self.p.spawn(np.array([1.00, 1.00])).evaluate()
        ratio = yield self.r._fileReport(i3, None)
        self.assertEqual(self.r.iBest, i3)
        self.assertEqual(ratio, 0)
        self.assertEqual(fh.getvalue()[-1], "!")
    
    @defer.inlineCallbacks
    def test_call_double(self):
        fh = StringIO()
        msg(fh)
        iPrev = yield self.p.spawn(np.array([1.005, 1.005])).evaluate()
        ratiosExpected = [0, 4, 3, 0, 10]
        for ratioExpected, x in zip(ratiosExpected, (1.0, 0.1, 0.05, 0.5, 0.06)):
            #print "\n", ratioExpected, x
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            ratio = yield self.r(i, iPrev)
            self.assertEqual(ratio, ratioExpected)
            iPrev = i
        self.assertEqual(fh.getvalue(), "0+4+3+X9")
        msg(None)

    @defer.inlineCallbacks
    def test_call_single(self):
        fh = StringIO()
        msg(fh)
        ratiosExpected = [0, 0, 0, 13, 1]
        for ratioExpected, x in zip(ratiosExpected, (1.0, 0.1, 0.05, 0.5, 0.06)):
            #print "\n", ratioExpected, x
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            ratio = yield self.r(i)
            self.assertEqual(ratio, ratioExpected)
            iPrev = i
        self.assertEqual(fh.getvalue(), "*!!91")
        msg(None)

        
class TestPopulation(tb.TestCase):
    Np = 20

    verbose = False
    
    def setUp(self):
        self.p = population.Population(
            tb.ackley, ["x", "y"], [(-5, 5), (-5, 5)], popsize=self.Np)

    def tearDown(self):
        abort.shutdown()
        
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
        self.assertAlmostEqual(self.p.statusQuoScore, self.p.Np*3.0/100)
        # 0 x 1
        self.p.replacement(0)
        self.assertEqual(self.p.replacementScore, 0)
        self.assertFalse(self.p.replacement())
        # 1 x 1
        self.p.replacement(1)
        self.assertEqual(self.p.replacementScore, 0.25)
        self.assertFalse(self.p.replacement())
        # 1 x 2
        self.p.replacement(1)
        self.p.replacement(1)
        self.assertEqual(self.p.replacementScore, 0.50)
        self.assertFalse(self.p.replacement())
        # 2 x 1
        self.p.replacement(2)
        self.assertEqual(self.p.replacementScore, 1.25)
        self.assertTrue(self.p.replacement())
        # 2 x 2
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


class TestPopulation_Abort(tb.TestCase):
    def setUp(self):
        self.p = population.Population(
            self.tenthSecond, ["x"], [(-5, 5)], popsize=100)

    def tenthSecond(self, x):
        return self.deferToDelay(0.1).addCallback(lambda _: 1.23)

    @defer.inlineCallbacks
    def test_setup_no_abort(self):
        t0 = time.time()
        yield self.p.setup()
        t1 = time.time()
        self.assertGreater(t1-t0, 0.1)
        self.assertLess(t1-t0, 0.13*self.p.N_maxParallel)
    
    @defer.inlineCallbacks
    def test_abort_during_setup(self):
        t0 = time.time()
        d = self.p.setup()
        self.deferToDelay(0.5).addCallback(lambda _: self.p.abort())
        yield d
        self.assertLess(time.time()-t0, 0.72)
