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

import time, os.path, random, pickle
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
        abort.restart()
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

        
class TestPopulation(tb.TestCase):
    Np = 20
    verbose = False
    
    def setUp(self):
        abort.restart()
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

    @defer.inlineCallbacks
    def test_pickle(self):
        yield self.p.setup()
        text = pickle.dumps(self.p)
        p = pickle.loads(text)
        self.assertEqual(repr(p), repr(self.p))

    @defer.inlineCallbacks
    def test_save_load(self):
        yield self.p.setup()
        fp = tb.fileInModuleDir("ade-test.dat", absolute=True, isTemp=True)
        self.p.save(fp)
        newBounds = [(-2, 2), (-2, 2)]
        p = population.Population.load(fp, func=tb.ackley, bounds=newBounds)
        for i in p:
            for value in i.values:
                self.assertLess(abs(value), 2)
                SSE = i.SSE
                i = yield i.evaluate()
                self.assertLessEqual(i.SSE, SSE)


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
