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

import time, os.path

import numpy as np

from twisted.internet import defer, reactor

from ade.util import *
from ade.population import Population
from ade import de

from ade.test import testbase as tb


# If verbose
msg(True)


class TestFManager(tb.TestCase):
    def test_get_single(self):
        self.fm = de.FManager(0.6, 0.5, 100, True)
        self.assertEqual(self.fm.get(), 0.6)

    def checkVariate(self, F):
        Fk_prev = 0
        bottomTenth = F[0] + 0.1*(F[1]-F[0])
        hadOneBelowBottomTenth = False
        for k in range(100):
            Fk = self.fm.get()
            self.assertGreater(Fk, F[0])
            self.assertLess(Fk, F[1])
            self.assertNotEqual(Fk, Fk_prev)
            if Fk < bottomTenth:
                hadOneBelowBottomTenth = True
            Fk_prev = Fk
        self.assertTrue(hadOneBelowBottomTenth)
    
    def test_get_variate(self):
        F = (0.5, 0.9)
        self.fm = de.FManager(F, 0.5, 100, True)
        self.checkVariate(F)

    def adjust(self, fName, fCheck=None, k=0):
        F_prev = 1000
        count = 0
        while True:
            count += 1
            getattr(self.fm, fName)()
            F = self.fm.get()
            if hasattr(F, '__iter__'):
                if F[k] == F_prev:
                    break
                else: F_prev = F[k]
            else:
                if F == F_prev:
                    break
                F_prev = F
            if fCheck:
                fCheck(F)
        return count, F 
            
    def test_downUp_single(self):
        self.fm = de.FManager(0.6, 0.5, 100, True)
        count, F = self.adjust('down')
        F_critical = np.sqrt(0.75/100)
        self.assertGreater(F, F_critical)
        self.assertLess(F, 1.5*F_critical)
        self.assertGreater(count, 15)
        self.assertLess(count, 20)
        count, F = self.adjust('up')
        self.assertEqual(F, 0.6)
        self.assertGreater(count, 15)
        self.assertLess(count, 25)

    def test_F_reduced_if_adaptive(self):
        F_orig = [0.5, 1.0]
        self.fm = de.FManager(F_orig, 0.9, 100, True)
        for k in range(10):
            self.fm.down()
        F = [0.5*(self.fm.scaleLowest**10), 1.0]
        self.checkVariate(F)

    def test_F_reduced_until_limit(self):
        F_orig = [0.5, 1.0]
        self.fm = de.FManager(F_orig, 0.9, 100, True)
        for k in range(1000):
            self.fm.down()
            if self.fm.limited:
                break
        self.assertGreater(k, 10)
        self.assertLess(k, 500)
        self.assertAlmostEqual(self.fm.F[0], self.fm.criticalF, 2)
        
    def test_F_increased_if_adaptive(self):
        F = [0.1, 1.0]
        self.fm = de.FManager(F, 0.9, 100, True)
        for k in range(10):
            self.fm.down()
        for k in range(100):
            self.fm.up()
        self.checkVariate(F)
    
    def test_F_same_if_not_adaptive(self):
        F_orig = [0.5, 1.0]
        self.fm = de.FManager(F_orig, 0.9, 100, False)
        self.fm.down()
        self.checkVariate(F_orig)
            
                
class TestDifferentialEvolution(tb.TestCase):
    @defer.inlineCallbacks
    def makeDE(self, Np, Nd):
        self.p = Population(
            tb.ackley,
            [sub("x{:d}", k) for k in range(Nd)],
            [(-5, 5)]*Nd, popsize=Np/Nd)
        yield self.p.setup()
        self.de = de.DifferentialEvolution(self.p, maxiter=35)

    @defer.inlineCallbacks
    def test_crossover(self):
        def doCrossovers(CR):
            N_allMutant = 0
            self.de.CR = CR
            for trial in range(N_runs):
                i1 = tb.MockIndividual(self.de.p, np.ones(Nd))
                self.de.crossover(i0, i1)
                self.assertNotEqual(i1, i0)
                if i1.equals(np.ones(Nd)):
                    N_allMutant += 1
            return float(N_allMutant)/N_runs

        N_runs = 1000
        Np = 100; Nd = 4
        yield self.makeDE(Np, Nd)
        i0 = tb.MockIndividual(self.p, np.zeros(Nd))
        self.assertEqual(doCrossovers(0), 0.0)
        self.assertEqual(doCrossovers(1.0), 1.0)
        # 1    2    3    4  
        # 1.0  0.5  0.5  0.5  x = 0.125
        self.assertAlmostEqual(doCrossovers(0.5), 0.125, 1)
        # 1    2    3    4  
        # 1.0  0.8  0.8  0.8  x = 0.729
        self.assertAlmostEqual(doCrossovers(0.9), 0.729, 1)
        
    @defer.inlineCallbacks
    def test_challenge(self):
        Np = 100; Nd = 5
        yield self.makeDE(Np, Nd)
        from ade import individual
        for k in (10,11):
            i = individual.Individual(self.p, np.zeros(Nd))
            i.SSE = 0.00001*k
            if k == 10:
                i10 = i
            self.de.p[k] = i
        yield self.de.challenge(10, 11)
        self.assertEqual(self.p.kBest, 10)
        self.assertEqual(self.p.individuals(10), i10)

    @defer.inlineCallbacks
    def test_call(self):
        yield self.makeDE(20, 2)
        p = yield self.de()
        x = p.best()
        self.assertAlmostEqual(x[0], 0, 4)
        self.assertAlmostEqual(x[1], 0, 4)
