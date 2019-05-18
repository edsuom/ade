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

from StringIO import StringIO

import numpy as np

from twisted.internet import defer

from ade.util import *
from ade import report

from ade.test import testbase as tb

#import twisted.internet.base
#twisted.internet.base.DelayedCall.debug = True


class TestReporter(tb.TestCase):
    def setUp(self):
        self.calls = []
        self.p = tb.MockPopulation(tb.ackley, ['x', 'y'], [(0,1)]*2, popsize=1)
        self.r = report.Reporter(self.p, self.processComplaint)
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
            # Closer to (0,0) means lower SSE
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            ratio = self.r.msgRatio(iPrev, i)
            self.assertEqual(ratio, expectedRatios[k])
            ratio = self.r.msgRatio(i, iPrev)
            self.assertEqual(ratio, None)
            iPrev = i
        text = fh.getvalue()
        msg(None)
        self.assertEqual(text, "0X2X9X3X")

    @defer.inlineCallbacks
    def test_msgRatio_nan(self):
        def reciprocal(x):
            if x[0] == 0: return np.nan
            return 1.0 / x[0]
        self.p = tb.MockPopulation(reciprocal, ['x'], [(-5, 5)], popsize=1)
        self.r = report.Reporter(self.p)
        fh = StringIO()
        msg(fh)
        iPrev = yield self.p.spawn([1.0]).evaluate()
        expectedRatios = [0, 1000, 1, 2, 0, 0, 1000, 3]
        for k, x in enumerate([0, 1, 1.4, 3, 0, 0, 1, 3.1]):
            # Closer to (0), thus higher SSE
            i = yield self.p.spawn(np.array([x])).evaluate()
            ratio = self.r.msgRatio(iPrev, i)
            self.assertAlmostEqual(ratio, expectedRatios[k])
            iPrev = i
        text = fh.getvalue()
        msg(None)
        self.assertEqual(text, "#!12#%!3")

    @defer.inlineCallbacks
    def test_msgRatio_inf(self):
        def reciprocal(x):
            if x[0] == 0: return float('+inf')
            return 1.0 / x[0]
        self.p = tb.MockPopulation(reciprocal, ['x'], [(-5, 5)], popsize=1)
        self.r = report.Reporter(self.p)
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
        self.assertEqual(text, "#0123")
        
    @defer.inlineCallbacks
    def test_call_otherNone(self):
        fh = StringIO()
        msg(fh)
        # First -> best
        i1 = yield self.p.spawn(np.array([0.1, 0.1])).evaluate()
        self.r(i1, None)
        self.assertEqual(self.r.iBest, i1)
        self.assertEqual(fh.getvalue()[-1], "*")
        # Second, 4x worse
        i2 = yield self.p.spawn(np.array([1.005, 1.005])).evaluate()
        self.r(i2, None)
        self.assertEqual(self.r.iBest, i1)
        self.assertEqual(fh.getvalue()[-1], "4")
        # Force it to be best, for further testing
        self.r.iBest = i2
        # Third, slightly better than second
        i3 = yield self.p.spawn(np.array([1.00, 1.00])).evaluate()
        self.r(i3, None)
        self.assertEqual(self.r.iBest, i3)
        self.assertEqual(fh.getvalue()[-1], "!")
    
    @defer.inlineCallbacks
    def test_call_double(self):
        fh = StringIO()
        msg(fh)
        iPrev = yield self.p.spawn(np.array([1.005, 1.005])).evaluate()
        ratiosExpected = [0, 4, 3, None, 10]
        for ratioExpected, x in zip(ratiosExpected, (1.0, 0.1, 0.05, 0.5, 0.06)):
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            ratio = self.r(i, iPrev)
            self.assertEqual(ratio, ratioExpected)
            iPrev = i
        self.assertEqual(fh.getvalue(), "0+4+3+X9")
        msg(None)

    @defer.inlineCallbacks
    def test_call_single(self):
        fh = StringIO()
        msg(fh)
        for x in (1.0, 0.1, 0.05, 0.5, 0.06):
            i = yield self.p.spawn(np.array([x, x])).evaluate()
            self.r(i)
            if x == 0.05: iBest = i
        self.assertEqual(fh.getvalue(), "*!!91")
        self.assertEqual(self.r.iBest, iBest)
        msg(None)

    @defer.inlineCallbacks
    def test_call_single_forced(self):
        i = yield self.p.spawn(np.array([1.0, 1.0])).evaluate()
        self.r(i)
        self.assertEqual(len(self.calls), 1)
        self.assertAlmostEqual(self.calls[0][3], 3.62538494)
        i = yield self.p.spawn(np.array([0.99999, 1.0])).evaluate()
        self.r(i)
        self.assertEqual(len(self.calls), 1)
        i = yield self.p.spawn(np.array([0.99998, 1.0])).evaluate()
        self.r(i, force=True)
        self.assertEqual(len(self.calls), 2)
        self.assertAlmostEqual(self.calls[1][3], 3.6253522)


        
        
