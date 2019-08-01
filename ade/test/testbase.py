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
Utility stuff used by L{ade} unit tests.
"""

import os.path, inspect, re, atexit
from contextlib import contextmanager

import numpy as np

from twisted.internet import reactor, defer, task, threads
from twisted.trial import unittest

from asynqueue.util import DeferredTracker

from ade.util import sub, msg, oops


VERBOSE = False


class Bogus:
    pass


def deleteIfExists(fileNameOrPath):
    def tryDelete(fp):
        if os.path.exists(fp):
            os.remove(fp)
            return True
        return False
    if not tryDelete(fileNameOrPath):
        tryDelete(fileInModuleDir(fileNameOrPath))

def tempFiles(*args):
    for fileName in args:
        atexit.register(deleteIfExists, fileName)
    return args

def moduleDir(absolute=False, klass=None):
    if klass is None:
        klass = Bogus
    modulePath = inspect.getfile(klass)
    if absolute:
        modulePath = os.path.abspath(modulePath)
    return os.path.dirname(modulePath)

def fileInModuleDir(fileNameOrPath, absolute=False, isTemp=False, klass=None):
    filePath = os.path.normpath(
        os.path.join(moduleDir(absolute, klass), fileNameOrPath))
    if isTemp:
        tempFiles(filePath)
    return filePath


def realAckley(X):
    """
    The Ackley function. Limits: (-5, +5). f(0, ..., 0) = 0
    """
    z =  -20 * np.exp(-0.2 * np.sqrt(0.5*(np.sum(np.square(X)))))
    z += -np.exp(0.5*(np.sum(np.cos(2*np.pi*X))))
    return z + np.e + 20

EVAL_COUNT = [0]
def ackley(X):
    def done(result):
        EVAL_COUNT[0] += 1
        return result
    return threads.deferToThread(realAckley, X).addCallbacks(done, oops)

def evals_reset():
    EVAL_COUNT[0] = 0
def evals():
    return EVAL_COUNT[0]


class MsgBase(object):
    """
    A mixin for providing a convenient message method.
    """
    def isVerbose(self):
        if hasattr(self, 'verbose'):
            return self.verbose
        if 'VERBOSE' in globals():
            return VERBOSE
        return False
    
    def verboserator(self):
        if self.isVerbose():
            yield None

    def msg(self, proto, *args):
        for null in self.verboserator():
            if not hasattr(self, 'msgAlready'):
                proto = "\n" + proto
                self.msgAlready = True
            if args and args[-1] == "-":
                args = args[:-1]
                proto += "\n{}".format("-"*40)
            print(proto.format(*args))
            

class MockIndividual(object):
    class PlaceHolder:
        pass
    
    def __init__(self, p=None, values=None):
        if p is None:
            p = self.PlaceHolder()
            if values: p.Nd = len(values)
        self.p = p
        if values is None:
            self.values = np.empty(p.Nd)
        elif len(values) != p.Nd:
            raise ValueError("Expected {:d} values".format(self.p.Nd))
        self.values = values
        self.SSE = None
        self.partial_SSE = False

    def __repr__(self):
        return str(self.SSE) + ": " + str(self.values)
        
    def spawn(self, values):
        return MockIndividual(self.p, values)

    def copy(self):
        i = MockIndividual(self.p, list(self.values))
        i.SSE = self.SSE
        i.partial_SSE = self.partial_SSE
        return i

    def __sub__(self, other):
        return MockIndividual(self.values - other.values)

    def __add__(self, other):
        return MockIndividual(self.values + other.values)

    def __nonzero__(self):
        if self.SSE is None or np.isnan(self.SSE) or self.SSE >= 0:
            return True
        return False

    def __hash__(self):
        return hash(bytes(self.SSE) + np.array(self.values).tobytes())
    
    def __eq__(self, other):
        return self.SSE == other.SSE
    
    def equals(self, other):
        if hasattr(other, 'values'):
            other = other.values
        if not hasattr(other, 'shape'):
            other = np.array(other)
        return np.all(self.values == other)

    def __lt__(self, other):
        return self.SSE is not None and self.SSE < other.SSE

    def __gt__(self, other):
        return self.SSE is None or self.SSE > other.SSE
    
    def __getitem__(self, k):
        return self.values[k]

    def __setitem__(self, k, value):
        self.values[k] = value
            
    def __iter__(self):
        for value in self.values:
            yield value

    def __len__(self):
        return len(self.values)

    def limit(self):
        np.clip(self.values, self.p.pm.mins, self.p.pm.maxs, self.values)
        return self

    def blacklist(self):
        pass
    
    def evaluate(self):
        def done(SSE):
            self.SSE = SSE
            return self
        return self.p.evalFunc(self.values).addCallback(done)


class MockParameterManager(object):
    maxLineLength = 100
    
    def fromUnity(self, values):
        return -5 + 10*values
    
    def pv(self, values):
        return 1.0

    def lottery(self, values):
        return np.all(np.abs(values < 5))

    def prettyValues(self, values, prelude=None):
        lineParts = [prelude] if prelude else []
        for k, value in enumerate(values):
            name = "p{:d}".format(k)
            lineParts.append("{}={:g}".format(name, value))
        text = " ".join(lineParts)
        return text

    
class MockPopulation(object):
    debug = False
    
    def __init__(self, func, names, bounds, constraints=[], popsize=10):
        def evalFunc(values):
            return defer.maybeDeferred(func, values)

        self.Nd = len(bounds)
        self.Np = self.Nd * popsize
        self.pm = MockParameterManager()
        if not callable(func):
            raise ValueError(sub("Object '{}' is not callable", func))
        self.evalFunc = evalFunc
        self.counter = 0
        self.running = None

    def abort(self):
        self.running = False
        
    @property
    def kBest(self):
        return np.argmin([i.SSE for i in self.iList])
    
    def __getitem__(self, k):
        return self.iList[k]
        
    def __setitem__(self, k, i):
        self.iList[k] = i

    def spawn(self, values):
        return MockIndividual(self, values)
        
    @defer.inlineCallbacks
    def setup(self, uniform=False):
        XY = np.random.rand(self.Np, self.Nd) + 1E-2
        i = MockIndividual(self, np.array([2.0, 2.0]))
        yield i.evaluate()
        self.iList = [i]
        for xy in XY:
            i = MockIndividual(self, xy)
            yield i.evaluate()
            self.iList.append(i)
        self.running = True

    def replacement(self, *args, **kw):
        return True
            
    def push(self, i):
        self.iList.append(i)
        self.iList.pop(0)

    def best(self):
        return self.iList[self.kBest]

            
class TestCase(MsgBase, unittest.TestCase):
    """
    Slightly improved TestCase
    """
    # Nothing should take longer than 10 seconds, and often problems
    # aren't apparent until the timeout stops the test.
    timeout = 10

    def __init__(self, *args, **kw):
        if getattr(self, 'verbose', False) or globals().get('VERBOSE', False):
            msg(True)
        self.pendingCalls = {}
        super(TestCase, self).__init__(*args, **kw)

    def tearDown(self):
        msg(None)
        while self.pendingCalls:
            call = self.pendingCalls.keys()[0]
            if call.active():
                self.pendingCalls[call].callback(None)
                call.cancel()
        msg(None)
        
    def oops(self, failureObj):
        print "FAIL!!!!!!!\n{}\n{}".format('-'*40, failureObj.value)
        import pdb; pdb.set_trace()

    def deferToDelay(self, t):
        def delayOver(null):
            self.pendingCalls.pop(call, None)
            
        d = defer.Deferred().addCallbacks(delayOver, self.oops)
        call = reactor.callLater(t, d.callback, None)
        self.pendingCalls[call] = d
        return d
        
    def doCleanups(self):
        if hasattr(self, 'msgAlready'):
            del self.msgAlready
        return super(TestCase, self).doCleanups()

    def plot(self, ph):
        from matplotlib import pyplot
        X = [i.values[0] for i in ph.p]
        Y = [i.values[1] for i in ph.p]
        pyplot.plot(X, Y, '-o')
        pyplot.plot(X[-1], Y[-1], 'xr')
        pyplot.show()

    def multiplerator(self, N, expected):
        def check(null):
            self.assertEqual(resultList, expected)
            del self.d
        
        dList = []
        resultList = []
        for k in xrange(N):
            yield k
            self.d.addCallback(resultList.append)
            dList.append(self.d)
        self.dm = defer.DeferredList(dList).addCallback(check)
            
    def checkOccurrences(self, pattern, text, number):
        occurrences = len(re.findall(pattern, text))
        if occurrences != number:
            info = \
                u"Expected {:d} occurrences, not {:d}, " +\
                u"of '{}' in\n-----\n{}\n-----\n"
            info = info.format(number, occurrences, pattern, text)
            self.assertEqual(occurrences, number, info)
    
    def checkBegins(self, pattern, text):
        pattern = r"^\s*%s" % (pattern,)
        self.assertTrue(bool(re.match(pattern, text)))

    def checkProducesFile(self, fileName, executable, *args, **kw):
        producedFile = fileInModuleDir(fileName)
        if os.path.exists(producedFile):
            os.remove(producedFile)
        result = executable(*args, **kw)
        self.assertTrue(
            os.path.exists(producedFile),
            "No file '{}' was produced.".format(
                producedFile))
        os.remove(producedFile)
        return result

    def runerator(self, executable, *args, **kw):
        return Runerator(self, executable, *args, **kw)

    def assertPattern(self, pattern, text):
        proto = "Pattern '{}' not in '{}'"
        if '\n' not in pattern:
            text = re.sub(r'\s*\n\s*', '', text)
        if isinstance(text, unicode):
            # What a pain unicode is...
            proto = unicode(proto)
        self.assertTrue(
            bool(re.search(pattern, text)),
            proto.format(pattern, text))

    def assertStringsEqual(self, a, b, msg=""):
        N_seg = 20
        def segment(x):
            k0 = max([0, k-N_seg])
            k1 = min([k+N_seg, len(x)])
            return "{}-!{}!-{}".format(x[k0:k], x[k], x[k+1:k1])
        
        for k, char in enumerate(a):
            if char != b[k]:
                s1 = segment(a)
                s2 = segment(b)
                msg += "\nFrom #1: '{}'\nFrom #2: '{}'".format(s1, s2)
                self.fail(msg)

    def assertItemsAlmostEqual(self, a, b, tol=1):
        for ak, bk in zip(a, b):
            self.assertAlmostEqual(ak, bk, tol)
    
    def assertWithinOnePercent(self, x, xExpected):
        ratio = x / xExpected
        msg = sub("{} not within 1% of {}", x, xExpected)
        self.assertAlmostEqual(ratio, 1, 2, msg)

    def assertWithinFivePercent(self, x, xExpected):
        ratio = x / xExpected
        msg = sub("{} not within 5% of {}", x, xExpected)
        self.assertAlmostEqual(0.2*ratio, 0.2, 2, msg)

    def assertWithinTenPercent(self, x, xExpected):
        ratio = x / xExpected
        msg = sub("{} not within 10% of {}", x, xExpected)
        self.assertAlmostEqual(0.1*ratio, 0.1, 2, msg)
        
    def assertBetween(self, x, xMin, xMax):
        self.assertGreaterEqual(x, xMin)
        self.assertLessEqual(x, xMax)

