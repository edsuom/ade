#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pingspice:
# Object-oriented circuit construction and efficient asynchronous
# simulation with Ngspice and twisted.
#
# Copyright (C) 2017 by Edwin A. Suominen,
# http://edsuom.com/pingspice
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
Unit testing for the B{pingspice} (easier to type out than
"PyNgspice") package by Edwin A. Suominen.

References with "§" are to subsections of the Ngspice manual, Version
26, by Paolo Nenzi and Holger Vogt.
"""

import os.path, inspect, re, atexit
from contextlib import contextmanager

import numpy as np

from twisted.internet import reactor, defer, task
from twisted.trial import unittest

from pingspice.util import sub, msg, deferToThread


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

def ackley(X, xSSE):
    return deferToThread(realAckley, X)
        

class RC_Mixin:
    _time = 0.01

    def fudge(self, R, C):
         # Due to R2, L2
        scale = 1.0 if R >= 1.0 else 0.95
        if C == 47000E-6:
            scale *= 1.06
        else:
            scale *= 0.97
        return scale

    def makeNetlist(self):
        filePath = fileInModuleDir('rc.cir', absolute=True)
        with open(filePath) as fh:
            self.netlist = fh.read()

    def makeAV(self):
        from pingspice.av import AV_Manager, AV_Maker
        if not hasattr(self, 'avm'):
            self.avm = AV_Manager()
            self.avList = self.avm.avList
            self.f = AV_Maker(self.avm)
        self.ID = getattr(self, 'ID', -1) + 1
        self.avm.setSetupID(self.ID)
        av = self.f.av('res', 5, 20)
        av.dev = 'R1'
        return self.ID

    @defer.inlineCallbacks
    def addSetup(self, tStep, tStop, mr=None, maxWait=None, uic=False):
        self.makeAV()
        if mr is None:
            if not hasattr(self, 'mr'):
                self.mr = self.MultiRunner(self.avList, self.N_cores)
            mr = self.mr
        from pingspice.analysis.sim import TRAN
        analyzerDef = [TRAN, tStep, tStop]
        if uic:
            analyzerDef.append('uic')
        yield mr.addSetup(self.ID, ['time', 'V(2)'], analyzerDef, self.netlist)
        if maxWait:
            aList = []
            dq = mr.setups[self.ID].dq
            for k in range(mr.N_cores):
                analyzer = yield dq.get()
                analyzer.r.np.maxWait = maxWait
                aList.append(analyzer)
            for analyzer in aList:
                dq.put(analyzer)
        defer.returnValue(self.ID)
    
    def mrSetup(self, tStep, tStop, mr=None, maxWait=None):
        self.makeNetlist()
        return self.addSetup(tStep, tStop, mr=mr, maxWait=maxWait)
    
    def _voltage(self, t, R, C):
        return self.fudge(R, C) * 12.0 * (1.0 - np.exp(-t / (R * C)))
    
    def checkRiseTime(self, V, R, C=47000E-6):
        self.assertGreater(len(V.time), 50)
        k = np.searchsorted(V.time, self._time)
        t = V.time[k]
        v = V.v2[k]
        ratio = v / self._voltage(t, R, C)
        self.msg(
            "Capacitor reached {:f}V after {:f} sec (k={:d}/{:d})",
            v, t, k, len(V.time))
        self.msg(
            "Simulated vs expected (with {:f} fudge factor): {:f}",
            self.fudge(R, C), ratio)
        self.assertAlmostEqual(ratio, 1.0, 1)


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
            

class MockElements(MsgBase):
    def __init__(self, n):
        self.n = n
        self.calls = []
        self.alterables = []
    
    def mockCallable(self, *args, **kw):
        self.calls.append([self.name, args, kw])
    
    def __getattr__(self, name):
        self.name = name
        return self.mockCallable

    def av(self, *args):
        self.alterables.append(args)


class MockNetlist(MsgBase):
    circuitName = "pingspice-circuit"
    knowns = {}
    avm = None
    
    def __init__(self, *args, **options):
        self.calls = 0
        if args and isinstance(args[0], str):
            filePath = args[0]
            self.circuitName = os.path.splitext(os.path.split(filePath)[-1])[0]
            self.fh = open(filePath, 'w')
            args = args[1:]
        elif len(args) > 1:
            self.fh, self.circuitName = args[:2]
            args = args[2:]
        if args:
            # NO TYPE CHECKING IN MOCK VERSION
            self.avm = args[0]
        self.options = options
    
    @contextmanager
    def __call__(self, *args, **params):
        if self.calls == 0 and args:
            self.circuitName = args[0]
        self.calls += 1
        self.f = MockElements(self)
        yield self.f


class MockNgspiceRunner(MsgBase):
    _N = 100
    #          0       1       2      3             4
    _names = ['time', 'V(1)', 'neg', 'vcs#branch', 'V(2)']

    def __init__(self, maxWait=60, verbose=False, spew=False, console=False):
        self.maxWait = maxWait
        self.gotten = []
        self.calls = {}
        self.memUsed = 360000

    def shutdown(self):
        pass

    def deferToDelay(self, x=None):
        return defer.succeed(None)
    
    def memUsage(self):
        return defer.succeed(self.memUsed)
    
    def clearData(self):
        return defer.succeed(None)

    def source(self, *args, **kw):
        return defer.succeed(None)

    def setting(self, *args):
        return defer.succeed(None)
    
    def get(self, *names):
        result = []
        for name in names:
            if name not in self._names:
                raise Exception("BOGUS NAME '{}'".format(name))
            self.gotten.append(name)
            result.append(self._names.index(name)*np.ones(self._N))
        if len(result) == 1:
            result = result[0]
        return defer.succeed(result)

    def alter(self, *args):
        self.calls.setdefault('alter', []).append(args)
        return defer.succeed(None)

    def altermod(self, *args):
        self.calls.setdefault('altermod', []).append(args)
        return defer.succeed(None)

    def tran(self, *args):
        self.calls.setdefault('tran', []).append(args)
        result = (100, {}, 25.0)
        if len(args) > 2:
            return task.deferLater(reactor, 1.0, lambda : result)
        return defer.succeed(result)


class MockVector(MsgBase):
    def __init__(self, r, names):
        self.names = names
        self.values = {}
        self.independents = []

    def __contains__(self, name):
        return name in self.names
        
    def __getitem__(self, name):
        return self.values[name]

    def __getattr__(self, name):
        return self[name]
        
    def addName(self, name, independent=False):
        if independent and name not in self.independents:
            self.independents.append(name)
        if name not in self.names:
            self.names.append(name)
        return name

    def addVector(self, name, X, independent=False):
        self.addName(name, independent)
        self.values[name] = X

    def copy(self):
        V2 = MockVector(None, self.names)
        V2.values.update(self.values)
        V2.independents = self.independents
        return V2
        

class MockAV_Maker(MsgBase):
    def __init__(self, avManager=None, knowns={}):
        if avManager is None:
            avManager = MockAV_Manager(knowns)
        self.avm = avManager

    def av(self, name, *args):
        # Copied from av.py, r1634
        def makeChild():
            for avExisting in self.avm.namerator(name):
                if avExisting.isParent():
                    avExisting.adopt(thisAV)
                    return

        def makeParent():
            for avExisting in self.avm.namerator(name):
                if avExisting.isOrphan:
                    thisAV.adopt(avExisting)
            for avExisting in self.avm.namerator(name):
                if avExisting.isParent() and avExisting.vbRoot is None:
                    thisAV.vbRoot = avExisting.vb
                    break

        from pingspice.av import AV
        if not args:
            thisAV = AV(name)
            makeChild()
        else:
            thisAV = AV(name, *args)
            makeParent()
        self.avm.add(thisAV)
        return thisAV
        
        
class MockAV_Manager(MsgBase):
    def __init__(self, knowns={}):
        self.knowns = knowns
        self.avList = []
        self.ID = 0

    def namerator(self, name):
        for av in self.avList:
            if av.name == name:
                yield av

    def add(self, av):
        name = av.name
        if name in self.knowns:
            known = self.knowns[name]
            av.setValueBasis(known)
        av.associateWith(self.ID)
        self.avList.append(av)

    def setSetupID(self, ID):
        self.ID = ID

    def finalize(self):
        for av in self.avList:
            if av.parent:
                av = av.parent
            for avChild in av.children:
                if avChild not in self.avList:
                    self.avList.append(avChild)


class MockEvalSpec(object):
    def __init__(self, *args, **kw):
        self.nameList = ['foo', 'foo_squared']
        self.indieFlags = [True, False]

    def canInterpolate(self):
        self.kIndep = 0
        return True
        
        
class MockIndividual(object):
    def __init__(self, p, values=None):
        self.p = p
        if values is None:
            self.values = np.empty(p.Nd)
        elif len(values) != p.Nd:
            raise ValueError("Expected {:d} values".format(self.p.Nd))
        self.values = values
        self.SSE = None
        self.partial_SSE = False

    def __repr__(self):
        print self.SSE, self.values
        
    def spawn(self, values):
        return MockIndividual(self.p, values)

    def __sub__(self, other):
        return MockIndividual(self.values - other.values)

    def __add__(self, other):
        return MockIndividual(self.values + other.values)

    def __nonzero__(self):
        return bool(self.SSE)
    
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
    
    def evaluate(self, xSSE=None):
        def done(SSE):
            self.SSE = SSE
            self.partial_SSE = xSSE is not None
            return self
        return self.p.evalFunc(self.values, xSSE).addCallback(done)


class MockParameterManager(object):
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
    def __init__(self, func, bounds, names, constraints=[], popsize=10):
        def evalFunc(values, xSSE):
            return defer.maybeDeferred(func, values, xSSE)

        self.Nd = len(bounds)
        self.Np = self.Nd * popsize
        self.pm = MockParameterManager()
        if not callable(func):
            raise ValueError(sub("Object '{}' is not callable", func))
        self.evalFunc = evalFunc
        self.counter = 0

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
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)
        super(TestCase, self).__init__(*args, **kw)

    def shutdown(self):
        msg(None)
        if hasattr(self, 'triggerID'):
            reactor.removeSystemEventTrigger(self.triggerID)
            del self.triggerID
        for call in self.pendingCalls:
            if call.active():
                self.pendingCalls[call].callback(None)
                call.cancel()
        
    def oops(self, failureObj):
        print "FAIL!!!!!!!\n{}\n{}".format('-'*40, failureObj.value)
        import pdb; pdb.set_trace()

    def deferToDelay(self, x=None):
        def delayOver(null):
            self.pendingCalls.pop(call, None)
            
        if x is None:
            x = self.unknownResponseWaitTime
        d = defer.Deferred().addCallbacks(delayOver, self.oops)
        call = reactor.callLater(x, d.callback, None)
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
