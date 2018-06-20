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


"""
A Population class and helpers.
"""

import random
from copy import copy
from textwrap import TextWrapper

import numpy as np
from scipy import stats

from pyDOE import lhs
from twisted.internet import defer

from asynqueue.util import DeferredTracker

from individual import Individual
from util import *


class ParameterManager(object):
    """
    """
    maxLineLength = 100
    dashes = "-"*maxLineLength

    def __init__(self, names, bounds, constraints=[]):
        if len(bounds) != len(names):
            raise ValueError(
                "Define one parameter name for each lower/upper bound")
        self.names = names
        self.sortedNameIndices = [
            names.index(name) for name in sorted(names)]
        self.mins = np.array([x[0] for x in bounds])
        self.maxs = np.array([x[1] for x in bounds])
        self.scales = self.maxs - self.mins
        self.mids = self.mins + 0.5 * self.scales
        self.constraints = constraints if hasattr(
            constraints, '__iter__') else [constraints]
        self.fill = TextWrapper(
            width=self.maxLineLength, break_on_hyphens=False).fill

    def prettyValues(self, values, *args):
        """
        Returns an easily readable string representation of the supplied
        I{values} with their parameter names, sorted.

        You can provide as an additional argument a prelude string, or
        a string proto with additional args, and the string will
        precede the values.
        """
        lineParts = []
        if args:
            lineParts.append(args[0].format(*args[1:]))
        for k, name, value in self.sortedNamerator(values):
            lineParts.append("{}={:g}".format(name, value))
        text = " ".join(lineParts)
        return self.fill(text)

    def sortedNamerator(self, values=None):
        """
        Generates tuples of sorted names with (1) the index in a
        I{values} list of parameters where each named parameter
        appears, and (2) the name itself.

        If such a list of I{values} is supplied, each tuple also
        includes (3) the value for that name.
        """
        if values is None:
            for k in self.sortedNameIndices:
                yield k, self.names[k]
        else:
            for k in self.sortedNameIndices:
                yield k, self.names[k], values[k]
        
    def scale(self, values, coeff=1.0):
        return coeff * self.scales * values
        
    def fromUnity(self, values):
        """
        Translates the normalized I{values} from the standardized range of
        0-1 into my range of actual parameter values within the ranges
        specified in the bounds supplied to my constructor.
        """
        return self.mins + self.scale(values)

    def passesConstraints(self, values):
        """
        Call with a 1-D array of parameter I{values} to check them against
        all of the constraints. Each callable in my I{constraints}
        list must return C{True} if it found the parameters (supplied
        to each callable as a dict) to be acceptable. The result will
        be C{True} if and only if all constraints were satisfied. (Or
        if you constructed me with an empty list.)
        """
        if not self.constraints: return True
        param = {}
        for name, value in zip(self.names, values):
            param[name] = value
        for func in self.constraints:
            if not func(param):
                # This constraint was violated, bail out
                return False
        return True

    def limit(self, values):
        """
        Limits the supplied I{values} to my boundaries using the
        simple and well-accepted "reflection" method.

        According to a study by Kreischer, Magalhaes, et
        al. ("Evaluation of Bound Constraints Handling Methods in
        Differential Evolution using the CEC2017 Benchmark"), this is
        second and performance only to resampling for a new DE
        mutant. (They also propose a "scaled mutant" method that is
        more complicated and would present huge obstacles to
        calculating L{prTransition}, but according to their Tables 1,
        2, doesn't appear significantly better.)
        """
        values = np.where(values < self.mins, 2*self.mins - values, values)
        values = np.where(values > self.maxs, 2*self.maxs - values, values)
        return np.clip(values, self.mins, self.maxs)


class Reporter(object):
    """
    I report on the SSE of individuals in an evolving population.

    Call an instance of me with an Individual to report on its SSE
    compared to the best one I've reported on thus far. A single
    numeric digit 1-9 will be logged indicating how much worse (higher
    SSE) the new individual is than the best one, or an "X" if the new
    individual becomes the best one. The integer ratio will be
    returned, or 0 if if the new individual became best.

    Call my instance with two Individuals to report on the SSE of the
    first one compared to the second. A single numeric digit 1-9 will
    be logged indicating how much B{better} (lower SSE) the first
    individual is than the second one, or an "X" if the first
    individual is actually worse. The integer ratio will be returned,
    or 0 if the first individual was worse.

    If you set the keyword I{ratioBetterThanBest} to C{True} in the
    call with just a single Individual, the ratio will be how much
    I{better} it is than the best one, and the symbol "!" will be
    logged if it was actually worse. This is useful for non-greedy
    algorithms like simulated annealing, where a worse individual is
    sometimes accepted.

    In either case, whenever the first- or only-supplied Individual is
    better than the best one I reported on thus far, and thus becomes
    the new best one, I will run any callbacks registered with me.
    """
    minDiff = 0.01
    
    def __init__(self, population):
        self.p = population
        self.pm = self.p.pm
        self.iBest = None
        self.iLastReported = None
        self.callbacks = []
        self.cbrInProgress = False
        self.cbrScheduled = Bag()
        self.dt = DeferredTracker()
        self.lock = defer.DeferredLock()

    def addCallback(self, func, *args, **kw):
        """
        Call this to register as a callback a function that accepts (1) a
        1-D Numpy array of I{values} from a new best Individual and
        (2) the value of my report I{counter}, as well as any other
        arguments and keywords you provide.

        I will run all the callbacks whenever I report on an
        Individual with an SSE that is lower than any of the other
        Individuals I've reported on up to that point.
        """
        if not callable(func):
            raise ValueError("You must supply a callable")
        self.callbacks.append((func, args, kw))

    def isEquivSSE(self, iBetter, iWorse):
        if iBetter is None or iWorse is None:
            return False
        betterSSE = iBetter.SSE
        worseSSE = iWorse.SSE
        if betterSSE is None or worseSSE == 0 or worseSSE is None:
            return False
        ratio = betterSSE / worseSSE
        return abs(ratio - 1.0) < self.minDiff 

    def runCallbacks(self, values):
        @defer.inlineCallbacks
        def runUntilFree():
            counter = self.p.counter
            self.cbrInProgress = True
            prev_values = None
            while self.cbrScheduled:
                values = self.cbrScheduled.pop()
                if prev_values is not None and \
                   np.all(np.equal(values, prev_values)):
                    break
                prev_values = values
                for func, args, kw in self.callbacks:
                    yield defer.maybeDeferred(
                        func, values, counter, *args, **kw)
            self.cbrInProgress = False

        self.cbrScheduled(values)
        if self.cbrInProgress:
            return
        self.dt.put(runUntilFree())

    def waitForCallbacks(self):
        return self.dt.deferToAll()

    @defer.inlineCallbacks
    def newBest(self, i):
        """
        Registers and reports a new best Individual. Calling this method
        is the only safe way to update I{iBest}.

        Triggers a run of callbacks with the best individual's 1-D
        array of I{values} and the integer report count for any
        callbacks that have been registered via my L{addCallback}
        method. The primary use of this is displaying a plot of the
        current best set of parameters.

        If a callback run is currently in progress, another will be
        done as soon as the current one is finished. That next run
        will refer to whatever is the best individual at the time. A
        rapid flurry of calls to L{newBest} will only callbacks
        necessary to ensure that the best individual has been
        referenced in a callback run.

        Returns a reference to the previous best value.
        """
        yield self.lock.acquire()
        if i.partial_SSE:
            # Inefficient: This evaluation is done again, and yet
            # again later by solver if callbacks are run. But they
            # don't get run all that often.
            yield i.evaluate()
        # Double-check that it's really better than my best
        if self.iBest is None or i < self.iBest:
            self.iBest = i
            values = copy(i.values)
            if not self.iLastReported or \
               not self.isEquivSSE(self.iLastReported, i):
                self.iLastReported = i
                self.runCallbacks(values)
        self.lock.release()
        
    def msgRatio(self, iNumerator, iDenominator, sym_lt="X"):
        """
        Returns 0 if numerator or denominator is C{None}, if numerator SSE
        is less than denominator SSE, if denominator SSE is C{None},
        or if numerator and denominator SSE are equivalent.

        Otherwise returns the rounded integer ratio of numerator SSE
        divided by denominator SSE.
        """
        if not iNumerator or not iDenominator:
            return 0
        if iNumerator < iDenominator:
            ratio = 0
            sym = sym_lt
        elif self.isEquivSSE(iDenominator, iNumerator):
            ratio = 0
            sym = "0"
        else:
            ratio = np.round(iNumerator.SSE / iDenominator.SSE)
            sym = str(int(ratio)) if ratio < 10 else "9"
        msg.writeChar(sym)
        return ratio

    def __call__(self, i=None, iOther=None, rbb=False, force=False):
        if i is None:
            if self.iBest:
                values = self.iBest.values
                self.runCallbacks(values)
            return
        if not i:
            return 0
        if iOther is None:
            if self.iBest is None:
                iOther = None
                self.newBest(i)
            else:
                iOther = self.iBest
                if force or i < iOther:
                    iOther = i
                    i = self.iBest
                    self.newBest(i)
            if rbb:
                return self.msgRatio(iOther, i, sym_lt="!")
            return self.msgRatio(i, iOther)
        if i < self.iBest:
            self.newBest(i)
        return self.msgRatio(iOther, i)
        
        
class Population(object):
    """
    Construct me with a sequence of I{bounds} that each define the
    lower and upper limits of my parameters, a callable I{func} that
    accepts a 1-D Numpy array of values within those limits and of the
    same length as the sequence of bounds and returns a fitness metric
    where lower values indicate better fitness.
    """
    # Default population size per parameter
    popsize = 10
    # Population is never smaller than this, no matter how few
    # parameters or requested size per parameter
    Np_min = 20
    # Population is never bigger than this, no matter how many
    # parameters or requested size per parameter
    Np_max = 300
    # Target score of improvements in each generation/iteration
    targetFraction = 5.0 / 100
    
    def __init__(self, func, names, bounds, constraints=[], popsize=None):
        def evalFunc(values, xSSE):
            return defer.maybeDeferred(func, values, xSSE)

        if not callable(func):
            raise ValueError(sub("Object '{}' is not callable", func))
        self.evalFunc = func # evalFunc
        self.Nd = len(bounds)
        self.pm = ParameterManager(names, bounds, constraints)
        self.reporter = Reporter(self)
        if popsize: self.popsize = popsize
        self.Np = min([self.popsize * self.Nd, self.Np_max])
        self.Np = max([self.Np_min, self.Np])
        self.kBest = None
        self.isProblem = False
        self.replacementScore = None
        self.statusQuoScore = self.targetFraction * self.Np
        self._sortNeeded = True
        self.counter = 0
        
    def __getitem__(self, k):
        return self.iList[k]
        
    def __setitem__(self, k, i):
        """
        Use only this method (item setting) and L{push} to replace
        individuals.
        """
        if not isinstance(i, Individual):
            raise TypeError("You can only set me with Individuals")
        if i.SSE == np.inf:
            self.isProblem = True
        self.iList[k] = i
        self._sortNeeded = True
        if self.kBest is None or i < self.iList[self.kBest]:
            # This one is now the best I have
            self.kBest = k
        
    def __len__(self):
        return self.Np

    def __iter__(self):
        for i in self.iList:
            yield i

    def __contains__(self, i):
        return i in self.iList

    @property
    def iSorted(self):
        if self._sortNeeded:
            self._iSorted = sorted(self.iList, key=lambda i: i.SSE)
            self._sortNeeded = False
        return self._iSorted
    
    def __repr__(self):
        def field(x):
            return "{:>11.5g}".format(x)

        def addRow():
            lineParts = ["{:>11s}".format(columns[0]), '|']
            for x in columns[1:]:
                lineParts.append(x)
            lines.append(" ".join(lineParts))
        
        N_top = (self.pm.maxLineLength-3) / 13
        iTops = self.iSorted[:N_top]
        if len(iTops) < N_top: N_top = len(iTops)
        lines = [
            sub("Population: Top {:d} of {:d} individuals", N_top, self.Np)]
        lines.append("")
        columns = ["SSE"] + [sub("{:>11.5f}", i.SSE) for i in iTops]
        addRow()
        lines.append(self.pm.dashes)
        X = np.empty([self.Nd, N_top])
        for kc, i in enumerate(iTops):
            X[:,kc] = i.values
        for kr, name in self.pm.sortedNamerator():
            columns = [name] + [field(X[kr,kc]) for kc in range(N_top)]
            addRow()
        lines.append(self.pm.dashes)
        lines.append(sub("Best individual:\n{}\n", repr(self.iSorted[0])))
        return "\n".join(lines)

    def limit(self, i):
        """
        Limits the individual's parameter values to the bounds in the way
        that my L{ParameterManager} is configured to do, modifying the
        individual in place.
        """
        values = self.pm.limit(i.values)
        i.update(values)

    def spawn(self, values, fromUnity=False):
        if fromUnity:
            values = self.pm.fromUnity(values)
        return Individual(self, values)

    @defer.inlineCallbacks
    def setup(self, uniform=False, blank=False):
        """
        Sets up my initial population using a Latin hypercube to
        initialize pseudorandom parameters with minimal
        clustering. With parameter constraints, this doesn't work as
        well, because the initial values matrix must be refreshed,
        perhaps many times. But it may still be better than uniform
        initial population sampling.

        Returns a C{Deferred} fires when the population has been set up.
        """
        def refreshIV():
            kIV[0] = 0
            IV = np.random.uniform(size=(self.Np, self.Nd)) \
                 if uniform else \
                    lhs(self.Nd, samples=self.Np, criterion='m')
            kIV[1] =  self.pm.fromUnity(IV)

        def getNextIV():
            k, IV = kIV
            if k+1 == IV.shape[0]:
                refreshIV()
                k, IV = kIV
            kIV[0] += 1
            return IV[k,:]

        def getIndividual():
            for k in range(1000):
                values = getNextIV()
                if self.pm.passesConstraints(values):
                    break
            else:
                raise RuntimeError(
                    "Couldn't generate a conforming Individual!")
            return Individual(self, self.pm.limit(values))
        
        def evaluated(i):
            ds.release()
            if i.SSE is None or len(self.iList) >= self.Np:
                return
            self.reporter(i)
            self.iList.append(i)
            self.dLocks.append(defer.DeferredLock())
        
        dList = []
        kIV = [None]*2
        self.iList = []
        self.dLocks = []
        refreshIV()
        msg(0, "Initializing {:d} population members", self.Np, '-'); msg()
        ds = defer.DeferredSemaphore(10)
        while True:
            yield ds.acquire()
            if len(self.iList) >= self.Np:
                break
            i = getIndividual()
            if blank:
                i.SSE = 1000
                evaluated(i)
            else: i.evaluate().addCallbacks(evaluated, oops)
        msg(0, repr(self))
        self._sortNeeded = True
        self.kBest = self.iList.index(self.iSorted[0])

    def addCallback(self, func, *args, **kw):
        self.reporter.addCallback(func, *args, **kw)

    def replacement(self, improvementRatio=None, sqs=None):
        """
        Call with an integer I{improvementRatio} to record a replacement
        occurring in this generation or iteration, or with the keyword
        I{sqs} set to a float I{statusQuoScore} other than my
        default. Otherwise, call with nothing to determine if the
        replacement that occurred in the previous generation/iteration
        were enough to warrant maintaining the status quo, and to
        reset the record.

        The status quo will be maintained if several small
        improvements are made, or fewer larger ones, with the required
        number and/or size increasing for a larger population. For
        small populations where even a single improvement would be
        significant, the probability of status quo maintenance
        increases with smaller population and will sometimes happen
        even with no improvements for a given generation or iteration.
        
        Returns C{True} if the status quo should be maintained.
        """
        if sqs:
            self.statusQuoScore = sqs
            return
        if improvementRatio is None:
            # Inquiry call, initialize score to zero
            score = self.replacementScore
            self.replacementScore = 0
            if score is None:
                # This is the first time ever called, so of course
                # status quo should be maintained
                return True
            if score:
                # Positive score, return True if greater than status
                # quo threshold
                return score >= self.statusQuoScore
            if self.statusQuoScore < 1.0:
                # A second chance for small populations with status
                # quo thresholds < 1
                return self.statusQuoScore < np.random.random_sample()
            # No replacements were made at all, so of course no status quo
            return False
        # An adjustment call
        if improvementRatio and self.replacementScore is not None:
            # 1 has only 0.25 weight
            # 2 has 1.25, or 5x as much as 1
            # 3 has 2.25, or nearly 2x as much as 2
            self.replacementScore += (improvementRatio - 0.75)

    def report(self, iNew=None, iOld=None, rbb=False, force=False):
        """
        Provides a message via the log messenger about the supplied
        Individual, optionally with a comparison to another
        Individual. If no second individual is supplied, the
        comparison will be with the best individual thus far reported
        on.

        Does a call to L{replacement} if the new individual is better.
        """
        ratio = self.reporter(iNew, iOld, rbb, force)
        if ratio:
            self.replacement(ratio)

    def waitForReports(self):
        return self.reporter.waitForCallbacks()
            
    def push(self, i):
        """
        Pushes the supplied individual into my population and kicks out
        the worst individual there to make room.
        """
        kWorst, self.kBest = [
            self.iList.index(self.iSorted[k]) for k in (-1, 0)]
        self[kWorst] = i
        self._sortNeeded = True
    
    def sample(self, N, *exclude):
        """
        Returns a sample of I{N} indices from my population that are
        unique from each other and from any excluded indices supplied
        as additional arguments.
        """
        kRange = [k for k in range(self.Np) if k not in exclude]
        result = random.sample(kRange, N)
        if N == 1:
            return result[0]
        return result

    def individuals(self, *indices):
        """
        Immediately returns a list of the individuals at the specified
        indices.
        """
        if len(indices) == 1:
            return self.iList[indices[0]]
        return [self.iList[k] for k in indices]
    
    def lock(self, *indices):
        """
        Obtains the locks for individuals at the specified indices,
        submits a request to acquire them, and returns a C{Deferred}
        that fires when all of them have been acquired.

        Release the locks (as soon as possible) by calling L{release}
        with the indices that are locked.
        """
        dList = []
        for k in indices:
            if indices.count(k) > 1:
                raise ValueError(
                    "Requesting the same lock twice will result in deadlock!")
            dList.append(self.dLocks[k].acquire())
        return defer.DeferredList(dList).addErrback(oops)

    def release(self, *indices):
        for k in indices:
            self.dLocks[k].release()

    def best(self):
        return self.iSorted[0]
    
        
