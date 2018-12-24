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

from asynqueue.null import NullQueue
from asynqueue.util import DeferredTracker

from individual import Individual
from util import *


class ParameterManager(object):
    """
    I manager the digital DNA parameters of the evolving species. I
    can pretty-print values with their parameter names, check if
    values pass constraints, limit values to their bounds, scale
    unity-range values to their appropriate ranges, and let you
    iterate over sorted parameter names.
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
        more complicated, but according to their Tables 1, 2, doesn't
        appear significantly better.)
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

    # Set True to show replacement of best individual on STDOUT
    debug = False
        
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
        self._syms_on_line = 0
        self.q = NullQueue(returnFailure=True)

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
                        func, values, counter, *args, **kw).addErrback(oops)
            self.progressChar()
            self.cbrInProgress = False

        self.cbrScheduled(values)
        if self.cbrInProgress:
            return
        self.dt.put(runUntilFree())

    def waitForCallbacks(self):
        return self.dt.deferToAll()

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
        # Double-check that it's really better than my best, after we
        # know a full evaluation has been done
        if self.iBest is None or i < self.iBest:
            if self.debug:
                try:
                    print sub("\n{}\n\t--->\n{}\n", repr(self.iBest), repr(i))
                except: pass
            self.iBest = i
            values = copy(i.values)
            if True or not self.iLastReported or \
               not self.isEquivSSE(i, self.iLastReported):
                self.iLastReported = i
                self.runCallbacks(values)
        else:
            import pdb; pdb.set_trace()
        
    def msgRatio(self, iNumerator, iDenominator, sym_lt="X"):
        """
        Returns 0 if numerator or denominator is C{None}, if numerator SSE
        is less than denominator SSE, if denominator SSE is C{None},
        or if numerator and denominator SSE are equivalent.

        Otherwise returns the rounded integer ratio of numerator SSE
        divided by denominator SSE.
        """
        if not iNumerator or not iDenominator:
            # This shouldn't happen
            return 0
        if iNumerator < iDenominator or np.isnan(iDenominator.SSE):
            ratio = 0
            sym = sym_lt
        elif self.isEquivSSE(iDenominator, iNumerator):
            ratio = 0
            sym = "0"
        elif np.isnan(iNumerator.SSE):
            ratio = 1000
            sym = "9"
        else:
            ratio = np.round(iNumerator.SSE / iDenominator.SSE)
            sym = str(int(ratio)) if ratio < 10 else "9"
        self.progressChar(sym)
        return ratio

    def progressChar(self, sym=None):
        """
        Logs the supplied ASCII character I{sym} to the current
        console/log line. If the number of symbols logged to the
        current line reaches my line-length limit, a is inserted. To
        reset the count of symbols logged to the current line, call
        this method with no symbol provided.
        """
        if sym is None:
            self._syms_on_line = 0
            return
        msg.writeChar(sym)
        self._syms_on_line += 1
        if self._syms_on_line == self.pm.maxLineLength-1:
            msg("")
            self._syms_on_line = 0

    def _fileReport(self, i, iOther):
        if iOther is None:
            if self.iBest is None:
                # First, thus best
                self.newBest(i)
                self.progressChar("*")
                result = 0
            elif i < self.iBest:
                # Better than (former) best, so make best. The "ratio"
                # of how much worse than best will be 0
                self.newBest(i)
                self.progressChar("!")
                result = 0
            else:
                # Worse than best (or same, unlikely), ratio is how
                # much worse
                result = self.msgRatio(i, self.iBest)
        else:
            # Ratio is how much better this is than other. Thus,
            # numerator is other, because ratio is other SSE vs this
            # SSE
            result = self.msgRatio(iOther, i)
            # If better than best (or first), make new best
            if self.iBest is None or i < self.iBest:
                self.newBest(i)
                self.progressChar("+")
        return result
    
    def __call__(self, i=None, iOther=None):
        """
        Files a report on the individual I{i}, perhaps vs. another
        individual I{iOther}. Returns with the
        ratio of how much better I{i} is than I{iOther}, or, if
        I{iOther} isn't specified, how much B{worse} I{i} is than the
        best individual reported thus far. (A "better" individual has
        a lower SSE.)
        """
        if i is None:
            if self.iBest:
                values = self.iBest.values
                self.runCallbacks(values)
            return
        if not i:
            return 0
        return self._fileReport(i, iOther)


class Population(object):
    """
    Construct me with a callable evaluation I{func} that accepts a 1-D
    Numpy array of parameter values, a sequence of I{names} for the
    parameters, and a sequence of I{bounds} containing 2-tuples that
    each define the lower and upper limits of the values.

    The evaluation function must return the sum of squared errors
    (SSE) as a single float value. In addition to the array of
    parameter values, it must also accept a keyword or optional second
    argument I{xSSE}, which (if provided) is the SSE of the target
    individual being challenged. The evaluation function need not
    continue its computations if it accumulates an SSE greater than
    I{xSSE}; it can just return that greater SSE and conclude
    operations. That's because DE uses a greedy evaluation function,
    where the challenger will always be accepted if it is *any* better
    than the target.

    If just one argument (the paremeter value 1-D array) is provided,
    your evaluation function must return the fully computed SSE.
    """
    # Default population size per parameter
    popsize = 10
    # Population is never smaller than this, no matter how few
    # parameters or requested size per parameter
    Np_min = 20
    # Population is never bigger than this, no matter how many
    # parameters or requested size per parameter
    Np_max = 500
    # Target score of improvements in each generation/iteration
    targetFraction = 4.0 / 100
    
    def __init__(self, func, names, bounds, constraints=[], popsize=None):
        def evalFunc(values):
            return defer.maybeDeferred(func, values)

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
        lines.append(sub("Best individual:\n{}\n", repr(self.best())))
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
    def setup(self, uniform=False, blank=False, filePath=None):
        """
        Sets up my initial population using a Latin hypercube to
        initialize pseudorandom parameters with minimal clustering,
        unless I{uniform} is set. With parameter constraints, this
        doesn't work as well, because the initial values matrix must
        be refreshed, perhaps many times. But it may still be better
        than uniform initial population sampling.

        If I{blank} is set, the initial individuals are all given a
        placeholder infinite SSE instead of being evaluated.

        #TODO: Load from filePath.

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
                i.SSE = np.inf
                evaluated(i)
            else: i.evaluate().addCallbacks(evaluated, oops)
        msg(0, repr(self))
        self._sortNeeded = True
        self.kBest = self.iList.index(self.iSorted[0])
    
    def save(self, filePath):
        """
        TODO
        """
        
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

    def report(self, iNew=None, iOld=None):
        """
        Provides a message via the log messenger about the supplied
        Individual, optionally with a comparison to another
        Individual. If no second individual is supplied, the
        comparison will be with the best individual thus far reported
        on.
        
        Does a call to L{replacement} if the new individual is better.
        """
        ratio = self.reporter(iNew, iOld)
        if ratio: self.replacement(ratio)

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
