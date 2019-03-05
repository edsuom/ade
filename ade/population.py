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
A L{Population} class and helpers.

What you'll need to be concerned with is mostly constructing an
instance, setting it up, and passing it to
L{de.DifferentialEvolution}. The constructor requires an evaluation
function, parameter names, and parameter bounds. You'll need to wait
for the C{Deferred} that L{Population.setup} returns before
proceeding.
"""

import random
from copy import copy
from textwrap import TextWrapper

import numpy as np
from scipy import stats

from pyDOE import lhs
from twisted.internet import defer, task

from asynqueue.null import NullQueue
from asynqueue.util import DeferredTracker

import abort
from individual import Individual
from util import *


class ParameterManager(object):
    """
    I manage the digital DNA parameters of the evolving species.

    I can pretty-print values with their parameter names, check if
    values pass constraints, limit values to their bounds, scale
    unity-range values to their appropriate ranges, and let you
    iterate over sorted parameter names.

    @ivar mins: Lower bound of each parameter.
    @ivar maxs: Lower bound of each parameter.
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

        Adds a '*' if < 5% of the way from lower to upper bound, or
        '**' if > 95% of the way

        You can provide as an additional argument a prelude string, or
        a string proto with additional args, and the string will
        precede the values.
        """
        lineParts = []
        if args:
            lineParts.append(args[0].format(*args[1:]))
        unityValues = self.toUnity(values)
        for k, name, value in self.sortedNamerator(values):
            part = "{}={:g}".format(name, value)
            uv = unityValues[k]
            if uv < 0.05:
                part += "*"
            elif uv > 0.95:
                part += "**"
            lineParts.append(part)
        text = " ".join(lineParts)
        return self.fill(text)

    def sortedNamerator(self, values=None):
        """
        Generates tuples of sorted names.

        Each tuple contains (1) the index in a I{values} list of
        parameters where each named parameter appears, and (2) the
        name itself. If such a list of I{values} is supplied, each
        tuple also includes (3) the value for that name.
        """
        if values is None:
            for k in self.sortedNameIndices:
                yield k, self.names[k]
        else:
            for k in self.sortedNameIndices:
                yield k, self.names[k], values[k]
        
    def fromUnity(self, values):
        """
        Translates normalized into actual values.
        
        Converts the supplied normalized I{values} from the
        standardized range of 0-1 into my range of actual parameter
        values within the ranges specified in the bounds supplied to
        my constructor.
        """
        scaled = self.scales * values
        return self.mins + scaled

    def toUnity(self, values):
        """
        Translates actual into normalized values.
        
        Converts the supplied actual parameter I{values} into the
        standardized range of 0-1 within the ranges specified in the
        bounds supplied to my constructor.
        """
        return (values - self.mins) / self.scales
    
    def passesConstraints(self, values):
        """
        Checks if I{values} pass all my constraints.
        
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

    Construct an instance of me with a L{Population}. Then you can
    call the instance (as many times as you like) with an
    L{Individual} to report on its SSE compared to the best one I've
    reported on thus far.

    @see: L{__call__}.

    @keyword complaintCallback: Set to a callable that accepts an
        L{Individual} instance and the result of a callback function
        that may complain about the individual whose values and SSE
        were reported to it.

    @cvar minDiff: The minimum fractional difference between a new
        best individual's SSE and that of the previous best individual
        for a report to be filed about the new one. The default is
        0.02, or at least a 2% improvement required.
    """
    minDiff = 0.02

    def __init__(self, population, complaintCallback=None):
        self.p = population
        self.complaintCallback = complaintCallback
        self.pm = self.p.pm
        self.iBest = None
        self.iLastReported = None
        self.callbacks = []
        self.cbrInProgress = False
        self.cbrScheduled = Bag()
        self.dt = DeferredTracker()
        self._syms_on_line = 0

    def abort(self):
        """
        Tells me not to wait for any pending or future callbacks to run.
        """
        self.p.abort(ignoreReporter=True)
        self.dt.quitWaiting()
    
    def addCallback(self, func, *args, **kw):
        """
        Call this to register a new callback.

        The supplied I{func} must accept as arguments (1) a 1-D Numpy
        array of I{values} from a new best Individual, (2) the value
        of my report I{counter}, and (3) the individual's SSE,
        followed by any other arguments and keywords you provide.

        I will run all the callbacks whenever I report on an
        Individual with an SSE that is lower than any of the other
        Individuals I've reported on up to that point.
        """
        if not callable(func):
            raise ValueError("You must supply a callable")
        self.callbacks.append((func, args, kw))

    def isEquivSSE(self, iBetter, iWorse):
        """
        Returns C{True} if the SSE of individual I{iBetter} is not
        significantly different than that of individual I{iWorse}.

        What is "significant" is defined by my I{minDiff} attribute.
        """
        if iBetter is None or iWorse is None:
            return False
        betterSSE = iBetter.SSE
        worseSSE = iWorse.SSE
        if betterSSE is None or worseSSE == 0 or worseSSE is None:
            return False
        ratio = float(betterSSE) / float(worseSSE)
        return abs(ratio - 1.0) < self.minDiff 

    def runCallbacks(self, i, iPrevBest=None):
        """
        Queues up a report for the supplied L{Individual} I{i}, calling
        each of my registered callbacks in turn.

        If any callback complains about the report by returning a
        result (deferred or immediate) that is not C{None}, processes
        the complaint and then gives the individual a worst-possible
        SSE.

        Default processing is to enter the debugger so the user can
        figure out what the callback was complaining about. But if my
        I{complaintCallback} is set to a callable (must accept the
        individual and the complainer's returned result as its two
        args), that will be called instead of the debugger.

        If the callback had an error, it is logged, processing of any
        further callbacks is halted, and I get aborted.

        @keyword iPrevBest: Set to the previous best individual to
            allow for it to be restored to its rightful place if a
            callback complains about I{i}.
        """
        def failed(failureObj, func):
            # TODO: Provide a complete func(*args, **kw).
            callback = repr(func)
            info = failureObj.getTraceback()
            msg(0, "FATAL ERROR in report callback {}:\n{}\n{}\n",
                callback, info, '-'*40)
            return CallbackFailureToken()
        
        @defer.inlineCallbacks
        def runUntilFree():
            counter = self.p.counter
            self.cbrInProgress = True
            msg.lineWritten()
            iPrev = None
            while self.p.running and self.cbrScheduled:
                i = self.cbrScheduled.pop()
                if iPrev and i == iPrev:
                    continue
                iPrev = i
                for func, args, kw in self.callbacks:
                    if not self.p.running: break
                    d = defer.maybeDeferred(
                        func, i.values, counter, i.SSE, *args, **kw)
                    d.addErrback(failed, func)
                    result = yield d
                    if isinstance(result, CallbackFailureToken):
                        self.abort()
                        break
                    if result is not None and i and self.p.running:
                        if self.complaintCallback:
                            yield defer.maybeDeferred(
                                self.complaintCallback, i, result)
                        else:
                            text = sub(
                                "\n\n" +\
                                "Callback function complained about {}\n", i)
                            print(text)
                            import pdb; pdb.set_trace()
                        # Modify the individual in-place to never again be
                        # winner of a challenge or basis for new individuals
                        i.blacklist()
                        if iPrevBest and iPrevBest < self.iBest:
                            self.iBest = iPrevBest
            if msg.lineWritten(): self.progressChar()
            self.cbrInProgress = False

        self.cbrScheduled(i)
        if self.cbrInProgress:
            return
        self.dt.put(runUntilFree())

    def waitForCallbacks(self):
        """
        Returns a C{Deferred} that fires when all reporting callbacks
        currently queued up have been completed.
        """
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
            if self.p.debug:
                msg(0, "{}\n\t--->\n{}\n", self.iBest, i)
            iPrevBest = self.iBest
            self.iBest = i
            if self.iLastReported and self.isEquivSSE(i, self.iLastReported):
                return
            self.iLastReported = i
            self.runCallbacks(i, iPrevBest)
            return
        print("Well, shit. New best wasn't actually best. Fix this!\n")
        import pdb; pdb.set_trace()
        
    def msgRatio(self, iNumerator, iDenominator, sym_lt="X"):
        """
        Returns 0 if I{iNumerator} or I{iDenominator} is C{None},
        numerator SSE < denominator SSE, denominator SSE is C{None},
        or numerator and denominator SSE are equivalent.

        "Equivalent" means a call to L{isEquivSSE} determines that the
        two individuals have SSEs with a fractional difference less
        than my I{minDiff} attribute.

        Otherwise returns the rounded integer ratio of numerator SSE
        divided by denominator SSE.

        For example, if I{iDenominator} has an SSE of 100.0, returns 1
        if I{iNumerator} has an SSE between 101.1 and 149.9. If
        I{iNumerator} has an SSE of 100.9, it is considered equivalent
        to I{iDenominator} and 0 will be returned. If its SSE is
        between 150.0 and 249.9, the return value is 2.
        """
        def has_nan(i):
            try: return np.isnan(i.SSE)
            except: pass
        
        if not iNumerator or not iDenominator:
            # This shouldn't happen
            return 0
        if not iNumerator.SSE or not iDenominator.SSE:
            # Neither should this
            return 0
        if iNumerator < iDenominator or has_nan(iDenominator):
            ratio = 0
            sym = sym_lt
        elif self.isEquivSSE(iDenominator, iNumerator):
            ratio = 0
            sym = "0"
        elif has_nan(iNumerator):
            ratio = 1000
            sym = "9"
        else:
            ratio = np.round(float(iNumerator.SSE) / float(iDenominator.SSE))
            sym = str(int(ratio)) if ratio < 10 else "9"
        self.progressChar(sym)
        return ratio

    def progressChar(self, sym=None):
        """
        Logs the supplied ASCII character I{sym} to the current
        console/log line.

        If the number of symbols logged to the current line reaches my
        line-length limit, a is inserted. To reset the count of
        symbols logged to the current line, call this method with no
        symbol provided.
        """
        if sym is None:
            if self._syms_on_line: msg("")
            self._syms_on_line = 0
            return
        msg.writeChar(sym)
        self._syms_on_line += 1
        if self._syms_on_line > self.pm.maxLineLength-1:
            self._syms_on_line = 0
            msg("")

    def _fileReport(self, i, iOther):
        """
        Called by L{__call__}. Calls L{msgRatio} with I{iOther} vs I{i} to
        get the ratio of how much better I{i} is than I{other}, if not
        C{None}.

        If I{other} is C{None} and I{i} is worst than my best
        L{Individual}, calls L{msgRatio} with I{i} vs. the best
        individual to get the ratio of how much worse I{i} is than it.

        Prints a progress character to STDOUT or the log indicating
        the improvement, adding a "+" if I{i} is a new best
        individual. A new best individual, no matter how small the
        ratio, adds a small bonus to my L{Population} object's
        replacements score, equivalent to a rounded improvement ratio
        of 1.
        """
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
                # Bonus rir
                self.p.replacement(1)
        return result
    
    def __call__(self, i=None, iOther=None):
        """
        Files a report on the individual I{i}, perhaps vs. another
        individual I{iOther}.

        Logs (to STDOUT or a logfile) a single numeric digit 1-9
        indicating how much worse (higher SSE) the new individual is
        than the best one, or an "X" if the new individual becomes the
        best one. The integer ratio will be returned, or 0 if if the
        new individual became best.
    
        Call with two individuals to report on the SSE of the first
        one compared to the second. A single numeric digit 1-9 will be
        logged indicating how much B{better} (lower SSE) the first
        individual is than the second one, or an "X" if the first
        individual is actually worse. The integer ratio will be
        returned, or 0 if the first individual was worse.
    
        In either case, whenever the first- or only-supplied
        Individual is better than the best one I reported on thus far,
        and thus becomes the new best one, I will run any callbacks
        registered with me.

        Returns the ratio of how much better I{i} is than I{iOther},
        or, if I{iOther} isn't specified, how much B{worse} I{i} is
        than the best individual reported thus far. (A "better"
        individual has a lower SSE.)

        @see: L{_fileReport}.
        """
        if i is None:
            if self.iBest:
                self.runCallbacks(self.iBest)
            return
        if not i:
            return 0
        return self._fileReport(i, iOther)


class Population(object):
    """
    I contain a population of parameter-combination L{Individual}
    objects.
    
    Construct me with a callable evaluation I{func}, a sequence of
    parameter I{names}, and a sequence of I{bounds} containing
    2-tuples that each define the lower and upper limits of the
    values:

        - I{func}: A callable to which an L{Individual} can send its
          parameter values and from which it receives a sum-of-squared
          error float value as a result.

        - I{names}: A sequence of parameter names.

        - I{bounds}: A list of 2-tuples, one for each parameter
          name. The first element of each tuple is the lower bound of
          a parameter in the second the upper bound.

    The callable I{func} must accept a single 1-D Numpy array as its
    sole argument and return the sum of squared errors (SSE) as a
    single float value. To shut down I{ade}, it can return a negative
    SSE value. If I{ade} is shutting down, it will use I{None} as the
    argument, and the callable should act accordingly.
    
    @keyword constraints: A list of callables that enforce any
        constraints on your parameter values. See
        L{ParameterManager.passesConstraints}.
    
    @keyword popsize: The number of individuals per parameter in the
        population, if not the default.

    @keyword debug: Set C{True} to override my default I{debug}
        setting and ensure that I show individuals getting replaced.

    @keyword complaintCallback: A callable that my L{Reporter} calls
        with an individual and the non-None result of a complaining
        reporter callback. See L{Reporter.runCallbacks}.

    @keyword targetFraction: Set this to a (small) float to override
        my default target for the total score of improvements in each
        iteration.

    @cvar N_maxParallel: The maximum number of parallel evaluations
        during population L{setup}. Uses an instance of
        C{asynqueue.util.DeferredTracker} for concurrency limiting.
    
    @ivar popsize: The number of individuals per parameter. The
        population size will scale with the number of parameters, up
        until I{Np_max} is reached. Default is 10 individuals per
        parameter.
    
    @ivar Np_min: Minimum population size, i.e., my total number of
        individuals. Default is 20.

    @ivar Np_max: Maximum population size. Default is 500, which is
        really pretty big.

    @ivar targetFraction: The desired total score of improvements in
        each iteration in order for I{ade}'s adaptive algorithm to not
        change the current differential weight. See L{replacement} and
        L{FManager} for details. The default is 3%.

    @ivar debug: Set C{True} to show individuals getting
        replaced. (Results in a very messy log or console display.)

    @ivar running: Indicates my run status: C{None} after
        instantiation but before L{setup}, C{True} after setup, and
        C{False} if I{ade} is aborting.

    @see: U{asynqueue.util.DeferredTracker<http://edsuom.com/AsynQueue/asynqueue.util.DeferredTracker.html>}, used to limit concurrency during population L{setup}.
    """
    popsize = 10
    Np_min = 20
    Np_max = 500
    N_maxParallel = 12
    targetFraction = 0.03
    debug = False
    
    def __init__(
            self, func, names, bounds,
            constraints=[], popsize=None,
            debug=False, complaintCallback=None, targetFraction=None):
        """
        C{Population(func, names, bounds, constraints=[], popsize=None,
        debug=False, complaintCallback=None)}
        """
        def evalFunc(values):
            if self.running is False:
                values = None
            return defer.maybeDeferred(func, values)

        if not callable(func):
            raise ValueError(sub("Object '{}' is not callable", func))
        self.evalFunc = evalFunc
        self.Nd = len(bounds)
        if debug: self.debug = True
        if targetFraction:
            self.targetFraction = targetFraction
            msg("WARNING: Non-default target improvement score of {:f}",
                targetFraction)
        self.pm = ParameterManager(names, bounds, constraints)
        self.reporter = Reporter(self, complaintCallback)
        if popsize: self.popsize = popsize
        self.Np = max([
            self.Np_min, min([self.popsize * self.Nd, self.Np_max])])
        self.kBest = None
        self.isProblem = False
        self.replacementScore = None
        self.statusQuoScore = self.targetFraction * self.Np
        self._sortNeeded = True
        self.counter = 0
        self.iList = []
        self.dLocks = []
        self.running = None
        abort.callOnAbort(self.abort)
        
    def __getitem__(self, k):
        """
        Sequence-like access to my individuals.
        """
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
        """
        Sequence-like container of individuals: length.

        My length will be equal to my I{Np} attribute unless setup
        has not been completed.
        """
        return len(self.iList)

    def __iter__(self):
        """
        Sequence-like container of individuals: iteration.
        """
        for i in self.iList:
            yield i

    def __contains__(self, i):
        """
        Sequence-like container of individuals: "in".
        """
        return i in self.iList
    
    @property
    def iSorted(self):
        """
        Property: A list of my individuals, sorted by increasing
        (worsening) SSE.
        """
        if self._sortNeeded:
            self._iSorted = sorted(self.iList, key=lambda i: i.SSE)
            self._sortNeeded = False
        return self._iSorted
    @iSorted.deleter
    def iSorted(self):
        """
        Property: "Deleting" my sorted list of individuals forces
        regeneration of the sorted list that will be returned next
        time the I{iSorted} property is accessed.
        """
        self._iSorted = sorted(self.iList, key=lambda i: i.SSE)
    
    def __repr__(self):
        """
        An informative string representation with a text table of my best
        individuals.
        """
        def field(x):
            return sub("{:>11.5g}", float(x))

        def addRow():
            lineParts = ["{:>11s}".format(columns[0]), '|']
            for x in columns[1:]:
                lineParts.append(x)
            lines.append(" ".join(lineParts))
        
        N_top = (self.pm.maxLineLength-3) / 13
        iTops = self.iSorted[:N_top]
        if len(iTops) < N_top: N_top = len(iTops)
        SSEs = [float(i.SSE) for i in self]
        lines = [sub(
            "Population: {:d} individuals with SSE {:.5g} to "+\
            "{:.5g}, avg eval time {:.3g} sec. Top {:d}:",
            self.Np, min(SSEs), max(SSEs), np.mean(self.evalTimes()), N_top)]
        lines.append("")
        columns = ["SSE"] + [field(i.SSE) for i in iTops]
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
        """
        Spawns a new L{Individual} with the supplied I{values}. If
        I{fromUnity} is set C{True}, the values are converted from 0-1
        range into their proper ranges.
        """
        if fromUnity:
            values = self.pm.fromUnity(values)
        return Individual(self, values)

    def abort(self, ignoreReporter=False):
        """
        Aborts my operations ASAP. Repeated calls will release any
        locks that got acquired since the last call.

        L{Reporter.abort} calls this with I{ignoreReporter} set
        C{True} to avoid infinite recursion.
        """
        self.running = False
        if not ignoreReporter:
            msg("Shutting down reporter")
            self.reporter.abort()
        # This next little line may run a bunch of stuff that was
        # waiting for locks
        msg("Releasing locks")
        self.release()
        msg("Population object stopped")
    
    def setup(self, uniform=False, blank=False, filePath=None):
        """
        Sets up my initial population using a Latin hypercube to
        initialize pseudorandom parameter values with minimal clustering.
        
        Unless I{uniform} is set, that is. Then each parameter values
        is just uniformly random without regard to the others.

        With parameter constraints, the Latin hypercube doesn't work
        that well. The initial values matrix must be refreshed,
        perhaps many times. But it may still be better than uniform
        initial population sampling.

        TODO: Load from I{filePath}.

        Sets my I{running} flag C{True} and returns a C{Deferred} that
        fires when the population has been set up.
        
        @keyword uniform: Use uniform random variates instead of a
            Latin hypercube (LHS). Using LHS (the default) is usually
            better because initializes pseudorandom parameter values
            with minimal clustering.

        @keyword blank: Set C{True} to give the initial individuals an
             infinite placeholder SSE instead of being evaluated.

        """
        def running():
            return self.running is not False
        
        def refreshIV():
            kIV[0] = 0
            IV = np.random.uniform(
                size=(self.Np, self.Nd)) if uniform else lhs(
                    self.Nd, samples=self.Np, criterion='m')
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
            else: raise RuntimeError(
                    "Couldn't generate a conforming Individual!")
            return Individual(self, self.pm.limit(values))

        def addIndividual(i):
            self.iList.append(i)
            self.dLocks.append(defer.DeferredLock())
        
        def evaluated(i, d):
            if not i:
                msg(0, "Bogus initial evaluation of {}, aborting", i)
                self.abort()
                return
            if i.SSE is not None and len(self.iList) < self.Np:
                addIndividual(i)
                self.reporter(i)

        @defer.inlineCallbacks
        def populate():
            k = 0
            while running():
                i = getIndividual()
                if blank:
                    i.SSE = np.inf
                    addIndividual(i)
                    continue
                k += 1
                d = i.evaluate()
                d.addCallback(evaluated, d)
                d.addErrback(oops)
                dt.put(d)
                yield dt.deferUntilFewer(self.N_maxParallel)
                if k >= self.Np:
                    break
            yield dt.deferToAll()

        def done(null):
            if running():
                msg(0, repr(self))
                self._sortNeeded = True
                self.kBest = self.iList.index(self.iSorted[0])
                self.running = True
            else: self.Np = 0
            
        if not running():
            return defer.succeed(None)
        dt = DeferredTracker(interval=0.05)
        kIV = [None]*2; refreshIV()
        msg(0, "Initializing {:d} population members having {:d} parameters",
            self.Np, self.Nd, '-')
        return populate().addCallback(done)
            
    def save(self, filePath):
        """
        (Not implemented yet.)
        
        This will save my individuals to a data file at I{filePath} in
        a way that can repopulate a new instance of me with those same
        individuals and without any evaluations being required.
        """
        raise NotImplementedError("TODO...")
        
    def addCallback(self, func, *args, **kw):
        """
        Adds callable I{func} to my reporter's list of functions to call
        each time there is a significantly better L{Individual}.

        @see: L{Reporter.addCallback}.
        """
        self.reporter.addCallback(func, *args, **kw)

    def replacement(self, rir=None, sqs=None):
        """
        Records the replacement of an L{Individual} in this generation or
        iteration.

        Call with an integer B{r}ounded B{i}mprovement B{r}atio in a
        loser's SSE vs. the successful challenger's SSE, unless you
        are calling to inquire about whether the status quo I{F}
        value(s) should be maintained or to set my I{statusQuoteScore}
        with the I{sqs} keyword.
        
        Three types of calls
        ====================

            The rounded improvement ratio I{rir} indicates how much
            better the challenger is than the individual it
            replaced. I use that ratio to adjust a running score for
            the current iteration to inform the status quo inquiry
            that will occur when the iteration is done, unless I'm not
            running in adaptive mode.
            
            You can set my target I{statusQuoScore} by setting I{sqs}
            to a (small) float value. That will replace my default
            value for future evaluation of replacement individuals.
    
            Finally, a status quo inquiry is a call with no keywords
            set. I will determine if the replacements that occurred
            in the previous generation/iteration were enough to
            warrant maintaining the status quo, and then reset the
            record. You will receive a result of C{True} if the status
            quote should be maintained.

            The status quo should be maintained if several small
            improvements are made, or fewer larger ones, with the
            required number and/or size increasing for a larger
            population. For small populations where even a single
            improvement would be significant, the probability of
            status quo maintenance increases with smaller population
            and will sometimes happen even with no improvements for a
            given generation or iteration.

        Improvement Ratios
        ==================

            An I{rir} of 1 indicates that the successful challenger
            was better (i.e., lower) and not considered equivalent to
            that of the individual it replaced, and that its SSE was
            no better than 1.5x as good (2/3 as high) as the replaced
            individual's SSE. An I{rir} of 2 indicates that the
            challenger had an SSE between 1.5x and 2.5x better than
            (2/5 to 2/3 as high as) the individual it replaced.

            I don't give much weight to an I{rir} of 1. The
            improvement is pretty modest and could be as little as 2%
            (assuming C{Reporter.minDiff}=0.02, the default). An
            I{rir} of 2 gets five times as much weight as that.

            An I{rir} of 3 also gets disproportionately more weight,
            nearly twice as much as I{rir}=2. Beyond that, though, the
            weight scales in a nearly linear fashion. For example, an
            I{rir} of 9 adds just a little more than three times to
            the score (3.67x) as I{rir}=3 does.

            Here's a practical example, with a population of 100
            individuals: If you see 10 "1" characters on the screen
            for one iteration with other 90 being "X," your ratio
            score for that iteration will be 2.5. But if you see just
            one non-X individual with a "8" character, the score will
            be 7.25. That one amazing success story counts far more in
            a sea of failures than a bunch of marginal improvements,
            which is kind of how evolution works in real life. (See
            the literature around "hopeful monsters.")
        
        @keyword rir: A rounded improvement ratio obtained from a call
            to L{msgRatio}, where the numerator is the SSE of the
            individual that was replaced and the denominator is the
            SSE of its successful challenger.
        
        @see: L{report}, which calls this.
        """
        if sqs:
            self.statusQuoScore = sqs
            return
        if rir is None:
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
        if rir and self.replacementScore is not None:
            # 1 has only 0.25 weight
            # 2 has 1.25, or 5x as much as 1
            # 3 has 2.25, or nearly 2x as much as 2
            self.replacementScore += (rir - 0.75)

    def report(self, iNew=None, iOld=None):
        """
        Provides a message via the log messenger about the supplied
        Individual, optionally with a comparison to another
        Individual.

        If no second individual is supplied, the comparison will be
        with the best individual thus far reported on.
        
        Gets the ratio from a call to my L{Reporter} instance, and
        does a call to L{replacement} with it if the new individual is
        better.
        """
        if self.running is False: return
        ratio = self.reporter(iNew, iOld)
        if ratio: self.replacement(ratio)

    def waitForReports(self):
        """
        Returns a C{Deferred} that fires when all reporter callbacks have
        finished.
        """
        if not self.running:
            return defer.succeed(None)
        return self.reporter.waitForCallbacks()
            
    def push(self, i):
        """
        Pushes the supplied L{Individual} I{i} onto my population and
        kicks out the worst individual there to make room.
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
        integer index or indices.
        """
        result = []
        for k in indices:
            if k >= len(self.iList): return
            result.append(self.iList[k])
        return result[0] if len(result) == 1 else result
    
    def lock(self, *indices):
        """
        Obtains the locks for individuals at the specified indices,
        submits a request to acquire them, and returns a C{Deferred}
        that fires when all of them have been acquired.

        Release the locks (as soon as possible) by calling L{release}
        with the indices that are locked.

        If I'm shutting down, the returned C{Deferred} fires
        immediately.
        """
        if self.running is False:
            return defer.succeed(None)
        dList = []
        for k in indices:
            if indices.count(k) > 1:
                raise ValueError(
                    "Requesting the same lock twice will result in deadlock!")
            if k >= len(self.dLocks):
                # Invalid index, we must be shutting down
                self.release()
                return defer.succeed(None)
            dList.append(self.dLocks[k].acquire())
        return defer.DeferredList(dList).addErrback(oops)

    def release(self, *indices):
        """
        Releases any active lock for individuals at the specified index or
        indices.

        If no indices are supplied, releases all active locks. (This
        is for aborting only.)
        """
        def tryRelease(dLock):
            if dLock.locked:
                dLock.release()

        if indices:
            for k in indices:
                tryRelease(self.dLocks[k])
            return
        for dLock in self.dLocks:
            tryRelease(dLock)

    def best(self):
        """
        Returns my best individual, or C{None} if I have no individuals yet.
        """
        if self.iSorted:
            return self.iSorted[0]

    def evalTimes(self):
        """
        Returns a list of the most recent elapsed evaluation times for
        each of my individuals that have done evaluations.
        """
        dtList = []
        for i in self:
            if i.dt is None: continue
            dtList.append(i.dt)
        return dtList
        
