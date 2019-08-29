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

import random, pickle, os.path, bz2
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
from report import Reporter
from history import History
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
    maxLineLength = 120
    dashes = "-"*maxLineLength
    fill = TextWrapper(width=maxLineLength, break_on_hyphens=False).fill

    def __init__(self, names, bounds, constraints=[]):
        if len(bounds) != len(names):
            raise ValueError(
                "Define one parameter name for each lower/upper bound")
        self.names = names
        self.sortedNameIndices = [
            names.index(name) for name in sorted(names)]
        self.constraints = constraints if hasattr(
            constraints, '__iter__') else [constraints]
        self.setup(bounds)

    def setup(self, bounds):
        """
        Call to set (or reset) the bounds of my parameters.
        """
        self.mins = np.array([x[0] for x in bounds])
        self.maxs = np.array([x[1] for x in bounds])
        self.scales = self.maxs - self.mins
        self.mids = self.mins + 0.5 * self.scales

    def __getstate__(self):
        """
        For pickling.
        """
        state = {}
        names = {
            'names', 'sortedNameIndices',
            'mins', 'maxs', 'scales', 'mids', 'constraints',
        }
        for name in names:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """
        For unpickling.
        """
        for name in state:
            setattr(self, name, state[name])

    def stringValue(self, k, value, forColumn=False):
        """
        For the parameter with position I{k}, returns the float
        I{value} formatted as a string. Adds a '*' if value within 5%
        of the lower bound, or '**' if within 5% of the upper bound.
        """
        uValue = (value - self.mins[k]) / self.scales[k]
        suffix = "*" if uValue < 0.05 else "**" if uValue > 0.95 else ""
        proto = "{:>10.5g}{:2s}" if forColumn else "{:g}{}"
        return sub(proto, float(value), suffix)
        
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
            part = sub("{}={}", name, self.stringValue(k, value))
            lineParts.append(part)
        text = " ".join(lineParts)
        return self.fill(text)

    def sortedNamerator(self, values=None, namesOnly=False):
        """
        Generates tuples of sorted names, or just the sorted names if
        I{namesOnly} is set C{True}.

        Each tuple contains (1) the index in a I{values} list of
        parameters where each named parameter appears, and (2) the
        name itself. If such a list of I{values} is supplied, each
        tuple also includes (3) the value for that name.
        """
        if namesOnly:
            for k in self.sortedNameIndices:
                yield self.names[k]
        elif values is None:
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

    def toUnity(self, values, ):
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


class ProbabilitySampler(object):
    """
    Call an instance of me with a sequence of indices, sorted in
    ascending order of the SSE of the individual they point to, and a
    float version of I{randomBase} to get a best-biased index sample.
    """
    N_chunk = 100
    
    def __init__(self):
        self.rc = None
        self.RV = None

    def trapz(self, rc):
        """
        Returns a random variate from a half-trapezoid distribution with
        the start of the triangular portion I{rc} specified between
        0.0 and 1.0.
        """
        pr_tri = 0.0 if rc >= 1.0 else (1.0 - rc) / (1.0 + rc)
        if random.random() < pr_tri:
            # Sample from triangular portion
            return rc + random.triangular(0, 1.0-rc, 0)
        # Sample from uniform (rectangular) portion
        return random.uniform(0, rc)
    
    def __call__(self, K, rb):
        if rb > 0.5:
            rc = 2*(rb - 0.5)
            rv = self.trapz(rc)
        else:
            rc = 2*rb
            rv = random.triangular(0, rc, 0)
        return K[int(rv*len(K))]

    
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

    My I{targetFraction} attribute determines how much success
    challengers must have to maintain the status quo in adaptive
    mode. Consider the default of 2.5%: In a population of 100, that
    is reached with a score of 2.5, which can be achieved, for
    example, with

        - ten challengers winning with a rounded improvement ratio
          of 1; or

        - one challenger winning with an I{rir} of 2 and five with an
          I{rir} of 1; or

        - just one challenger winning with an I{rir} of 3.

        - Or, if you're somehow positioned at a subtle transition in
          the fitness landscape along just the right multi-dimensional
          angle, fully half of the challengers winning with an I{rir}
          of 0. (Unlikely!)
    
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
        L{FManager} for details. The default is 2%. (Previously, it
        was 2.5% but that seemed too strict for the application the
        author is mostly using ADE for.)

    @ivar debug: Set C{True} to show individuals getting
        replaced. (Results in a very messy log or console display.)

    @ivar running: Indicates my run status: C{None} after
        instantiation but before L{setup}, C{True} after setup, and
        C{False} if I{ade} is aborting.

    @see: U{asynqueue.util.DeferredTracker<http://edsuom.com/AsynQueue/asynqueue.util.DeferredTracker.html>}, used to limit concurrency during population L{setup}.
    """
    maxTries = 2000
    
    popsize = 10
    Np_min = 20
    Np_max = 500
    N_maxParallel = 12
    targetFraction = 0.02
    debug = False
    failedConstraintChar = " "
    # Property placeholders
    _KS = None; _iSorted = None
    
    def __init__(
            self, func, names, bounds,
            constraints=[], popsize=None,
            debug=False, complaintCallback=None, targetFraction=None):
        """
        C{Population(func, names, bounds, constraints=[], popsize=None,
        debug=False, complaintCallback=None)}
        """
        if not callable(func):
            raise ValueError(sub("Object '{}' is not callable", func))
        self.func = func
        self.Nd = len(bounds)
        if debug: self.debug = True
        if targetFraction:
            self.targetFraction = targetFraction
            msg("WARNING: Non-default target improvement score of {:f}",
                targetFraction)
        self.history = History(names)
        self.pm = ParameterManager(names, bounds, constraints)
        self.reporter = Reporter(self, complaintCallback)
        self.clear()
        if popsize: self.popsize = popsize
        self.Np = max([
            self.Np_min, min([self.popsize * self.Nd, self.Np_max])])
        self.statusQuoScore = self.targetFraction * self.Np
        abort.callOnAbort(self.abort)

    @classmethod
    def load(cls, filePath, **kw):
        """
        Returns a new instance of me with values initialized from the
        original version that was pickled and written with BZ2
        compression to I{filePath}.

        The pickled version will not have a reference to the
        evaluation I{func} that was supplied to the original version
        in its constructor, nor to any I{complaintCallback}. If you
        want to do further evaluations, you can supply a reference to
        those functions (or even a different one, though that would be
        weird) with the I{func} and I{complaintCallback} keywords.

        @keyword func: Evaluation function, specify if you want to
            resume evaluations. All individuals in the loaded
            population should have their SSEs re-evaluated if anything
            at all has changed about that function.

        @keyword complaintCallback: Callback function for complaining
            about new-best reports during resumed evaluations.

        @keyword bounds: A list of bounds to update my restored
            I{ParameterManager} object with. Specify if you refined
            the parameter bounds since the last run and want to resume
            evaluations with the refined bounds. Each I{Individual} in
            the new instance will have its values limited to the new
            bounds with a call to L{Population.limit}.
        
        @see: L{save} for the way to create compressed pickles of an
            instance of me.
        """
        filePath = os.path.expanduser(filePath)
        with bz2.BZ2File(filePath, 'r') as fh:
            p = pickle.load(fh)
        p.func = kw.get('func', None)
        bounds = kw.get('bounds', None)
        if bounds:
            p.pm.setup(bounds)
            for i in p:
                p.limit(i)
        p.reporter = Reporter(p, kw.get('complaintCallback', None))
        return p
        
    def __getstate__(self):
        """
        For pickling. Note that neither the user-supplied evaluation
        function nor any complaint callback function is included.
        """
        state = {}
        names = {
            # Bools
            'debug', 'running',
            # Scalars
            'Nd', 'Np', 'popsize', 'targetFraction', 'statusQuoScore',
            # Other
            'iList', 'kr', 'pm', 'history',
        }
        for name in names:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.clear()
        for name in state:
            setattr(self, name, state[name])
        if self.running is False: self.running = True
        for i in self.iList:
            i.p = self
            self.dLocks.append(defer.DeferredLock())

    def save(self, filePath):
        """
        Writes a BZ2-compressed pickled version of me to the specified
        I{filePath}.

        Note that the user-supplied evaluation function will not be
        included in the pickled version. However, you can supply it as
        a keyword to L{load}.
        """
        filePath = os.path.expanduser(filePath)
        with bz2.BZ2File(filePath, 'w') as fh:
            pickle.dump(self, fh)
            
    def __getitem__(self, k):
        """
        Sequence-like access to my individuals.
        """
        return self.iList[k]
        
    def __setitem__(self, k, i):
        """
        Use only this method (item setting) to replace individuals in my
        I{iList}.

        The only other place my I{iList} is ever manipulated directly
        is the C{addIndividual} function of L{setup}.
        """
        if not isinstance(i, Individual):
            raise TypeError("You can only set me with Individuals")
        # The history object uses a DeferredLock to ensure that it
        # updates its internals properly, so no need to keep track of
        # the deferreds that get returned from the notInPop and add
        # method calls.
        if len(self.iList) > k:
            iPrev = self.iList[k]
            self.history.notInPop(iPrev)
        self.history.add(i)
        # Here is the only place iList should ever be set directly
        self.iList[k] = i
        # Invalidate sorting
        del self.KS
        
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

    def __nonzero__(self):
        """
        Sequence-like container of individuals: I am C{True} if I have
        any.
        """
        return bool(self.iList)

    @property
    def KS(self):
        """
        Property: A list of indices to I{iList}, sorted by increasing
        (worsening) SSE of the individuals there. The best individual
        will have the first index in I{KS}.
        """
        if self._KS is None and self.iList:
            self._KS = np.argsort([float(i.SSE) for i in self.iList])
        return self._KS
    @KS.deleter
    def KS(self):
        """
        Property: "Deleting" my SSE-sorted list of indices forces
        regeneration of it the next time the I{KS} property is
        accessed. It also "deletes" I{iSorted}.
        """
        self._KS = None
        del self.iSorted
    
    @property
    def iSorted(self):
        """
        Property: A list of my individuals, sorted by increasing
        (worsening) SSE.
        """
        if self._KS is None or self._iSorted is None:
            if self.iList:
                self._iSorted = [self.iList[k] for k in self.KS]
        return self._iSorted
    @iSorted.deleter
    def iSorted(self):
        """
        Property: "Deleting" my sorted list of individuals forces
        regeneration of the sorted list that will be returned next
        time the I{iSorted} property is accessed.
        """
        self._iSorted = None

    @property
    def kBest(self):
        """
        Property: The index to I{iList} of the best individual. C{None} if
        I have no individuals yet.
        """
        if self.KS is not None:
            return self.KS[0]
        
    def __repr__(self):
        """
        An informative string representation with a text table of my best
        individuals.
        """
        def addRow():
            lineParts = ["{:>11s}".format(columns[0]), '|']
            for x in columns[1:]:
                lineParts.append(x)
            lines.append(" ".join(lineParts))

        if not self: return "Population: (empty)"
        N_top = (self.pm.maxLineLength-3) / 15
        iTops = self.iSorted[:N_top]
        if len(iTops) < N_top: N_top = len(iTops)
        SSEs = [float(i.SSE) for i in self]
        lines = [sub(
            "Population: {:d} individuals with SSE {:.5g} to "+\
            "{:.5g}, avg eval time {:.3g} sec. Top {:d}:",
            self.Np, min(SSEs), max(SSEs), np.mean(self.evalTimes()), N_top)]
        lines.append("")
        columns = ["SSE"] + [sub("{:>10.5g}  ", float(i.SSE)) for i in iTops]
        addRow()
        lines.append(self.pm.dashes)
        X = np.empty([self.Nd, N_top])
        for kc, i in enumerate(iTops):
            X[:,kc] = i.values
        for kr, name in self.pm.sortedNamerator():
            columns = [name] + [
                self.pm.stringValue(kr, X[kr,kc], forColumn=True)
                for kc in range(N_top)]
            addRow()
        lines.append(self.pm.dashes)
        lines.append(sub("Best individual:\n{}\n", repr(self.best())))
        return "\n".join(lines)

    def evalFunc(self, values, xSSE=None):
        """
        A wrapper for the user-supplied evaluation function.
        """
        if self.running is False:
            values = None
        if xSSE is None:
            return defer.maybeDeferred(self.func, values)
        return defer.maybeDeferred(self.func, values, xSSE=xSSE)
    
    def clear(self):
        """
        Wipes out any existing population and sets up everything for a
        brand new one.
        """
        self.counter = 0
        self.iList = []
        self.dLocks = []
        if hasattr(self, 'history'): self.history.clear()
        self.running = None
        self.replacementScore = None
        del self.KS
        # This is only here because clear is called by both __init__
        # and __setstate__
        self.ps = ProbabilitySampler()
    
    def limit(self, i):
        """
        Limits the individual's parameter values to the bounds in the way
        that my L{ParameterManager} is configured to do, modifying the
        individual in place.

        B{Note}: The individual's population status is not considered
        or affected. If it's a population member, you will want to
        re-evaluate it and invalidate my sort with a C{del self.KS} or
        C{del self.iSorted} if its SSE has changed.
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

    def initialize(self):
        """
        Invalidates the last sort of my individuals, sets my I{running}
        flag to C{True}, and prints/logs a representation of my populated
        instance.
        """
        del self.KS
        self.running = True
        msg(0, repr(self))
        
    def setup(self, uniform=False, blank=False):
        """
        Sets up my initial population using a Latin hypercube to
        initialize pseudorandom parameter values with minimal clustering.
        
        Unless I{uniform} is set, that is. Then each parameter values
        is just uniformly random without regard to the others.

        With parameter constraints, the Latin hypercube doesn't work
        that well. The initial values matrix must be refreshed,
        perhaps many times. But it may still be better than uniform
        initial population sampling.

        Sets my I{running} flag C{True} and returns a C{Deferred} that
        fires when the population has been set up, with C{True} if
        it's ready to go and setup didn't get aborted.
        
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
            for k in range(self.maxTries):
                values = getNextIV()
                if self.pm.passesConstraints(values):
                    break
                self.showFailedConstraint()
            else:
                msg(0, "Couldn't generate a conforming Individual, aborting!")
                self.abort()
            return Individual(self, self.pm.limit(values))

        def addIndividual(i):
            """
            This is the only place other than L{__setitem__} where my I{iList}
            is manipulated.
            """
            self.iList.append(i)
            self.dLocks.append(defer.DeferredLock())
            self.history.add(i)

        def needMore():
            return len(self.iList) < self.Np
        
        def evaluated(i, d):
            if not i:
                msg(0, "Bogus initial evaluation of {}, aborting", i)
                self.abort()
                return
            self.reporter(i)
            isFinite = not np.isinf(float(i.SSE))
            if isFinite and needMore(): addIndividual(i)

        @defer.inlineCallbacks
        def populate():
            k = 0
            while running() and needMore():
                i = getIndividual()
                if blank:
                    i.SSE = np.inf
                    addIndividual(i)
                    continue
                k += 1
                d = i.evaluate()
                d.addCallback(evaluated, d)
                d.addErrback(oops)
                if k < self.Np:
                    dt.put(d)
                    yield dt.deferUntilFewer(self.N_maxParallel)
                else: yield d
            yield dt.deferToAll()

        def done(null):
            if running():
                self.initialize()
                return True
            
        if not running():
            return defer.succeed(None)
        if self: self.clear()
        dt = DeferredTracker(interval=0.05)
        kIV = [None]*2; refreshIV()
        msg(0, "Initializing {:d} population members having {:d} parameters",
            self.Np, self.Nd, '-')
        return populate().addCallback(done)
        
    def addCallback(self, func, *args, **kw):
        """
        Adds callable I{func} to my reporter's list of functions to call
        each time there is a significantly better L{Individual}.

        @see: L{Reporter.addCallback}.
        """
        self.reporter.addCallback(func, *args, **kw)

    def _keepStatusQuo(self, score):
        """
        Returns C{True} with a probability that increases as I{score}
        approaches my I{statusQuoteScore}.
        """
        x = score / self.statusQuoScore
        if x > 1:
            # Greater than status quo threshold, always remains
            return True
        prob = 0.5 + 0.5*np.sin(np.pi*(x-0.5))
        return np.random.random_sample() < prob
        
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

            I give very little weight to an I{rir} of zero, which
            indicates that the challenger was better but still has an
            equivalent SSE, i.e., is no more than 2% better with the
            default value of I{Reporter.minDiff}. See
            L{Reporter.isEquivSSE}.
        
            I give five times much weight to an I{rir} of 1, though
            it's still pretty small. The improvement is modest and
            could be as little as 2% (assuming
            C{Reporter.minDiff}=0.02, the default). An I{rir} of 2
            gets three times as much weight as that.

            An I{rir} of 3 also gets disproportionately more weight,
            five times as much as I{rir}=1. Beyond that, though, the
            weight scales in a nearly linear fashion. For example, an
            I{rir} of 9 adds just a little more than three times to
            the score (3.4x) as I{rir}=3 does.

            Here's a practical example, with a population of 100
            individuals: If you see 10 "1" characters on the screen
            for one iteration with other 90 being "X," your ratio
            score for that iteration will be 5.0. But if you see just
            one non-X individual with a "8" character, the score will
            be 7.5. That one amazing success story counts more in a
            sea of failures than a bunch of marginal improvements,
            which is kind of how evolution works in real life. (See
            the literature around "hopeful monsters.")
        
        @keyword rir: A rounded improvement ratio obtained from a call
            to L{Reporter.msgRatio}, where the numerator is the SSE of
            the individual that was replaced and the denominator is
            the SSE of its successful challenger.
        
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
            return self._keepStatusQuo(score)
        # An adjustment call
        if self.replacementScore is not None:
            # 0 has a tiny weight, just 0.1
            # 1 has only 0.5 weight
            # 2 has 1.5, or 3x as much as 1
            # 3 has 2.5, or 5x as much as 1
            addition = 0.1 if rir == 0 else rir - 0.5
            self.replacementScore += addition

    def report(self, iNew=None, iOld=None, noProgress=False, force=False):
        """
        Provides a message via the log messenger about the supplied
        L{Individual}, optionally with a comparison to another one.

        If no second individual is supplied, the comparison will be
        with the best individual thus far reported on.

        If no individual at all is supplied, reports on my best one,
        forcing callbacks to run even if the best individual's SSE is
        equivalent to the last-reported one's.
        
        Gets the ratio from a call to my L{Reporter} instance, and
        does a call to L{replacement} with it if the new individual is
        better. Returns (for unit testing convenience) the ratio.

        @keyword noProgress: Set C{True} to suppress printing/logging
            a progress character.

        @keyword force: Set C{True} to force callbacks to run even if
            the reported SSE is considered equivalent to the previous
            best one.

        @see: L{Reporter}.
        """
        if self.running is False: return
        if iNew is None and iOld is None:
            iNew = self.best()
            noProgress = True
            force = True
        ratio = self.reporter(iNew, iOld, noProgress, force)
        if ratio is not None: self.replacement(ratio)
        return ratio

    def waitForReports(self):
        """
        Returns a C{Deferred} that fires when all reporter callbacks have
        finished. (And also L{History} updates.)
        """
        if not self.running:
            return defer.succeed(None)
        return defer.DeferredList([
            self.history.shutdown(), self.reporter.waitForCallbacks()])

    def showFailedConstraint(self):
        """
        Outputs a progress character to indicate a failed constraint.
        """
        self.reporter.progressChar(self.failedConstraintChar)
    
    def push(self, i):
        """
        Pushes the supplied L{Individual} I{i} onto my population and
        kicks out the worst individual there to make room.
        """
        kWorst = self.KS[-1]
        self[kWorst] = i
        del self.KS

    def sample(self, N, *exclude, **kw):
        """
        Returns a sample of I{N} indices from my population that are
        unique from each other and from any excluded indices supplied
        as additional arguments.

        The I{randomBase} keyword lets you use a significant
        improvement offered by ADE: Non-uniform probability of base
        individual selection. Implementation is done by an instance of
        L{ProbabilitySampler}.

        The traditional DE/best/1/bin and DE/rand/1/bin are really
        opposite extremes of what can be a continuous range of base
        individual selection regimes. By specifying a float value for
        I{randomBase} between 0.0 and 1.0, you can select a regime
        anywhere in that range.

        The higher the value, the more uniform the probability
        distribution is. Setting it to near 0.0 makes it much more
        likely that the index of the best individual or one nearly as
        good will be chosen. Setting it to near 1.0 makes the worst
        individual nearly as likely to be chosen as the best.

        A I{randomBase} value of 0.5 is a compromise between
        DE/best/1/bin and DE/rand/1/bin. With that setting, the
        probability of an individual having its index selected will
        gradually drop as it gets worse in the SSE rankings. As
        I{randomBase} goes above 0.5, the probability will take longer
        to start dropping, until at 1.0 it doesn't drop at all. As
        I{randomBase} goes below 0.5, the probability will start
        dropping sooner, until at 0.0 it drops to zero for anything
        but the best individual.

        @keyword randomBase: Sample probability uniformity value
            between 0.0 (only the best individual is ever selected)
            and 1.0 (uniform probability). Setting it I{False} is
            equivalent to 0.0, and setting it I{True} (the default) is
            equivalent to 1.0.
        """
        K = [k for k in self.KS if k not in exclude]
        rb = kw.get('randomBase', True)
        if not rb:
            if N > 1:
                raise ValueError("Can't have > 1 unique best samples!")
            result = [K[0]]
        elif rb in (True, 1.0):
            # Sampling without replacement, so all items of result
            # will be unique
            result = random.sample(K, N)
        elif rb > 1.0:
            raise ValueError(
                "randomBase must be False, True, or between 0.0 and 1.0")
        else:
            result = []
            while len(result) < N:
                k = self.ps(K, rb)
                if k in result: continue
                result.append(k)
        return result[0] if N == 1 else result

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
        if self.iList:
            return self.iList[self.kBest]

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
        
