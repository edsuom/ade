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
The DifferentialEvolution class and helpers.
"""

import random
from copy import copy

import numpy as np
from pyDOE import lhs
from scipy import stats
from twisted.internet import defer, task, reactor

from population import Population
from util import *


class Result:
    pass


class FManager(object):
    """
    I manage the mutation coefficient I{F} for adaptive differential
    evolution.

    If I{adaptive} is set C{True}, adaptive mode is used. My attribute
    I{scale} is the amount to scale F down by with each call to
    L{down}. A single F value will shrink slower than the lower end of
    a uniform random variate range.

    If you want the lower bound of F to be the critical value for my
    I{CR} and I{Np}, set it to zero, like this: C{(0, 1)}. I will not
    run in adaptive mode in that case, ignoring whatever you set
    I{is_adaptive} to.

    My I{limited} attribute gets set C{True} when a call to L{down}
    failed to reduce F anymore, and gets reset to C{False} if and when
    L{up} gets called thereafter.
    """
    scaleLowest = 0.89
    scaleHighest = 0.87
    scaleSingle = 0.89
    scaleUpPower = 0.88

    minHighestVsLowest = 3
    
    def __init__(self, F, CR, Np, adaptive=False):
        self.criticalF = np.sqrt((1.0 - 0.5*CR) / Np)
        self.is_sequence = hasattr(F, '__iter__')
        self.is_adaptive = adaptive
        if self.is_sequence:
            self.F = list(F)
            if self.F[0] == 0:
                self.is_adaptive = False
                self.F[0] = self.criticalF
            self.scales = (self.scaleLowest, self.scaleHighest)
        else:
            self.F = F
            self.scales = [self.scaleSingle]*2
        self.origLowest = self.lowest
        self.origHighest = self.highest
        self.limited = False
    
    def __repr__(self):
        if self.is_sequence:
            return sub("U({})", ", ".join([str(x) for x in self.F]))
        return str(self.F)

    @property
    def lowest(self):
        return self.F[0] if self.is_sequence else self.F
    @lowest.setter
    def lowest(self, value):
        if value > self.origLowest:
            value = self.origLowest
        if self.is_sequence:
            self.F[0] = value
        else:
            self.F = value

    @property
    def highest(self):
        return self.F[1] if self.is_sequence else self.F
    @highest.setter
    def highest(self, value):
        if value > self.origHighest:
            value = self.origHighest
        if self.is_sequence:
            self.F[1] = value
        else:
            self.F = value
            
    def get(self):
        """
        Returns I{F} if it is a single coefficient, or a uniform variate
        in U(F[0], F[1]) otherwise.

        If I am running in adaptive mode, returns the adapted value of
        I{F}. Otherwise, always returns the original value you
        provided to my constructor and the adapting is done just to
        determine if my (hidden) I{F} value has reached its lower
        limit.
        """
        if self.is_sequence:
            return random.uniform(*self.F)
        return self.F
    
    def down(self):
        """
        Adapts I{F} downward. Call this when none of your challengers are
        winning their tournaments.

        If I{F} is a range for uniform variates, the lower limit is
        what is adjusted downward. If the lower limit reaches the
        critical value, then the upper limit begins to adjust downward
        as well, keeping it above the lower limit.

        The critical value was defined by Zaharie, as described in Das
        and Suganthan, "Differential Evolution: A Survey of the
        State-of-the-Art," IEEE Transactions on Evolutionary
        Computation, Vol. 15, No. 1, Feb. 2011.
        """
        if not self.is_adaptive:
            return
        lowest = self.lowest
        proposedF = self.scales[0] * lowest
        if proposedF > self.criticalF:
            self.lowest = proposedF
            return
        if self.is_sequence:
            proposedF = self.scales[1] * self.highest
            if proposedF > self.minHighestVsLowest * lowest:
                self.highest = proposedF
                return
        self.limited = True
    
    def up(self):
        """
        Adapt I{F} upward. Call this when at least one of your challengers
        has won its tournament.

        If I{F} is a range for uniform variates, the lower limit is
        what is adjusted upward, and the upper limit is returned to
        its original value.
        """
        def scaleUp(x, scale, upperLimit):
            return min([x*(scale**-self.scaleUpPower), upperLimit])

        if not self.is_adaptive:
            return
        self.limited = False
        highest = self.highest
        if self.is_sequence and highest < self.origHighest:
            self.highest = scaleUp(highest, self.scales[1], self.origHighest)
            return
        lowest = self.lowest
        if lowest < self.origLowest:
            self.lowest = scaleUp(lowest, self.scales[0], self.origLowest)

                
class DifferentialEvolution(object):
    """
    I perform asynchronous differential evolution, employing your
    multiple CPU cores in a very cool efficient way that does not
    change anything about the actual operations done in the DE
    algorithm. The very same target selection, mutation, scaling, and
    recombination (crossover) will be done for a sequence of each
    target in the population, just as it is with a DE algorithm that
    blocks during fitness evaluation. The magic lies in the use of
    C{DeferredLock} instances for each index of the population
    list. Because the number of population members is usually far
    greater than the number of CPU cores available, almost all of the
    time the asynchronous processing will find a target it can work on
    without disturbing the operation sequence.

    Construct me with a L{population.Population} instance and any
    keywords that set my runtime configuration different than my
    default I{attributes}. The Population object will need to be
    initialized with a population of L{individual.Individual} objects
    that can be evaluated according to the population object's
    evaluation function, which must return a fitness metric where
    lower values indicate better fitness.
    """
    attributes = {
        'CR':           0.8,
        'F':           (0.5, 1.0),
        'maxiter':      500,
        'randomBase':   False,
        'uniform':      False,
        'adaptive':     True,
        'bitterEnd':    False,
        'withPDF':      True,
        'dwellByGrave': 7,
    }

    def __init__(self, population, **kw):
        self.p = population
        self.p.reporter()
        for name in self.attributes:
            value = kw.get(name, getattr(self, name, None))
            if value is None:
                value = self.attributes[name]
            setattr(self, name, value)
        if self.CR < 0.0 or self.CR > 1.0:
            raise ValueError(sub("Invalid crossover constant {}", self.CR))
        self.fm = FManager(self.F, self.CR, self.p.Np, self.adaptive)
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)
        self.stopRunning = False

    def shutdown(self):
        self.stopRunning = True
        if hasattr(self, 'triggerID'):
            reactor.removeSystemEventTrigger(self.triggerID)
            del self.triggerID
        
    def crossover(self, parent, mutant):
        j = random.randint(0, self.p.Nd-1)
        for k in range(self.p.Nd):
            if k == j:
                # The mutant gets to keep at least this one value
                continue
            if self.CR < random.uniform(0, 1.0):
                # CR is probability of the mutant's value being
                # used. Only if U[0,1] random variate is bigger (as it
                # seldom will be with typical CR ~ 0.9), is the mutant's
                # value discarded and the parent's used.
                mutant[k] = parent[k]

    @defer.inlineCallbacks
    def challenge(self, kt, kb):
        """
        Challenges the target ("parent") individual at index I{kt} with a
        challenger (often referred to as a "trial" or "child")
        individual produced from DE mutation and crossover. The trial
        individual is formed from crossover between the target and a
        donor individual, which is formed from the vector sum of a
        base individual at index I{kb} and a scaled vector difference
        between two randomly chosen other individuals that are
        distinct from each other and both the target and base
        individuals::

          id = ib + F*(i0 - i1)         [1]

          ic = crossover(it, id)        [2]

        First, I calculate the vector difference between the two other
        individuals. Then I scale that difference by I{F}, the
        current, possibly random, possibly population-dependent value
        of which is obtained with a call to the C{get} method of my
        L{FManager}. Then I add the scaled difference to the donor
        base individual at I{kb} and perform crossover to obtain the
        donor.

        The crossover of [2], the "bin" in C{DE/[rand|best]/1/bin},
        consists of giving each parameter of the donor individual a
        chance (usually a very good chance) to appear in the
        challenger, as opposed to using the target's parameter. For
        each parameter, if a uniform random number in the range of 0-1
        is less than my attribute I{CR}, I use the donor's version of
        that parameter and thus preserve the mutation performed in
        [1]. Otherwise, I use the target's version and the discard the
        mutation for that parameter.

        Finally, I conduct a tournament between the target and the
        challenger. Whoever has the lowest result of a call to
        L{Individual.evaluate} is the winner and is assigned to index
        I{kt}.
        """
        if not self.stopRunning:
            k0, k1 = self.p.sample(2, kt, kb)
            # Await legit values for all individuals used here
            yield self.p.lock(kt, kb, k0, k1)
            if not self.stopRunning:
                iTarget, iBase, i0, i1 = self.p.individuals(kt, kb, k0, k1)
                # All but the target can be released right away,
                # because we are only using their local values here
                # and don't care if someone else changes them at this
                # point. The target can't be released yet because its
                # value might change and someone waiting on its result
                # will need the accurate one
                self.p.release(kb, k0, k1)
                # Do the mutation and crossover with unbounded values
                iChallenger = iBase + (i0 - i1) * self.fm.get()
                self.crossover(iTarget, iChallenger)
                # Continue with Pr(values) / Pr(midpoints)
                self.p.limit(iChallenger)
                if self.p.pm.passesConstraints(iChallenger.values):
                    # Passes constraints!
                    # Now the hard part: Evaluate fitness of the challenger
                    yield iChallenger.evaluate(xSSE=iTarget.SSE)
                    if iChallenger < iTarget:
                        # The challenger won the tournament, replace
                        # the target
                        self.p[kt] = iChallenger
                    if iChallenger:
                        self.p.report(iChallenger, iTarget)
                # Now that the individual at the target index has been
                # determined, we can finally release the lock for that index
                self.p.release(kt)
            
    @defer.inlineCallbacks
    def __call__(self):
        """
        Call this to run differential evolution on a population of
        individuals.

        At the conclusion of each generation's evaluations, I consider
        the amount of overall improvement if I am running in adaptive
        mode. If the overall improvement (sum of rounded ratios
        between SSE of replaced individuals and their replacements)
        exceeded that required to maintain the status quo, I bump up F
        a bit. If the overall improvement did not meet that threshold,
        I reduce F a bit, but only if there was no new best individual
        in the population.

        So long as the best individual in the population keeps getting
        better with each generation, I will continue to run, even with
        tiny overall improvements.
        """
        self.dwellCount = 0
        desc = sub("DE/{}/1/bin", "rand" if self.randomBase else "best")
        msg("Performing DE with CR={}, F={}, {}", self.CR, self.fm, desc, '-')
        # Evolve!
        for kg in range(self.maxiter):
            F_info = sub(" F={}", self.fm)
            msg(-1, "Generation {:d}/{:d} {}", kg+1, self.maxiter, F_info , '-')
            yield self.p.waitForReports()
            dList = []
            iBest = self.p.best()
            for kt in range(self.p.Np):
                if self.stopRunning:
                    break
                if self.randomBase or kt == self.p.kBest:
                    kb = self.p.sample(1, kt)
                else:
                    kb = self.p.kBest
                d = self.challenge(kt, kb).addErrback(oops)
                dList.append(d)
            else:
                yield defer.DeferredList(dList)
            if self.stopRunning:
                break
            if self.p.replacement():
                # There was enough overall improvement to warrant
                # scaling F back up a bit
                self.fm.up()
                self.dwellCount = 0
            else:
                # There was not enough overall improvement maintain the status
                # quo, so scale F down a bit
                self.fm.down()
                if not self.bitterEnd and self.fm.limited:
                    self.dwellCount += 1
                    if self.dwellCount > self.dwellByGrave:
                        msg(-1, "Challengers failing too much, stopped")
                        break
        else:
            msg(-1, "Maximum number of iterations reached")
        self.p.report()
        yield self.p.waitForReports()
        defer.returnValue(self.p)
