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
L{DifferentialEvolution} and support staff.

When an instance of C{DifferentialEvolution} is run on the command
line in a console, pressing the Enter key will cause it to run
L{DifferentialEvolution.shutdown} and quit running.
"""

import signal, random, sys
from copy import copy

import numpy as np
from pyDOE import lhs
from scipy import stats

from twisted.internet import defer, task, reactor, stdio, protocol

import abort
from population import Population
from util import *

# This is kind of hackish and ugly, but all the @defer.inlineCallbacks
# action can involve some pretty deep recursion
sys.setrecursionlimit(max([10000, sys.getrecursionlimit()]))


class FManager(object):
    """
    I manage the mutation coefficient I{F} for adaptive differential
    evolution.

    L{Population} constructs an instance of me with an initial value
    (or range of values) for I{F}, the crossover probability I{CR},
    and the population size I{Np}.
    
    If you want the lower bound of F to be the critical value for my
    I{CR} and I{Np}, set it to zero, like this: C{(0, 1)}. I will not
    run in adaptive mode in that case, ignoring the keyword setting.

    @ivar limited: Gets set C{True} when a call to L{down} failed to
        reduce F anymore, and gets reset to C{False} if and when L{up}
        gets called thereafter.
    
    @keyword adaptive: Set C{True} to use adaptive mode. My attribute
        I{scale} is the amount to scale F down by with each call to
        L{down}. A single F value will shrink slower than the lower
        end of a uniform random variate range.
    """
    scaleLowest = 0.89
    scaleHighest = 0.87
    scaleSingle = 0.89
    scaleUpPower = 0.88

    minHighestVsLowest = 3
    
    def __init__(self, F, CR, Np, adaptive=False):
        """FManager(F, CR, Np, adaptive=False)"""
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
        """
        My string representation is just my I{F} value or range.
        """
        if self.is_sequence:
            return sub("U({})", ", ".join([str(x) for x in self.F]))
        return str(self.F)

    @property
    def lowest(self):
        """
        Property: The lower bound or sole value of I{F}.
        """
        return self.F[0] if self.is_sequence else self.F
    @lowest.setter
    def lowest(self, value):
        """
        Property setter for the lower bound or sole value of I{F}.
        """
        if value > self.origLowest:
            value = self.origLowest
        if self.is_sequence:
            self.F[0] = value
        else:
            self.F = value

    @property
    def highest(self):
        """
        Property: The upper bound or sole value of I{F}.
        """
        return self.F[1] if self.is_sequence else self.F
    @highest.setter
    def highest(self, value):
        """
        Property setter for the upper bound or sole value of I{F}.
        """
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
    B{A}synchronous B{D}ifferential B{E}volution: The foundational
    object.

    I perform differential evolution asynchronously, employing your
    multiple CPU cores or CPUs in a very cool efficient way that does
    not change anything about the actual operations done in the DE
    algorithm. The very same target selection, mutation, scaling, and
    recombination (crossover) will be done for a sequence of each
    target in the population, just as it is with a DE algorithm that
    blocks during fitness evaluation.

    Lots of Lock
    ============
    
        My magic lies in the use of Twisted's C{DeferredLock}. There's
        one for each index of the population list. Because the number
        of population members is usually far greater than the number
        of CPU cores available, almost all of the time the
        asynchronous processing will find a target it can work on
        without disturbing the operation sequence.

    Population & Individuals
    ========================
    
        Construct me with a L{Population} instance and any keywords
        that set my runtime configuration different than my default
        I{attributes}. Before running my instance with L{__call__},
        you must call the L{Population.setup} method. That initializes
        it with a population of L{Individual} objects that can be
        evaluated according to the population object's evaluation
        function.
    
        The evaluation function must return a fitness metric where
        lower values indicate better fitness, C{None} or C{inf}
        represents an invalid or failing individual and thus
        worst-possible fitness, and a negative number represents a
        fatal error that will terminate operations. It must except
        either a 1-D Numpy array of parameter values or, if I am
        shutting down, a C{None} object.

    Darwin, Interrupted
    ===================
    
        When running in a console Python application, my L{shutdown}
        method gets called when the Enter key is pressed. It took
        quite a bit of work to implement that user-abort capability in
        a clean, Twisted-friendly manner, but it was worth it. Serious
        evolution of stuff with DE involves a lot of observing the
        distributions of parameter values vs SSE, stopping evolution,
        tweaking the parameter ranges, and resuming evolution again.

    Crossover Rate
    ==============
    
        My I{CR} attribute determines the B{c}rossover B{r}ate, the
        probability of the mutant's value being used for a given
        parameter instead of just copying the parent's value. A low
        I{CR} means less mutation and thus innovation. But the less
        drastic, lower-dimensional movement in the search space that a
        low I{CR} results in ultimately may be more productive.
    
        Montgomery & Chen, "An Analysis of the Operation of
        Differential Evolution at High and Low Crossover Rates"
        (2010): "Small values of I{CR} result in exploratory moves
        parallel to a small number of axes of the search space, while
        large values of I{CR} produce moves at angles to the search
        space’s axes. Consequently, the general consensus, supported
        by some empirical studies, is that small I{CR} is useful when
        solving separable problems while large I{CR} is more useful
        when solving non-separable problems."
    
        Despite the simplicity of I{CR} being proportional to
        exploration dimensionality, selecting a value for I{CR} is not
        terribly intuitive. Montgomery & Chen show that, for some
        well-known competition problems, performance is best when CR
        is near but not at either extreme of 0.0 or 1.0. The ranges
        0.1-0.2 and 0.8-0.9 look promising. They note that
        "characteristics and convergence rates are all highly
        different" at each end of the overall I{CR} range: "While DE
        with I{CR} = 0.9 relies on the population converging so that
        its moves may be scaled for finer-grained search, DE with
        I{CR} ≤ 0.1 maintains a highly diverse population throughout
        its course, especially in complex landscapes, as individual
        solutions conduct largely independent searches of the solution
        space."

    @cvar attributes: Default values for attributes I{CR}, I{F},
        I{maxiter}, I{randomBase}, I{uniform}, I{adaptive},
        I{bitterEnd}, and I{dwellByGrave} that define how a
        Differential Evolution run should be conducted. The attributes
        are set by my constructor, and the defaults can be overridden
        with constructor keywords. (That's why they're listed here as
        keywords.)
    @type attributes: dict

    @ivar p: An instance of L{Population} supplied as my first
        constructor argument.

    @ivar fm: An instance of L{FManager}.

    @ivar running: C{True} unless my L{shutdown} method has been called.

    @keyword logHandle: An open handle for a log file, or C{True} to
        log output to STDOUT or C{None} to suppress logging. Default
        is STDOUT.

    @keyword CR: The I{crossover rate} between parent (i.e., basis)
        and mutant (i.e., candidate, offspring). CR is the probability
        of the mutant's value being used for a given parameter. Only
        if a U[0,1] random variate is bigger than CR (as it seldom
        will be with typical CR around 0.8), and only if the parameter
        is not a reserved random one that B{must} be mutated, is the
        mutant's value discarded and the parent's used.
    @type CR: float

    @keyword F: A scalar or two-element sequence defining the
        I{differential weight}, or a range of possible weights from
        which one is obtained as a uniform random variate.

    @keyword maxiter: The maximum number of iterations (i.e.,
        generations) to run. It can be useful to set this to something
        realistic (e.g., 500 for big populations and lengthy
        evaluations) so that you have a nice summary of the best
        candidates when you come back and check results in an hour or
        so.

    @keyword randomBase: Set C{True} to use DE/rand/1/bin where a
        random individual his chosen from the L{Population} instead of
        the current best individual as the basis for mutants. Or set
        it to a float between 0.0 and 1.0 to use ADE's modified
        version DE/prob/1/bin where the probability of an individual
        being chosen increases with how close it is to being the best
        one in the population; the higher the number, the closer to
        uniformly random that probability will be.

    @keyword uniform: Set C{True} to initialize the population with
        uniform random variate's as the parameters instead of a Latin
        hypercube. Not usually recommended because you don't want to
        start off with clustered parameters.

    @keyword adaptive: Set C{True} to adapt the value (or values) of
        I{F} in a way that tries to maintain the number of successful
        challenges at a reasonable level. The adaptive algorithm
        accounts not just for whether a challenge succeeded but also
        how much better the challenger is than the individual it
        replaced.

    @keyword bitterEnd: Normally, I{ade} quits trying once there are
        too few successful challenges and it appears that further
        iterations won't accomplish much. Set this C{True} if you have
        all the time in the world and wanted to keep going until
        I{maxiter}.

    @keyword dwellByGrave: The number of iterations that I{ade} hangs
        around after it's decided that nothing more is being
        accomplished. if you think there really is some progress being
        with occasional marginally better replacements but don't want
        to go until the I{bitterEnd}, feel free to increase this from
        the default. Be aware that increasing it too much, e.g., to
        40, can effectively force the iterations to continue until the
        I{bitterEnd}, because a single adaptive increase in I{F} will
        reset the count and you'll need all those continuous
        no-progress iterations to happen all over again for it to
        quit.

    @keyword goalSSE: You can set a goal for SSE to indicate that any
        further iterations are pointless if that goal is reached.  If
        defined and the best individual has a better SSE than it at
        the end of an iteration, there will be no further iterations.

    @keyword xSSE: Set C{True} if your evaluation function can accept
        and make use of an I{xSSE} keyword defining an SSE value above
        which continuing the evaluation is pointless. If this is set,
        each call to the eval function for a challenger will include
        the I{xSSE} keyword set to its target's SSE. If the
        challenger's SSE exceeds I{xSSE}, the evaluation can terminate
        early because the challenge will fail no matter what.
    """
    attributes = {
        'CR':           0.8,
        'F':           (0.5, 1.0),
        'maxiter':      500,
        'randomBase':   False,
        'uniform':      False,
        'adaptive':     True,
        'bitterEnd':    False,
        'dwellByGrave': 5,
        'goalSSE':      None,
        'xSSE':         False,
    }

    def __init__(self, population, **kw):
        """C{DifferentialEvolution(population, **kw)}"""
        self.p = population
        # Log to an open file handle if provided (no logging if file
        # handle is None), otherwise to STDOUT
        fh = kw['logHandle'] if 'logHandle' in kw else True
        msg(fh)
        for name in self.attributes:
            value = kw.get(name, getattr(self, name, None))
            if value is None: value = self.attributes[name]
            setattr(self, name, value)
        if self.CR < 0.0 or self.CR > 1.0:
            raise ValueError(sub("Invalid crossover constant {}", self.CR))
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)
        self.running = True
        self.dChallenges = None
        abort.callOnAbort(self.shutdown)

    def shutdown(self):
        """
        Call this to shut me down gracefully.

        Repeated calls are ignored. Gets called when the Enter key is
        pressed.
        
        Sets my I{running} flag C{False}, which lets all my various
        loops know that it's time to quit early. Calls
        L{Population.abort} on my L{Population} object I{p} to shut it
        down ASAP.
        """
        if self.triggerID:
            reactor.removeSystemEventTrigger(self.triggerID)
            self.triggerID = None
        if self.running:
            self.running = False
            msg(0, "Shutting down DE...")
            if self.dChallenges and not self.dChallenges.called:
                self.dChallenges.errback(abort.AbortError())
        
    def crossover(self, parent, mutant):
        """
        Crossover of I{parent} and I{mutant} individuals.

        The probability of the mutant keeping any given value is
        I{CR}, except for a randomly chosen one that it always gets to
        keep.
        """
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
        challenger at index I{kb}.

        The challenger, often referred to as a "trial" or "child," is
        an L{Individual} that was produced from DE mutation and
        L{crossover}.

        The trial individual is formed from crossover between the
        target and a donor individual, which is formed from the vector
        sum of a base individual at index I{kb} and a scaled vector
        difference between two randomly chosen other individuals that
        are distinct from each other and both the target and base
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

        The "returned" C{Deferred} fires with C{None}.
        """
        if self.running:
            # Indices of two randomly chosen unique individuals that
            # are not at index kt or kb.
            k0, k1 = self.p.sample(2, kt, kb)
            # Await legit values for all individuals used here
            yield self.p.lock(kt, kb, k0, k1)
            if self.running:
                sample = self.p.individuals(kt, kb, k0, k1)
            else: sample = None
            if sample is None:            
                # Shutting down!
                self.p.release()
            else:
                iTarget, iBase, i0, i1 = sample
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
                    if self.xSSE:
                        yield iChallenger.evaluate(iTarget.SSE)
                    else: yield iChallenger.evaluate()
                    if iChallenger and self.running:
                        if iChallenger < iTarget:
                            # The challenger won the tournament, replace
                            # the target
                            self.p[kt] = iChallenger
                        elif not self.xSSE:
                            # Not doing partial evaluations with xSSE,
                            # so can make some use of failed challenge
                            # by recording its SSE in history
                            yield self.p.history.add(
                                iChallenger, neverInPop=True)
                        self.p.report(iChallenger, iTarget)
                    else:
                        # Oops! Fatal error occurred!
                        self.shutdown()
                else:
                    # Failed constraints
                    self.p.showFailedConstraint()
                # Now that the individual at the target index has been
                # determined, we can finally release the lock for that
                # index
                self.p.release(kt)
    
    @defer.inlineCallbacks
    def _run(self, func):
        """
        Called by L{__call__} to do most of the work. I{func} is a
        callback function that gets called after each generation with
        the generation number as the sole argument.

        Returns a C{Deferred} that fires with my L{Population} object
        I{p} when the DE run is completed.
        """
        def failed(failureObj):
            if failureObj.type == abort.AbortError:
                return
            info = failureObj.getTraceback()
            msg(-1, "Error during challenges:\n{}\n{}\n", '-'*40, info)

        # Evolve!
        for kg in range(self.maxiter):
            self.p.reporter.progressChar()
            info = sub("  F={}  N_hist={:d}", self.fm, self.p.history.N_total)
            msg(-1, "Generation {:d}/{:d}{}", kg+1, self.maxiter, info , '-')
            yield self.p.waitForReports()
            if not self.running: break
            dList = []
            iBest = self.p.best()
            for kt in range(self.p.Np):
                if self.randomBase or kt == self.p.kBest:
                    kb = self.p.sample(1, kt, randomBase=self.randomBase)
                else: kb = self.p.kBest
                if kb is None:
                    # We must be shutting down, abort loop now
                    self.shutdown()
                    break
                d = self.challenge(kt, kb)
                dList.append(d)
                if not self.running: break
            else:
                # Only wait for the challenges if there was no abort,
                # and abort if any challenge results in a failure
                self.dChallenges = defer.DeferredList(
                    dList, fireOnOneErrback=True).addErrback(failed)
                result = yield self.dChallenges
                self.dChallenges = None
                if result is None:
                    self.shutdown()
            if not self.running: break
            if self.goalSSE and self.p.best < self.goalSSE:
                msg(-1,
                    "Goal SSE of {:.5g} has been met, stopped.", self.goalSSE)
                break
            elif self.p.replacement():
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
                        msg(-1, "Challengers failing too much, stopped.")
                        break
            if func: func(kg)
        else: msg(-1, "Maximum number of iterations reached.")
        if self.running:
            # File report for best individual and shutdown
            self.p.report()
            yield self.p.waitForReports()
            self.shutdown()
        # "Return" value is the population object
        msg("DE shutdown complete.")
        defer.returnValue(self.p)

    def __call__(self, func=None):
        """
        Here is what you call to run differential evolution on my
        L{Population} I{p} of individuals.

        You have to construct me with the population object, and you
        have to run L{Population.setup} on it yourself. Make sure
        that's been done and the resulting C{Deferred} has fired
        before trying to call this.

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

        Returns a C{Deferred} that fires with my L{Population} object
        I{p} when the DE run is completed.

        @keyword func: Supply a callback function to have it called
            after each generation, with the generation number as the
            sole argument.
        """
        def ready(null):
            msg("Press the 'Enter' key to abort.")
            return self._run(func)
        
        if self.p.running is False:
            # Population setup got aborted
            return defer.succeed(self.p)
        self.dwellCount = 0
        self.fm = FManager(self.F, self.CR, self.p.Np, self.adaptive)
        desc = sub("DE/{}/1/bin", sub(
            "rand-{:.2f}", self.randomBase) if self.randomBase else "best")
        msg("Performing DE with CR={}, F={}, {}", self.CR, self.fm, desc, '-')
        self.p.report()
        return self.p.waitForReports().addCallback(ready)
