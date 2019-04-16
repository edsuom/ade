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
A L{Reporter} class for reporting on the SSE of individuals in an
evolving population.
"""

import numpy as np
from twisted.internet import defer

from asynqueue.util import DeferredTracker

from util import *


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
        """
        C{Reporter(population, complaintCallback=None)}
        """
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
        
    def msgRatio(self, iNumerator, iDenominator, bogus_char="!"):
        """
        Returns 0 if I{iNumerator} is better (lower SSE) than
        I{iDenominator}. Otherwise returns the rounded integer ratio
        of numerator SSE divided by denominator SSE.

        I{iNumerator} is automatically better if the SSE of
        I{iDenominator} is bogus, meaning C{None}, C{np.nan}, or
        infinite. That is, unless its own SSE is also bogus, in which
        case a ratio of zero is returned. If I{iNumerator} has a bogus
        SSE but I{iDenominator} does not, a very high ratio is
        returned; a bogus evaluation got successfully challenged, and
        that's very significant.

        I{iNumerator} is not considered better if its SSE is
        "equivalent" to that of I{iDenominator}, meaning that a call
        to L{isEquivSSE} determines that the two individuals have SSEs
        with a fractional difference less than my I{minDiff}
        attribute. For example, if I{iDenominator} has an SSE=100.0,
        returns 1 if I{iNumerator} has an SSE between 101.1 and 149.9.

        If I{iNumerator} has an SSE of 100.9, it is considered
        equivalent to I{iDenominator} and 0 will be returned. If its
        SSE is between 150.0 and 249.9, the return value is 2.

        Logs a progress character:

            - B{?} if either I{iNumerator} or I{iDenominator} evaluates
              as boolean C{False}. (This should only happen if there
              was a fatal error during evaluation and shutdown is
              imminent.)

            - The character supplied with the I{bogus_char} keyword
              (defaults to "!") if I{iNumerator} has a bogus SSE but
              I{iDenominator} does not. (Successful challenge because
              parent had error, or bogus new population candidate.)

            - B{%} if I{iNumerator} has a bogus SSE and so does
              I{iDenominator}. (Everybody's fucked up.)

            - B{#} if I{iDenominator} has a bogus SSE but I{iNumerator}
              does not. (Failed challenge due to error.)

            - B{X} if I{iNumerator} is better than I{iDenominator},
              indicating a failed challenge.
        
            - The digit B{0} if I{iNumerator} is worse than
              I{iDenominator} (challenger) but with an equivalent SSE.

            - A digit from 1-9 if I{iNumerator} is worse than
              I{iDenominator} (challenger), with the digit indicating
              how much better (lower) the SSE of I{iDenominator} is
              than that of I{iNumerator}. (A digit of "9" only
              indicates that the ratio was at least nine and might be
              much higher.)
        """
        def bogus(i):
            SSE = i.SSE
            if SSE is None: return True
            try: isNan = np.isnan(SSE)
            except: isNan = False
            return isNan or np.isinf(SSE)
        
        if not iNumerator or not iDenominator:
            ratio = 0
            sym = "?"
        elif bogus(iNumerator):
            if bogus(iDenominator):
                ratio = 0
                sym = "%"
            else:
                ratio = 1000
                sym = bogus_char
        elif bogus(iDenominator):
            ratio = 0
            sym = "#"
        elif iNumerator < iDenominator:
            ratio = 0
            sym = "X"
        elif self.isEquivSSE(iDenominator, iNumerator):
            ratio = 0
            sym = "0"
        else:
            if not iDenominator.SSE:
                ratio = 1000
            else:
                ratio = np.round(
                    float(iNumerator.SSE) / float(iDenominator.SSE))
            sym = str(int(ratio)) if ratio < 10 else "9"
        self.progressChar(sym)
        return ratio

    def progressChar(self, sym=None):
        """
        Logs the supplied ASCII character I{sym} to the current
        console/log line.

        If the number of symbols logged to the current line reaches my
        line-length limit, a newline is inserted. To reset the count
        of symbols logged to the current line, call this method with
        no symbol provided.
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
                result = self.msgRatio(i, self.iBest, bogus_char="#")
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
