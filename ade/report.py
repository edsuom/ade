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

        def aborted():
            return self.p.running is False
        
        @defer.inlineCallbacks
        def runUntilFree():
            counter = self.p.counter
            self.cbrInProgress = True
            msg.lineWritten()
            iPrev = None
            while not aborted() and self.cbrScheduled:
                i = self.cbrScheduled.pop()
                if iPrev and i == iPrev:
                    continue
                iPrev = i
                for func, args, kw in self.callbacks:
                    if aborted(): break
                    d = defer.maybeDeferred(
                        func, i.values, counter, i.SSE, *args, **kw)
                    d.addErrback(failed, func)
                    result = yield d
                    if isinstance(result, CallbackFailureToken):
                        self.abort()
                        break
                    if result is not None and i and not aborted():
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

    def newBest(self, i, force=False):
        """
        Registers and reports a new best L{Individual}. Calling this
        method is the only safe way to update I{iBest}.

        If the new individual's SSE is not merely equivalent to the
        last one (or if I{force} is set C{True}), triggers a run of
        callbacks with the best individual's 1-D array of I{values}
        and the integer report count for any callbacks that have been
        registered via my L{addCallback} method. The primary use of
        this is displaying a plot of the current best set of
        parameters.

        If a callback run is currently in progress, another will be
        done as soon as the current one is finished. That next run
        will refer to whatever is the best individual at the time. A
        rapid flurry of calls to L{newBest} will only callbacks
        necessary to ensure that the best individual has been
        referenced in a callback run.

        @keyword force: Set C{True} to force I{i} to be considered the
            best, even if it's not. B{Caution}: Don't use unless
            you're certain I{i} is better than (or the same as) my
            I{iBest}.
        """
        def goAhead():
            """
            Only follow through with the report if this returns C{True}.
            """
            if self.iLastReported and i == self.iLastReported:
                # Don't just repeat the exact same report, ever.
                return False
            if force:
                # Unless the report would be an exact duplicate,
                # always proceed if forced.
                return True
            if not isBest:
                # Not best so nope.
                return False
            # Best and not forced, so reporting depends on whether SSE
            # is considered equivalent.
            return not self.isEquivSSE(i, self.iLastReported)
        
        if self.iBest is None or i < self.iBest:
            # Register new best, regardless of whether it gets reported
            self.iBest = i
            isBest = True
        else: isBest = False
        if goAhead():
            if self.p.debug:
                msg(0, "{}\n\t--->\n{}\n", self.iBest, i)
            self.iLastReported = i
            iBackupBest = self.iBest
            self.runCallbacks(i, iBackupBest)
        
    def msgRatio(self, iNum, iDenom, bogus_char="!", noProgress=False):
        """
        Returns C{None} if I{iNum} is better (lower SSE) than
        I{iDenom}. Otherwise returns the rounded integer ratio of
        numerator SSE divided by denominator SSE.

        I{iNum} is automatically better if the SSE of I{iDenom} is
        bogus, meaning C{None}, C{np.nan}, or infinite. That is,
        unless its own SSE is also bogus, in which case a ratio of
        zero is returned. If I{iNum} has a bogus SSE but I{iDenom}
        does not, a very high ratio is returned; a bogus evaluation
        got successfully challenged, and that's very significant.

        I{iNum} is not considered better if its SSE is "equivalent" to
        that of I{iDenom}, meaning that a call to L{isEquivSSE}
        determines that the two individuals have SSEs with a
        fractional difference less than my I{minDiff} attribute. For
        example, if I{iDenom} has an SSE=100.0, returns 1 if I{iNum}
        has an SSE between 101.1 and 149.9. If its SSE is between
        150.0 and 249.9, the return value will be 2.

        A slightly better but "equivalent" I{iDenom} results in a
        return value of 0. (Not C{None}.) With an I{iDenom} SSE of
        100.0, and I{iNum} SSE of 100.9, is considered equivalent to
        I{iDenom} and 0 will be returned. 

        Logs a progress character:

            - B{?} if either I{iNum} or I{iDenom} evaluates as boolean
              C{False}. (This should only happen if there was a fatal
              error during evaluation and shutdown is imminent.)

            - The character supplied with the I{bogus_char} keyword
              (defaults to "!") if I{iNum} has a bogus SSE but
              I{iDenom} does not. (Successful challenge because parent
              had error, or bogus new population candidate.)

            - B{%} if I{iNum} has a bogus SSE and so does
              I{iDenom}. (Everybody's fucked up.)

            - B{#} if I{iDenom} has a bogus SSE but I{iNum} does
              not. (Failed challenge due to error.)

            - B{X} if I{iNum} is better than I{iDenom}, indicating a
              failed challenge.
        
            - The digit B{0} if I{iNum} is worse than I{iDenom}
              (challenger) but with an equivalent SSE.

            - A digit from 1-9 if I{iNum} is worse than I{iDenom}
              (challenger), with the digit indicating how much better
              (lower) the SSE of I{iDenom} is than that of I{iNum}. (A
              digit of "9" only indicates that the ratio was at least
              nine and might be much higher.)
        """
        def bogus(i):
            SSE = i.SSE
            if SSE is None: return True
            try:
                isNan = np.isnan(SSE)
            except: isNan = False
            if isNan: return True
            try:
                isInf = np.isinf(SSE)
            except: isInf = False
            return isInf
        
        if not iNum or not iDenom:
            ratio = None
            sym = "?"
        elif bogus(iNum):
            if bogus(iDenom):
                ratio = 0
                sym = "%"
            else:
                ratio = 1000
                sym = bogus_char
        elif bogus(iDenom):
            ratio = 0
            sym = "#"
        elif iNum < iDenom:
            ratio = None
            sym = "X"
        elif self.isEquivSSE(iDenom, iNum):
            ratio = 0
            sym = "0"
        else:
            if not iDenom.SSE:
                ratio = 1000
            else:
                ratio = np.round(
                    float(iNum.SSE) / float(iDenom.SSE))
            sym = str(int(ratio)) if ratio < 10 else "9"
        if not noProgress: self.progressChar(sym)
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

    def __call__(self, i, iOther=None, noProgress=False, force=False):
        """
        Files a report on the L{Individual} I{i}, perhaps vs. another
        individual I{iOther}.

        The type of report depends on whether one or two individuals
        are supplied. In any event, if the report results in my
        I{iBest} being replaced with a better individual, I will run
        any callbacks registered with me.

        One individual
        ==============
        
            With just I{i} supplied and not I{iOther}, the report is a
            lower-is-better SSE comparison between I{i} and my current
            best individual I{iBest}.
    
            If there isn't yet a best individual, then I{i}
            automatically becomes my first I{iBest} and a progress
            character of "*" is printed or logged.
    
            If there already is a best individual, I{i} only takes
            over as I{iBest} if it has a lower SSE, in which case the
            progress character is "!". Otherwise the progress
            character indicates how much worse I{i} is than I{iBest},
            and is "#" if evaluation of I{i} produced an error.

            With this call pattern, nothing is returned.

        Two individuals
        ===============

            When I{iOther} is supplied, I call L{msgRatio} with
            I{iOther} vs I{i} to get the rounded improvement ratio
            (0-9) of how much better I{i} is than I{other}, i.e., how
            much lower its SSE is. I then print or log a progress
            character that indicates how much better I{i} is than
            I{iOther}.

            Unless, of course, I{i} is actually worse than I{iOther}
            (higher SSE), in which case the progress character is "X",
            denoting a failed challenge and the ratio will be C{None}.

            If I{i} is also better than my I{iBest}, then there will
            be an additional progress character of "+", and a small
            bonus added to my L{Population} object's replacements
            score, equivalent to a rounded improvement ratio of 1.

            With this call pattern, the rounded improvement ratio is
            returned, or C{None} if I{i} was worse than I{iOther}.

        @keyword noProgress: Set C{True} to suppress printing/logging
            a progress character.

        @keyword force: Set C{True} to force callbacks to run even if
            the reported SSE is considered equivalent to the previous
            best one.
        
        @see: L{newBest}, L{progressChar}, and L{msgRatio}.
        """
        def progressChar(x):
            if noProgress: return
            self.progressChar(x)

        if iOther is None:
            if not i:
                # Sole individual, and with an error.
                progressChar("%")
                return
            if self.iBest is None:
                # First, thus best
                self.newBest(i, force)
                progressChar("*")
                return
            if i < self.iBest or force:
                # Better than (former) best, so make best. The "ratio"
                # of how much worse than best will be 0
                self.newBest(i, force)
                progressChar("!")
                return
            # Worse than best (or same, unlikely), ratio is how
            # much worse
            self.msgRatio(i, self.iBest, bogus_char="#", noProgress=noProgress)
            return
        # Ratio is how much better this is than other. Thus, numerator
        # is other, because ratio is other SSE vs this SSE
        rir = self.msgRatio(iOther, i, noProgress=noProgress)
        # If better than best (or first), make new best
        if self.iBest is None or i < self.iBest:
            self.newBest(i, force)
            progressChar("+")
            # Bonus rir
            self.p.replacement(1)
        return rir
