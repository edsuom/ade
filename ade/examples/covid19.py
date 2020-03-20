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
Example script I{covid19.py}: Identifying coefficients for a naive
logistic growth model of the number of reported cases vs time in
hours.

This example reads a CSV file you may obtain from John Hopkins (one
that was obtained as of the date of this commit/version). Each line
contains the number of reported cases for one region of the world.

You probably want to see how many cases may be reported in your
country. If those are Iran, Italy, and US, you're in luck (actually,
not), because there are already subclasses of L{Covid19Data} for
them. To try it out in the U.S. for example, if you're wondering why
our shithead president can't seem to grasp that this is a real crisis,

1. Clone this repo, and update it to make sure you have the latest
   commit,

2. Do "pip install -e ." from within the top-level 'ade' directory
   of the repo,

3. Run the command "ade-examples" to create a directory "~/ade-examples".

4. Do the command line "./covid19.py US" inside the ~/ade-examples
   directory. You'll want to have the "pqiv" program on your computer.
   It should be as simple as doing "apt install pqiv".

This works on Linux; you'll probably know how to do the equivalent in
lesser operating systems. The code might even run.

The datafile time_series_19-covid-Confirmed.csv is for educational
purposes and reproduced (via download from edsuom.com) under the Terms
of Use for the data file, which is as follows:

"This GitHub repo and its contents herein, including all data,
mapping, and analysis, copyright 2020 Johns Hopkins University,
all rights reserved, is provided to the public strictly for
educational and academic research purposes.  The Website relies
upon publicly available data from multiple sources, that do not
always agree. The Johns Hopkins University hereby disclaims any
and all representations and warranties with respect to the
Website, including accuracy, fitness for use, and merchantability.
Reliance on the Website for medical guidance or use of the Website
in commerce is strictly prohibited."

A few people may come across this source file out of their own
interest in and concern about the COVID-19 coronavirus. I hope this
example of my open-source evolutionary optimization software of mine
gives them some insights about the situation.

BUT PLEASE NOTE THIS CRITICAL DISCLAIMER: First, I disclaim everything
that John Hopkins does. I'm pretty sure their lawyers had good reason
for putting that stuff in there, so I'm going to repeat it. Except
think "Ed Suominen" when you are reading "The Johns Hopkins
University."

Second, I know very little about biology, beyond a layman's
fascination with it and the way everything evolved. (Including this
virus!) I do have some experience with modeling, including using this
ADE package to develop some really cool power semiconductor simulation
software that I'll be releasing in a month or so from when I'm doing
the GitHub commit with this COVID-19 example. The software (also to be
free and open-source!) has a sophisticated subcircuit model for power
MOSFETs that evolves 40+ parameters (an unfathomably huge search
space). And it does so with this very package whose example you are
now reading.

The model I'm using for the number of reported cases of COVID-19
follows the logistic growth model, with a small (and not terribly
significant) linear term added. It has just 4 parameters, and finding
the best combination of those parameters is no problem at all for
ADE.

So, YES, THIS IS STILL A DISCLAIMER, I am not an expert in any of the
actual realms of medicine, biology, etc. that we rely on for telling
us what's going on with this virus. I just know how to fit models to
data, in this case a model that is well understood to apply to
biological populations.

Don't even THINK of relying on this analysis or the results of it for
any substantive action. If you really find it that important, then
investigate for yourself the math, programming, and the theory behind
my use of the math for this situation. Run the code, play with it,
critique it, consider how well the model does or does not
apply. Consider whether the limiting part of the curve might occur
more drastically or sooner, thus making this not as big a deal. Listen
to experts and the very real reasoning they may have for their own
projections about how bad this could get.

It's on you. I neither can nor will take any responsibility for what
you do. OK, ENOUGH DISCLAIMER.

The comma-separated fields are as follows:

    1. Region Name. Blank if nation-wide data. Of interest here is the
       state in the U.S., which may have a space.

    2. Country Name. "US" is of interest here. (It will come as no
       surprise that the author is a U.S. citizen!)

    3. Latitude.

    4. Longitude.

    5. This and the remaining fields are the number of cases reported
       each day starting on January 22, 2020.

Uses asynchronous differential evolution to efficiently find a
population of best-fit combinations of four parameters for the
function::

    x(t) = L/(1 + exp(-k*(t-t0))) + a*t

against data for daily numbers of reported COVID-19 cases, where
I{x} is the number of expected reported cases and I{t} is the time
since the first observation, in hours.
"""

import re, random
from datetime import date, timedelta

import numpy as np
from scipy.stats import uniform

from twisted.python import failure
from twisted.internet import reactor, defer

from asynqueue.process import ProcessQueue
from yampex.plot import Plotter

from ade.population import Population
from ade.de import DifferentialEvolution
from ade.image import ImageViewer
from ade.abort import abortNow
from ade.util import *


from data import Data


DISCLAIMER = """
Dedicated to the public domain by Edwin A. Suominen.
My only relevant expertise is in fitting nonlinear models
to data, not biology or medicine. See detailed disclaimer
in source file covid19.py.
"""

def oops(failureObj):
    """
    A handy universal errback.

    Prints the failure's error message to STDOUT and then stops
    everything so you can figure out what went wrong.
    """
    if isinstance(failureObj, failure.Failure):
        info = failureObj.getTraceback()
    else: info = str(failureObj)
    print(sub("Failure:\n{}\n{}\n", '-'*40, info))
    abortNow()
    os._exit(1)


class Covid19Data(Data):
    """
    Run L{setup} on my instance to decompress and load the
    covid19.csv.bz2 CSV file from Johns Hopkins.

    The CSV file isn't included in the I{ade} package and will
    automatically be downloaded from U{edsuom.com}. Here's the privacy
    policy for my site (it's short, as all good privacy policies
    should be)::

        Privacy policy: I don’t sniff out, track, or share anything
        identifying individual visitors to this site. There are no
        cookies or anything in place to let me see where you go on the
        Internet--that’s creepy. All I get (like anyone else with a
        web server), is plain vanilla server logs with “referral” info
        about which web page sent you to this one.

    @see: The L{Data} base class.
    """
    basename = "covid19"
    reDate = re.compile(r'([0-9]+)/([0-9]+)/([0-9]+)')
    summaryPosition = 'NW'

    re_ps_yes = None
    re_ps_no = None

    def __len__(self):
        return len(self.dates)

    def trim(self):
        """
        Trims the dataset so that the first day is when the first case was
        reported and the last day is the last day with new cases reported.
        """
        K = np.flatnonzero(self.X)
        for name in ('t', 'X', 'dates'):
            # There's a bug with trying to start at first reported
            # case, not worth pursuing right now
            Z = getattr(self, name)[:K[-1]]
            #Z = getattr(self, name)[K[0]:K[-1]]
            setattr(self, name, Z)
    
    def parseDates(self, result):
        """
        Parses dates from first line of I{result} list of text-value
        lists.
        """
        def g2i(k):
            return int(m.group(k))

        def hours(dateObj):
            seconds = (dateObj - firstDate).total_seconds()
            return seconds / 3600
        
        self.t = []
        self.dates = []
        firstDate = None
        for dateText in result.pop(0)[4:]:
            m = self.reDate.match(dateText)
            if not m:
                raise ValueError(sub(
                    "Couldn't parse '{}' as a date!", dateText))
            thisDate = date(2000+g2i(3), g2i(1), g2i(2))
            if firstDate is None:
                firstDate = thisDate
            self.dates.append(thisDate)
            self.t.append(hours(thisDate))
        self.t = np.array(self.t)
    
    def valuerator(self, result):
        """
        Iterates over the lists of text-values in the I{result}, yielding
        a 1-D array of daily reported case numbers for each list whose
        region code and country code match the desired values.
        """
        for rvl in result:
            if rvl[1] != self.countryCode:
                continue
            if self.re_ps_no and self.re_ps_no.search(rvl[0]):
                continue
            if self.re_ps_yes and not self.re_ps_yes.search(rvl[0]):
                msg("Not included: {}, {}", rvl[0], rvl[1])
                continue
            yield np.array([int(x) for x in rvl[4:] if x])
                
    def parseValues(self, result):
        self.parseDates(result)
        NX = len(self)
        self.X = np.zeros(NX)
        for Y in self.valuerator(result):
            NY = len(Y)
            N = min([NX, NY])
            self.X[:N] += Y[:N]
        self.trim()
                   
    def setup(self):
        """
        Calling this gets you a C{Deferred} that fires when setup is done
        and my I{t} and I{X} ivars are ready.
        """
        return self.load().addCallbacks(self.parseValues, oops)


class Covid19Data_US(Covid19Data):
    countryCode = 'US'
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (1e7, 5e8)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (1.05e-2, 1.2e-2)),
        # Midpoint time (hours)
        ('t0',  (2100, 2400)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0, 5e-2)),
    ]


class Covid19Data_Italy(Covid19Data):
    countryCode = 'Italy'
    summaryPosition = 'E'
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (5e4, 1.2e5)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (8e-3, 1.1e-2)),
        # Midpoint time (hours)
        ('t0',  (1280, 1440)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0.0, 0.3)),
    ]

    
class Covid19Data_Iran(Covid19Data):
    countryCode = 'Iran'
    summaryPosition = 'E'
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (1.7e4, 2.3e4)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (8.5e-3, 1.25e-2)),
        # Midpoint time (hours)
        ('t0',  (1140, 1230)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0.0, 0.2)),
    ]


class Covid19Data_SouthKorea(Covid19Data):
    countryCode = 'Korea, South'
    summaryPosition = 'E'
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (7e3, 1e4)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (1.1e-2, 1.6e-2)),
        # Midpoint time (hours)
        ('t0',  (940, 1020)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0.0, 0.2)),
    ]


class Evaluator(Picklable):
    """
    I evaluate fitness of the function::

        x(t) = L/(1 + exp(-k*(t-t0))) + a*t

    against data for daily numbers of reported COVID-19 cases, where
    I{x} is the number of expected reported cases and I{t} is the time
    since the first observation in hours.
    
    Construct an instance of me, run the L{setup} method, and wait (in
    non-blocking Twisted-friendly fashion) for the C{Deferred} it
    returns to fire. Then call the instance a bunch of times with
    parameter values for a L{curve} to get a (deferred)
    sum-of-squared-error fitness of the curve to the actual data.
    """
    scale_SSE = 1e-3

    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def setup(self, klass):
        """
        Call with a subclass I{klass} of L{Covid19Data} with data and
        bounds for evaluation of my model.
        
        Returns a C{Deferred} that fires with two equal-length sequences,
        the names and bounds of all parameters to be determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.
        """
        def done(null):
            return names, bounds

        if not issubclass(klass, Covid19Data):
            raise TypeError("You must supply a subclass of Covid19Data")
        data = self.data = klass()
        names = []; bounds = []
        for name, theseBounds in data.bounds:
            names.append(name)
            bounds.append(theseBounds)
        return data.setup().addCallbacks(done, oops)

    def hoursFromStart(self, k):
        """
        Given an index I{k} to my I{dates} list, returns the number of
        hours since the first reported case.
        """
        seconds = (self.dates[k] - self.dates[0]).total_seconds()
        return seconds / 3600
    
    def curve(self, t, *args):
        """
        Given a 1-D time vector followed by arguments defining curve
        parameters, returns a 1-D vector of expected Covid-19 cases to
        be reported over that time, in hours.

        A logistic growth model with a small amount of linearity (due
        to increasing testing early on skewing the early case numbers
        upward) is implemented this equation:

        M{x = L/(1 + exp(-k*(t-t0))) + a*t}
        """
        L, k, t0, a = args
        inside = -k*(t-t0)
        X = np.zeros_like(t)
        K = np.flatnonzero(inside < 700)
        X[K] = L/(1 + np.exp(inside[K]))
        return X + a*t
    
    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}.
        """
        X_curve = self.curve(self.t, *values)
        return self.scale_SSE * np.sum(np.square(X_curve - self.X))


class Reporter(object):
    """
    An instance of me is called each time a combination of parameters
    is found that's better than any of the others thus far.

    Prints the sum-of-squared error and parameter values to the
    console and updates a plot image (PNG) at I{plotFilePath}.

    @cvar plotFilePath: The file name in the current directory of a
        PNG file to write an update with a Matplotlib plot image of
        the actual vs. modeled temperature versus thermistor
        resistance curves.

    @cvar minShown: Minimum number of reported cases to show at the
        bottom of the log plot.

    @cvar daysMax: Maximum number of days to show from date of first
        reported case(s).

    @cvar N: Number of random points in extrapolated scatter plot.
    """
    plotFilePath = "covid19.png"
    minShown = 10
    daysBack = 7
    Nt = 100
    k0 = (30, 50)
    daysMax = (40, 50)

    def __init__(self, evaluator, population):
        """
        C{Reporter(evaluator, population)}
        """
        self.ev = evaluator
        self.p = population
        self.prettyValues = population.pm.prettyValues
        self.pt = Plotter(
            2, filePath=self.plotFilePath, width=10, height=12, h2=1)
        self.pt.use_grid()
        ImageViewer(self.plotFilePath)

    def clipLower(self, X):
        return np.clip(X, self.minShown, None)

    def kToday(self):
        """
        Returns the current number of days since the date of first
        reported case.
        """
        firstDay = self.ev.dates[0]
        seconds_in = (date.today() - firstDay).total_seconds()
        return int(seconds_in / 3600 / 24)
    
    def dayText(self, k):
        """
        Returns text indicating the date I{k} days after the first
        reported case.
        """
        firstDay = self.ev.dates[0]
        return (firstDay + timedelta(days=k)).strftime("%m/%d")

    def hours(self, k0, k1, N=None):
        """
        Returns a vector of times in hours from I{k0} to I{k1} days after
        the first reported case specified.

        If I{N} is specified, that many samples of time will be drawn
        from a uniform probability distribution instead of being once
        per day. The lowest possible value of any sample is k0, and
        the highest is k0 plus my I{daysMax}.
        """
        if N is None:
            days = np.arange(k0, k1, 1)
        else:
            days = k0 + (k1-k0) * uniform.rvs(size=N)
        return 24 * days
    
    def curvePoints(self, values, t):
        """
        Given a sequence of parameter I{values} and a vector I{t} of hours
        from first reported case, returns a version of I{t} in days
        from first report, with each corresponding value of I{X}.
        """
        return t/24, self.ev.curve(t, *values)

    def annotate(self, sp, X, k, k0=0, error=False):
        """
        Adds an annotation to subplot I{sp} for the supplied data vector
        to be plotted I{X} at I{k} days from the vector's first value,
        showing the number of reported cases then on that curve.

        @keyword k0: If the first value in I{X} is for a day after the
            first reported case, specify how many days from first
            report it is delayed with the keyword I{k0}.

        @keyword error: Set C{True} to annotate with error between the
            value in I{X} and the corresponding value from the actual
            data.
        """
        cases = int(round(X[k]))
        proto = sub("{}: ", self.dayText(k+k0))
        if error:
            if k+k0 >= len(self.ev.X):
                return
            cases -= self.ev.X[k+k0]
            proto += "{:+,.0f}"
        else: proto += "{:,.0f}"
        sp.add_annotation(k, proto, cases, kVector=1)
    
    def data_vs_model(self, sp, values, k0=0, future=False):
        """
        Plots the data against the best-fit model in subplot I{sp}, given
        the supplied parameter I{values}.

        @keyword k0: Set to a starting day if not the date of first
            reported case.

        @keyword future: Set C{True} to annotate extrapolated values
            instead of errors from past values.
        """
        def getValue(name):
            for k, nb in enumerate(self.ev.bounds):
                if nb[0] == name:
                    return values[k]

        def annotate_day(k, error=False):
            if k < 0 or k >= len(X_curve): return
            if k not in kSet:
                kSet.add(k)
                self.annotate(sp, X_curve, k, k0, error=error)

        kSet = set()
        daysMax = self.daysMax[1 if future else 0]
        k1 = k0 + daysMax + 1
        tCurve, X_curve = self.curvePoints(values, self.hours(k0, k1))
        t0_days = getValue('t0') / 24
        if t0_days < tCurve.max():
            sp.add_annotation(
                t0_days, "t0: {}", self.dayText(t0_days), kVector=1)
        # Date of first case(s) plotted
        annotate_day(0)
        # Today
        k = self.kToday() - k0
        if future:
            # Expected numbers of future reported cases
            # Yesterday
            annotate_day(k-1)
            # Today
            annotate_day(k)
            # Tomorrow
            annotate_day(k+1)
            # Day after tomorrow
            annotate_day(k+2)
            # One week from today
            annotate_day(k+7)
            # Two weeks from today
            annotate_day(k+14)
            # When target case thresholds reached
            for xt in (1e4, 5e4, 1e5, 5e5, 1e6, 5e6):
                if xt > X_curve[k] and xt < X_curve[-1]:
                    kt = np.argmin(np.abs(X_curve - xt))
                    annotate_day(kt)
            # The last day of extrapolation
            annotate_day(daysMax-1)
        else:
            # Error in expected vs actual reported cases, going back
            # several days starting with today
            for k in range(k, k-self.daysBack, -1):
                annotate_day(k, error=True)
            
        ax = sp.semilogy(self.ev.t[k0:]/24, self.clipLower(self.ev.X[k0:]))
        ax.semilogy(
            tCurve, self.clipLower(X_curve),
            color='red', marker='o', linewidth=2, markersize=3)
        return ax
    
    def add_scatter(self, ax, k0=0):
        """
        Adds points, using randomly selected parameter combos in present
        population, at times that are chosen from a uniform
        probability distribution.
        """
        Ni = len(self.p)
        tc = np.empty((self.Nt, Ni))
        Xc = np.empty_like(tc)
        k1 = k0 + self.daysMax[1]
        for ki in range(Ni):
            t = self.hours(k0, k1, N=self.Nt)
            tc[:,ki], Xc[:,ki] = self.curvePoints(list(self.p[ki]), t)
        tc = tc.flatten()
        Xc = self.clipLower(Xc.flatten())
        ax.semilogy(tc, Xc, color='black', marker=',', linestyle='')
    
    def subplot_upper(self, sp, values):
        """
        Does the upper subplot with model fit vs data.
        """
        k0 = self.k0[0]
        for k, nb in enumerate(self.ev.bounds):
            sp.add_textBox('SE', "{}: {:.5g}", nb[0], values[k])
        # Data vs best-fit model
        self.data_vs_model(sp, values, k0=k0)
        
    def subplot_lower(self, sp, values):
        """
        Does the lower subplot with extrapolation, starting at my I{k0}
        days from first report.
        """
        k0 = self.k0[1]
        ax = self.data_vs_model(sp, values, k0=k0, future=True)
        # Scatter plot to sort of show extrapolation uncertainty
        self.add_scatter(ax, k0=k0)
        
    def __call__(self, values, *args):
        """
        Prints out a new best parameter combination and its curve vs
        observations, with lots of extrapolation to the right.
        """
        def tb(*args):
            sp.add_textBox(self.ev.summaryPosition, *args)

        self.pt.set_title(
            "Modeling Reported Cases in {} of the Covid-19 Coronavirus",
            self.ev.countryCode)
        self.pt.set_ylabel("N")
        self.pt.set_xlabel("Days")
        self.pt.add_line('-', 2)
        self.pt.use_minorTicks('x', 1.0)
        with self.pt as sp:
            tb("Reported cases in {} vs days after first case.",
               self.ev.countryCode)
            tb("Annotations show residuals between model and data.")
            self.subplot_upper(sp, values)
            tb("Expected cases reported in {} vs days after first case.",
               self.ev.countryCode)
            tb("Dots show predictions at random future times for each of")
            tb("a final population of 120 evolved parameter combinations.")
            for line in DISCLAIMER.split('\n'):
                sp.add_textBox("SE", line)
            self.subplot_lower(sp, values)
        self.pt.show()

    
class Runner(object):
    """
    I run everything to fit a curve to thermistor data using
    asynchronous differential evolution.

    Construct an instance of me with an instance of L{Args} that has
    parsed command-line options, then have the Twisted reactor call
    the instance when it starts. Then start the reactor and watch the
    fun.
    """
    def __init__(self, args):
        """
        C{Runner(args)}
        """
        self.args = args
        self.ev = Evaluator()
        N = args.N if args.N else ProcessQueue.cores()-1
        self.q = ProcessQueue(N, returnFailure=True)
        self.fh = open("covid19.log", 'w') if args.l else True
        msg(self.fh)

    @defer.inlineCallbacks
    def shutdown(self):
        """
        Call this to shut me down when I'm done. Shuts down my
        C{ProcessQueue}, which can take a moment.

        Repeated calls have no effect.
        """
        yield self.p.reporter.waitForCallbacks()
        if self.q is not None:
            msg("Shutting down...")
            yield self.q.shutdown()
            msg("Task Queue is shut down")
            self.q = None
            msg("Goodbye")

    def tryGetClass(self):
        """
        Tries to return a reference to a subclass of L{Covid19Data} with
        the supplied I{suffix} following an underscore.
        """
        if not len(self.args):
            raise RuntimeError(
                "You must specify a recognized country/state code")
        name = sub("Covid19Data_{}", self.args[0])
        try: klass = globals()[name]
        except ImportError:
            klass = None
        if klass is not None and issubclass(klass, Covid19Data):
            return klass
        raise ImportError(sub("No data subclass '{}' found!", name))
            
    def evaluate(self, values):
        """
        The function that gets called with each combination of parameters
        to be evaluated for fitness.
        """
        if values is None:
            return self.shutdown()
        values = list(values)
        if self.q: return self.q.call(self.ev, values)
    
    @defer.inlineCallbacks
    def __call__(self):
        args = self.args
        startTime = time.time()
        klass = self.tryGetClass()
        names, bounds = yield self.ev.setup(klass).addErrback(oops)
        if len(args) > 1:
            self.p = Population.load(
                args[1], func=self.evaluate, bounds=bounds)
        else:
            self.p = Population(self.evaluate, names, bounds, popsize=args.p)
            yield self.p.setup().addErrback(oops)
        reporter = Reporter(self.ev, self.p)
        self.p.addCallback(reporter)
        F = [float(x) for x in args.F.split(',')]
        de = DifferentialEvolution(
            self.p,
            CR=args.C, F=F, maxiter=args.m,
            randomBase=not args.b, uniform=args.u, dwellByGrave=1,
            adaptive=not args.n, bitterEnd=args.e, logHandle=self.fh)
        yield de()
        yield self.shutdown()
        msg(0, "Final population:\n{}", self.p)
        msg(0, "Elapsed time: {:.2f} seconds", time.time()-startTime, 0)
        if len(args.P) > 1:
            savePicklePath = args.P
            self.p.save(savePicklePath)
            msg("Saved final population of best parameter combinations "+\
                "to {}", savePicklePath)
        self.q.call(reporter, list(self.p.best))
        msg(None)
        reactor.stop()

    def run(self):
        return self().addErrback(oops)


def main():
    """
    Called when this module is run as a script.
    """
    if args.h:
        return
    r = Runner(args)
    reactor.callWhenRunning(r.run)
    reactor.run()


args = Args(
    """
    Parameter finder for model of Covid-19 number of US reported cases
    vs time using Differential Evolution.

    Downloads a compressed CSV file of Covid-19 data points from
    edsuom.com to the current directory (if it's not already
    present). The data points are plotted against the current best-fit
    curve, along with a statistical extrapolation plot, in the PNG
    file (also in the current directory) covid19.png.

    On Linux, a plot window should be opened automatically. If not,
    you can obtain the free "qiv" image viewer and see the plots,
    automatically updated, with the command "qiv -Te
    covid19.png". (Possibly the other OS's may have something that
    works, too.)

    Press the Enter key to quit early.
    """
)
args('-m', '--maxiter', 50, "Maximum number of DE generations to run")
args('-e', '--bitter-end', "Keep working to the end even with little progress")
args('-p', '--popsize', 30, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.8, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-b', '--best', "Use DE/best/1 instead of DE/rand/1")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")
args('-l', '--logfile',
     "Write results to logfile 'covid19.log' instead of STDOUT")
args('-P', '--pickle', "covid19.dat",
     "Pickle dump file for finalized ade.Population object ('-' for none)")
args("<Country/State Name> [<pickle file>]")
args(main)
