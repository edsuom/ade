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
do the command line "./covid19.py US" inside this file's directory
(presumably ~/ade-examples). This works on Linux; you'll probably know
how to do the equivalent in lesser operating systems. The code might
even run.

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

import re
from datetime import date, timedelta

import numpy as np
from scipy.stats import triang as dist

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
            if rvl[1] not in self.countryCodes:
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
    countryCodes = ['US']
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (1e5, 5e6)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (1.2e-2, 1.6e-2)),
        # Midpoint time (hours)
        ('t0',  (1500, 1800)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0.01, 0.05)),
    ]


class Covid19Data_Iran(Covid19Data):
    countryCodes = ['Iran']
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (1e4, 2e4)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (1.0e-2, 1.5e-2)),
        # Midpoint time (hours)
        ('t0',  (1080, 1180)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0.0, 0.15)),
    ]

    
class Covid19Data_Italy(Covid19Data):
    countryCodes = ['Italy']
    bounds = [
        # Maximum number of cases expected to be reported, ever
        ('L',   (1.6e4, 3.4e4)),
        # The logistic growth rate, proportional to the number of
        # cases being reported per hour at midpoint
        ('k',   (1.0e-2, 1.3e-2)),
        # Midpoint time (hours)
        ('t0',  (1100, 1260)),
        # Linear term (constant hourly increase in the number of
        # reported cases)
        ('a',   (0.0, 0.1)),
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
            for name in ('bounds', 't', 'X', 'dates'):
                setattr(self, name, getattr(data, name))
            return names, bounds

        if not issubclass(klass, Covid19Data):
            raise TypeError("You must supply a subclass of Covid19Data")
        data = klass()
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
    daysMax = 75
    N = 10000

    def __init__(self, evaluator, population):
        """
        C{Reporter(evaluator, population)}
        """
        self.ev = evaluator
        self.p = population
        self.prettyValues = population.pm.prettyValues
        self.pt = Plotter(
            1, filePath=self.plotFilePath, width=18, height=12)
        self.pt.use_grid()
        ImageViewer(self.plotFilePath)

    def clipLower(self, X):
        return np.clip(X, self.minShown, None)
        
    def day(self, k=None):
        """
        With an integer I{k}, returns text indicating the date I{k} days
        after the start. Otherwise, returns an integer value of
        I{k} for today.
        """
        firstDay = self.ev.dates[0]
        if k is None:
            seconds_in = (date.today() - firstDay).total_seconds()
            return int(seconds_in / 3600 / 24)
        return (firstDay + timedelta(days=k)).strftime("%m/%d")
        
    def curvePoints(self, values, tflr=None):
        """
        Returns a customized I{t} and I{X} with one point per day along
        the best-fit curve, extrapolating as necessary to reach my
        I{daysMax}.

        The unit for the returned I{t} is days even though the curve
        of my L{Evaluator} accepts its I{t} vector with the unit being
        hours.

        Instead of using an equidistant time series, you can specify
        the keyword I{tflr} for a single time in hours from the last
        reported case. The value of I{t} in hours from first report
        will be returned, as usual, with the single value of I{X}. In
        this mode, a constant is added to set the value of I{X} to
        make it exactly match the last number of cases reported, i.e.,
        the value it would have with M{tflr=0}.
        """
        if tflr is None:
            tDaily = np.arange(0, self.daysMax, 1)
            tDailyHours = 24 * tDaily
            X = self.ev.curve(tDailyHours, *values)
            return tDaily, X
        t0 = self.ev.hoursFromStart(-1)
        t = t0 + tflr
        X0, Xt = self.ev.curve(np.array([t0, t]), *values)
        return t, Xt-X0+self.ev.X[-1]

    def annotate(self, sp, k, X=None):
        """
        Adds an annotation to subplot I{sp} for the supplied data vector
        I{X} at day I{k} (start=0), showing the number of
        reported cases then on that curve.

        To add the annotation to the actual number of cases reported,
        omit the data vector.
        """
        if X is None:
            cases = self.ev.X[k]
            kVector = 0
        else:
            cases = X[k]
            kVector = 1
        sp.add_annotation(
            k, "{:n} on {}", int(round(cases)), self.day(k), kVector=kVector)
    
    def data_vs_model(self, sp, values):
        """
        Plots the data against the best-fit model.
        """
        def getValue(name):
            for k, nb in enumerate(self.ev.bounds):
                if nb[0] == name:
                    return values[k]

        def annotate_day(k=None):
            if k is None:
                k = self.day()
            elif k < 0:
                k = len(X_curve) + k
            self.annotate(sp, k, X_curve)
            return k

        t = self.ev.t
        tCurve, X_curve = self.curvePoints(values)
        t0_days = getValue('t0') / 24
        if t0_days < tCurve.max():
            sp.add_annotation(t0_days, "t0", kVector=1)
        # Date of first case(s)
        self.annotate(sp, 0)
        # Today
        k = annotate_day()
        # One week from today
        annotate_day(k+7)
        # Two weeks from today
        annotate_day(k+14)
        # When target case thresholds reached
        for xt in (10000, 50000, 100000, 500000):
            if xt > X_curve[k] and xt < X_curve[-1]:
                kt = np.argmin(np.abs(X_curve - xt))
                annotate_day(kt)
        ax = sp.semilogy(t/24, self.clipLower(self.ev.X))
        ax.semilogy(
            tCurve, self.clipLower(X_curve),
            color='red', marker='o', linewidth=2, markersize=3)
        return ax

    def add_scatter(self, ax):
        """
        Adds points, using randomly selected parameter combos in present
        population, at times that are chosen from a probability
        distribution skewed rightward.

        The rightward skewing is to show more points where there is
        more range.
        """
        N_days = self.daysMax - len(self.ev.dates)
        t0 = self.ev.hoursFromStart(-1)
        rvs = dist.rvs(1.0, size=self.N)**2
        t_fromLastReport = 24*N_days*rvs
        t = np.empty(self.N)
        X = np.empty(self.N)
        for k, tk in enumerate(t_fromLastReport):
            i = self.p[self.p.sample(1)]
            values = list(i)
            t[k], X[k] = self.curvePoints(values, tk)
        ax.semilogy(
            t/24, self.clipLower(X),
            color='black', marker=',', linestyle='')
    
    def __call__(self, values, counter, SSE):
        """
        Prints out a new best parameter combination and its curve vs
        observations, with lots of extrapolation to the right.
        """
        def titlePart(*args):
            titleParts.append(sub(*args))

        SSE_info = sub("SSE={:g}", SSE)
        titleParts = []
        titlePart("Cases vs Time (days)")
        titlePart(SSE_info)
        titlePart("k={:d}", counter)
        msg(0, self.prettyValues(values, SSE_info+", with"), 0)
        with self.pt as sp:
            sp.set_title(", ".join(titleParts))
            sp.add_line('-', 2)
            sp.set_xlabel("Days")
            sp.set_ylabel("N")
            for k, nb in enumerate(self.ev.bounds):
                sp.add_textBox('SE', "{}: {:.5g}", nb[0], values[k])
            # Data vs best-fit model
            ax = self.data_vs_model(sp, values)
            # Scatter plot to show extrapolation uncertainty
            self.add_scatter(ax)
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
args('-p', '--popsize', 50, "Population: # individuals per unknown parameter")
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