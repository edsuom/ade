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

    xd(t) = a * (t-ts)^n * exp(-(t-ts)/t0)    

against data for daily numbers of new reported COVID-19 cases, where
I{xd} is the number of new cases expected to be reported each day and
I{t} is the time since the first observation, in days.
"""

import re, random
from datetime import date, timedelta

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from twisted.python import failure
from twisted.internet import reactor, defer

from asynqueue import ThreadQueue, ProcessQueue
from yampex.plot import Plotter

from ade.population import Population
from ade.de import DifferentialEvolution
from ade.image import ImageViewer
from ade.abort import abortNow
from ade.util import *


from data import Data


DISCLAIMER = """
Dedicated to the public domain by Edwin A. Suominen. My
only relevant expertise is in fitting nonlinear models
to data, not biology or medicine. See detailed
disclaimer in source file covid19.py.
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
            self.t.append(hours(thisDate) / 24)
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
                
    def parseValues(self, result, daysAgo):
        self.parseDates(result)
        NX = len(self)
        self.X = np.zeros(NX)
        for Y in self.valuerator(result):
            NY = len(Y)
            N = min([NX, NY])
            self.X[:N] += Y[:N]
        if daysAgo:
            for name in ('t', 'dates', 'X'):
                setattr(self, name, getattr(self, name)[:-daysAgo])
                   
    def setup(self, daysAgo=0):
        """
        Calling this gets you a C{Deferred} that fires when setup is done
        and my I{t} and I{X} ivars are ready.

        @keyword daysAgo: Limit latest data to this many days ago
            (integer) rather than up to today (0 days ago).
        """
        d = self.load()
        d.addCallback(self.parseValues, daysAgo)
        d.addErrback(oops)
        return d


class Covid19Data_US(Covid19Data):
    countryCode = 'US'
    bounds_lg = [
        #--- Logistic Growth (list these first) -------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (2e6, 3.3e8)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (1e-2, 2.4e-1)),
    ]
    bounds_pl = [
        #--- Power-Law (list these second) ------------------------------------
        # Scaling coefficient
        ('a',   (1.0, 500.0)),
        # Power-law exponent
        ('n',   (0.1, 2.0)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (43, 55)),
        # Decay time constant (two years is 730 days)
        ('t0',  (10, 1000)),
    ]
    bounds_l = [
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0.0, 1.0)),
        #----------------------------------------------------------------------
    ]
    bounds = bounds_lg + bounds_pl + bounds_l


class Covid19Data_Italy(Covid19Data):
    countryCode = 'Italy'
    summaryPosition = 'E'
    bounds = [
        #--- Logistic Growth --------------------------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (2e4, 1.2e5)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (3e-4, 7e-4)),
        #--- Power-Law Component ----------------------------------------------
        # Scaling coefficient
        ('a',   (1.0, 200.0)),
        # Power-law exponent
        ('n',   (1.2, 3.5)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (20, 52)),
        # Decay time constant
        ('t0',  (1e1, 1e5)),
        #--- Linear -----------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 8.0)),
    ]


class Covid19Data_Iran(Covid19Data):
    countryCode = 'Iran'
    summaryPosition = 'E'


class Covid19Data_SouthKorea(Covid19Data):
    countryCode = 'Korea, South'
    summaryPosition = 'E'


class Covid19Data_Finland(Covid19Data):
    countryCode = 'Finland'
    summaryPosition = 'E'


class Evaluator(Picklable):
    """
    I evaluate fitness of one of two different models, or both in
    linear combination, against the number of new COVID-19 cases
    reported each day.

    One model (my preferred one due to its remarkable closeness of fit
    thus far) is the logistic (Verhulst) model: "A biological
    population with plenty of food, space to grow, and no threat from
    predators, tends to grow at a rate that is proportional to the
    population....Of course, most populations are constrained by
    limitations on resources--even in the short run--and none is
    unconstrained forever."
    U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html}

    It requires the integration of a first-order differential
    equation:::

        xd(t, x) = x*r*(1 - x/L)
    
    The other is the power-law model with exponential cutoff
    function of Vasquez (2006):::

        xd(t) = a * (t-ts)^n * exp(-(t-ts)/t0)

    Finally, you can add a linear component to either model or the
    combination of both with the single parameter I{b}:::

        xd(t) += b*t
    
    The data in my 1-D array I{Xd} contains daily numbers of B{new}
    reported COVID-19 cases. The modeled value of the functions xd{x}
    is the number of new cases expected to be reported and I{t} is the
    time since the first observation (in days).
    
    Construct an instance of me, run the L{setup} method, and wait (in
    non-blocking Twisted-friendly fashion) for the C{Deferred} it
    returns to fire. Then call the instance a bunch of times with
    parameter values for a L{curve} to get a (deferred)
    sum-of-squared-error fitness of the curve to the actual data.

    You select a model by defining its uniquely named parameters in
    the I{bounds} list of your chosen subclass of L{Covid19Data}. For
    example, to use a linear combination of both models, that list
    should define I{L} and L{r} for the logistic growth component and
    I{a}, I{n}, I{ts}, and I{t0} for the power-law component. If a
    linear component is desired (constant number of new cases each
    day), define it with parameter I{b}.

    @ivar Xd: The actual number of new cases per day, each day, for
        the selected country or region since the first reported case
        in the Johns Hopkins dataset on January 22, 2020, B{after} my
        L{transform} has been applied.
    """
    scale_SSE = 1e-2
    fList = None

    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def setup(self, klass, daysAgo=0):
        """
        Call with a subclass I{klass} of L{Covid19Data} with data and
        bounds for evaluation of my model.

        Computes a differential vector I{Xd} that contains the
        differences between a particular day's cumulative cases and
        the previous day's. The length this vector is the same as
        I{X}, with a zero value for the first day.
        
        Returns a C{Deferred} that fires with two equal-length
        sequences, the names and bounds of all parameters to be
        determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.
        """
        def done(null):
            # Differential
            self.Xd = np.zeros_like(self.X)
            msg("Cumulative and new cases reported thus far for {}",
                self.countryCode, '-')
            xPrev = None
            for k, x in enumerate(self.X):
                xd = 0 if xPrev is None else x-xPrev
                xPrev = x
                msg("{:03d}\t{}\t{:d}\t{:d}",
                    k, self.dayText(k), int(x), int(xd))
                self.Xd[k] = self.transform(xd)
            return names, bounds

        if not issubclass(klass, Covid19Data):
            raise TypeError("You must supply a subclass of Covid19Data")
        data = self.data = klass()
        names = []; bounds = []
        for name, theseBounds in data.bounds:
            names.append(name)
            bounds.append(theseBounds)
        return data.setup(daysAgo).addCallbacks(done, oops)

    def transform(self, Xd, inverse=False):
        """
        Applies a transform to the numbers of new cases per day each day,
        real or modeled. Set I{inverse} C{True} to apply the inverse
        of the transform.

        The crude transform currently used is just a square root of
        the absolute magnitude, with sign preserved.  The seems like a
        reasonable initial compromise between a log transform (useful
        for a purely exponential model), and not transforming at
        all. Will investigate Cox-Box as an option.
        """
        if inverse:
            return np.sign(Xd) * Xd**2
        return np.sign(Xd) * np.sqrt(np.abs(Xd))
    
    def dayText(self, k):
        """
        Returns text indicating the date I{k} days after the first
        reported case.
        """
        firstDay = self.dates[0]
        return (firstDay + timedelta(days=k)).strftime("%m/%d")

    def curve_logistic(self, t, x, L, r):
        """
        Logistic growth model (Verhulst model),
        U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html}
        
        Given a scalar time (in days) followed by arguments defining
        curve parameters, returns the number of B{new} Covid-19
        cases expected to be reported on that day.::

            xd(t, x) = x*r*(1 - x/L)
            
        This requires integration of a first-order differential
        equation, which is performed in L{curve}.
        """
        x = np.array(x)
        return r*x*(1 - x/L)
    
    def curve_powerlaw(self, t, x, a, n, ts, t0):
        """
        Power law model,
        U{https://www.medrxiv.org/content/10.1101/2020.02.16.20023820v2.full.pdf}
        
        Given a scalar time vector (in days) followed by arguments
        defining curve parameters, returns the number of B{new}
        Covid-19 cases expected to be reported on that day.::

            xd(t, x) = a * (t-ts)^n * exp(-(t-ts)/t0)

        Note that there is actually no dependence on I{x}. It is just
        included as an argument for consistency with
        L{curve_logistic}, since a linear combination of both requires
        the use of an ODE solver.

        Thanks to my new FB friend, applied statistician Ng Yi Kai
        Aaron of Singapore, for suggesting I look into the power law
        model as an alternative, and pointing me to an article
        discussing its possible application to COVID-19.
        """
        t -= ts
        if t < 0: return np.zeros_like(x)
        return a*t**n * np.exp(-t/t0) * np.ones_like(x)

    def curve_linear(self, t, x, b):
        """
        Linear component: Constant number of new cases reported each day.
        """
        return b * np.ones_like(x)
    
    def curve(self, t, *values):
        """
        Returns the expected number of new cases per day for each day in
        1-D array I{t}, using the logistic growth model, the power law
        model, or both, with or without a linear component.

        If there are 2 parameters, I{L} and I{k}, only logistic growth
        would be used, but that doesn't work because the initial
        number of reported cases is zero. You need one of the other
        terms, too.

        If there are 4 parameters, I{a}, I{n}, I{ts}, and I{t0},
        only power-law is used.

        If there are 6 parameters, both logistic growth and power
        law are used in a linear combination.

        If there are 3 parameters (logistic growth plus linear), 5
        parameters (power-law plus linear), or 7 parameters
        (everything), a linear component is added.
        """
        def f(t, x):
            Xd = np.zeros_like(x)
            for f, ks in self.fList:
                Xd_this = np.clip(f(t, x, *values[ks]), 0, None)
                Xd += Xd_this
            return Xd
        
        if self.fList is None:
            k = 0
            self.fList = []
            N = len(values)
            if N == 2:
                raise ValueError(
                    "You can't have just logistic growth by itself!")
            if N in (6, 3, 7):
                self.fList.append((self.curve_logistic, slice(k, k+2)))
                k += 2
            if N in (4, 6, 5, 7):
                self.fList.append((self.curve_powerlaw, slice(k, k+4)))
                k += 4
            if N in (3, 5, 7):
                self.fList.append((self.curve_linear, slice(k, k+1)))
        sol = solve_ivp(
            f, [0.0, t.max()], f(0.0, [0.0]), t_eval=t, vectorized=True)
        if not sol.success:
            raise RuntimeError("ODE solver failed!")
        return sol.y[0]
    
    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}. 
        """
        Xd_curve = self.transform(self.curve(self.t, *values))
        return self.scale_SSE * np.sum(np.square(Xd_curve - self.Xd))


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
    daysBack = 10
    k0 = 40
    daysMax = 50

    def __init__(self, evaluator, population):
        """
        C{Reporter(evaluator, population)}
        """
        self.ev = evaluator
        self.p = population
        self.prettyValues = population.pm.prettyValues
        self.pt = Plotter(
            3, filePath=self.plotFilePath, width=10, height=14, h2=[0, 2])
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
    
    def days(self, k0, k1, N=None):
        """
        Returns a vector of times in days from I{k0} to I{k1} days after
        the first reported case specified.

        If I{N} is specified, that many samples of time will be drawn
        from a uniform probability distribution instead of being once
        per day. The lowest possible value of any sample is k0, and
        the highest is k0 plus my I{daysMax}.
        """
        if N is None:
            return np.arange(k0, k1, 1)
        return k0 + (k1-k0) * uniform.rvs(size=N)
    
    def curvePoints(self, values, k0, k1, X0=None):
        """
        Given a sequence of parameter I{values} and a first I{k0} and
        second I{k1} number of days from first reported case, returns
        (1) a 1-D array with days from first reported case, starting
        at I{k0} and ending at I{k1}, and (2) a 1-D array with the
        expected number of B{cumulative} cases reported on each of
        those days.

        @keyword X0: Set this to a known number of days at I{k0}. The
            first element of the returned array will have that exact
            number, and elements for subsequent days will extrapolate
            from that point.
        """
        if k1 <= k0: raise ValueError("Number k1 must be > k0!")
        t = np.arange(0, k1)
        Xd = self.ev.curve(t, *values)
        X = np.zeros_like(t)
        for k, Xdk in enumerate(Xd):
            if X0 is not None:
                if k < k0:
                    continue
                if k == k0:
                    X[k] = X0
                    continue
            if k < 1:
                continue
            X[k] = X[k-1] + Xd[k]
        return t[k0:], X[k0:]
    
    def annotate(self, sp, k, k0, X_curve=None, X_data=None):
        """
        Adds an annotation to subplot I{sp} for the supplied data vector
        to be plotted, at I{k} days from the vector's first value,
        with a first plotted day of I{k0} from first reported.

        You must supply I{X_curve}, I{X_data}, or both:

            * If you supply I{X_curve}, the annotation will be to the
              modeled curve (kVector=1).

            * If you supply I{X_data}, the annotation will be to the
              data curve (kVector=0).

            * If you supply both, the annotation will be to the
              modeled curve (kVector=1), with an error between that
              and the corresponding point on the data curve.
        """
        if X_data is None:
            X = X_curve
            error = False
            kVector = 1
        elif X_curve is None:
            X = X_data
            error = False
            kVector = 0
        else:
            N = min([len(X_curve), len(X_data)])
            X = X_curve[:N] - X_data[:N]
            error = True
            kVector = 1
        if k < 0 or k >= len(X):
            # Annotation is beyond the plotted range, skip
            return
        proto = sub("{}: ", self.ev.dayText(k+k0))
        cases = int(round(X[k]))
        if error:
            proto += "{:+,.0f}"
        else: proto += "{:,.0f}"
        sp.add_annotation(k, proto, cases, kVector=kVector)
    
    def data_vs_model(self, sp, values, k0=0, future=False):
        """
        Plots the data against the best-fit model in subplot I{sp}, given
        the supplied parameter I{values}.

        Returns a 3-tuple with (1) the C{yampex.subplot.SpecialAx}
        object for the subplot, and (2) a 1-D array I{t_data} with the
        number of days since first reported case, in days, for the
        actual data, and (3) a 1-D array I{t_curve} for the modeled
        data, both starting with I{k0}.

        @keyword k0: Set to a starting day if not the date of first
            reported case.

        @keyword future: Set C{True} to annotate extrapolated values
            instead of errors from past values.
        """
        def getValue(name):
            for k, nb in enumerate(self.ev.bounds):
                if nb[0] == name:
                    return values[k]

        def annotate_past(k):
            self.annotate(sp, k, k0, X_data=X_data)
            
        def annotate_future(k):
            # Avoids duplicates
            if k in kSet: return
            self.annotate(sp, k, k0, X_curve=X_curve)
            kSet.add(k)

        def annotate_error(k):
            self.annotate(sp, k, k0, X_curve=X_curve, X_data=X_data)

        kSet = set()
        kToday = self.kToday()
        if future:
            k1 = k0 + self.daysMax + 1
        else: k1 = kToday + 3
        t_curve, X_curve = self.curvePoints(values, k0, k1)
        t_data = self.ev.t[k0:]
        X_data = self.clipLower(self.ev.X[k0:])
        k = kToday - k0
        # Date of first case(s) plotted
        annotate_past(0)
        # Today
        annotate_past(k)
        if future:
            # Expected numbers of future reported cases
            # Day before yesterday
            annotate_past(k-2)
            # Yesterday
            annotate_past(k-1)
            # Tomorrow
            annotate_future(k+1)
            # Day after tomorrow
            annotate_future(k+2)
            # One to three weeks from today
            for weeks in range(1, 4):
                annotate_future(k+7*weeks)
            # When target case thresholds reached
            for xt in (1e4, 5e4, 1e5, 5e5, 1e6, 5e6):
                if xt > X_curve[k] and xt < X_curve[-1]:
                    kt = np.argmin(np.abs(X_curve - xt))
                    annotate_future(kt)
            # The last day of extrapolation
            annotate_future(self.daysMax-1)
        else:
            # Error in expected vs actual reported cases, going back
            # several days starting with today
            for k in range(k, k-self.daysBack, -1):
                annotate_error(k)
        sp.add_axvline(-1)
        ax = sp.semilogy(t_data, X_data)
        ax.semilogy(
            t_curve, self.clipLower(X_curve),
            color='red', marker='o', linewidth=2, markersize=3)
        return ax, t_data, t_curve
    
    def add_scatter(self, ax, k0, k1):
        """
        Adds points, using randomly selected parameter combos in present
        population, at times that are chosen from a uniform
        probability distribution.
        """
        Nt = k1 - k0
        Ni = len(self.p)
        tc = np.empty((Nt, Ni))
        Xc = np.empty_like(tc)
        X0 = self.ev.X[k0]
        for ki in range(Ni):
            t, X = self.curvePoints(list(self.p[ki]), k0, k1, X0)
            tc[:,ki], Xc[:,ki] = t+0.1*stats.norm.rvs(size=Nt), X
        tc = tc.flatten()
        Xc = self.clipLower(Xc.flatten())
        ax.semilogy(tc, Xc, color='black', marker=',', linestyle='')
    
    def subplot_upper(self, sp, values):
        """
        Does the upper subplot with model fit vs data.

        Returns the I{t} vector for the actual-data dates being
        plotted.
        """
        sp.add_line('-', 2)
        k0 = self.k0
        for k, nb in enumerate(self.ev.bounds):
            sp.add_textBox('SE', "{}: {:.5g}", nb[0], values[k])
        # Data vs best-fit model
        return self.data_vs_model(sp, values, k0=k0)[1]

    def subplot_middle(self, sp, values, t):
        """
        Does a middle subplot with residuals between the data I{X_data}
        and modeled data I{X_curve}.
        """
        def tb(proto, *args):
            sp.add_textBox("SW", proto, *args)
        
        sp.set_ylabel("Residual")
        sp.set_xlabel("Modeled (fitted) new cases/day (square-root transform)")
        sp.set_zeroLine(color="red", linestyle='-', linewidth=3)
        sp.add_line('')
        sp.add_marker('o', 4)
        sp.set_colors("black")
        Xd_curve = self.ev.transform(self.ev.curve(t, *values))
        k0 = int(t[0])
        Xd = self.ev.Xd[k0:]
        R = Xd - Xd_curve
        cdf = ('t', len(R)-1)
        cdf_text = sub("{}({})", cdf[0], ", ".join([str(x) for x in cdf[1:]]))
        tb("Residuals: Modeled vs actual new cases/day (transformed)")
        tb("Kolmogorov-Smirnov goodness of fit to '{}' "+\
           "distribution: {:.4f} (p < {:.4f})",
           cdf_text, *stats.kstest(R, cdf[0], cdf[1:]))
        K = np.argsort(Xd_curve)
        sp(Xd_curve[K], R[K])
        
    def subplot_lower(self, sp, values):
        """
        Does the lower subplot with extrapolation, starting at my I{k0}
        days from first report.
        """
        sp.add_line('-', 2)
        k0 = self.k0
        ax, t_data, t_curve = self.data_vs_model(
            sp, values, k0=k0, future=True)
        # Scatter plot to sort of show extrapolation uncertainty
        self.add_scatter(ax, k0=int(t_data.max()), k1=int(t_curve.max()))
        
    def __call__(self, values, *args):
        """
        Prints out a new best parameter combination and its curve vs
        observations, with lots of extrapolation to the right.
        """
        def tb(*args):
            sp.add_textBox(self.ev.summaryPosition, *args)

        # Make a frozen local copy of the values list to work with, so
        # outdated stuff doesn't get plotted
        values = list(values)
        msg(0, "Values: {}", ", ".join([str(x) for x in values]))
        self.pt.set_title(
            "Modeled (red) vs Actual (blue) Reported Cases of COVID-19: {}",
            self.ev.countryCode)
        self.pt.set_ylabel("Reported Cases")
        self.pt.set_xlabel("Days after January 22, 2020")
        self.pt.use_minorTicks('x', 1.0)
        with self.pt as sp:
            tb("Reported cases in {} vs days after first case.",
               self.ev.countryCode)
            tb("Annotations show residuals between model and data.")
            t_data = self.subplot_upper(sp, values)
            self.subplot_middle(sp, values, t_data)
            tb("Expected cases reported in {} vs days after first",
               self.ev.countryCode)
            tb("case. Dots show predictions each day for each")
            tb("of a final population of 120 evolved parameter")
            tb("combinations. Annotations show actual values in")
            tb("past, best-fit projected values in future.")
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
        if args.t:
            self.q = ThreadQueue(returnFailure=True)
        else:
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
        names, bounds = yield self.ev.setup(klass, args.d).addErrback(oops)
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
        msg(0, "Final population:\n{}", self.p)
        msg(0, "Elapsed time: {:.2f} seconds", time.time()-startTime, 0)
        if len(args.P) > 1:
            savePicklePath = args.P
            self.p.save(savePicklePath)
            msg("Saved final population of best parameter combinations "+\
                "to {}", savePicklePath)
        yield self.shutdown()
        reactor.stop()
        msg(None)

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
args('-d', '--days-ago', 0,
     "Limit latest data to N days ago rather than up to today")
args('-m', '--maxiter', 50, "Maximum number of DE generations to run")
args('-e', '--bitter-end', "Keep working to the end even with little progress")
args('-p', '--popsize', 30, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.8, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-b', '--best', "Use DE/best/1 instead of DE/rand/1")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")
args('-t', '--threads',
     "Use a single worker thread instead of processes (for debugging)")
args('-l', '--logfile',
     "Write results to logfile 'covid19.log' instead of STDOUT")
args('-P', '--pickle', "covid19.dat",
     "Pickle dump file for finalized ade.Population object ('-' for none)")
args("<Country/State Name> [<pickle file>]")
args(main)
