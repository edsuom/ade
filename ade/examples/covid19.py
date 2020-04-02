#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018-20 by Edwin A. Suominen,
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
country. If those are Italy or the US, you're in luck (actually, not),
because there are already subclasses of L{Covid19Data} for them. To
try it out in the U.S. for example, if you're wondering whether this
is a real crisis despite what our shithead president said and didn't
do for weeks and weeks:

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
population of best-fit combinations of parameters for a first-order
differential equation modeling the number of new Covid-19 cases per
day. A variety of models are available. The desired linear combination
of them (two are exclusive, logistic growth and my own logistic growth
with curve flattening. The model(s) are selected based on what
parameter boundaries are defined in a subclass of L{Covid19Data}. You
can define any number of subclasses for any number of countries or
regions. There is one for each model combination for U.S. reported
cases.

The model is compared to data for daily numbers of new reported
COVID-19 cases, where I{xd} is the number of new cases expected to be
reported each day, I{t} is the time since the first observation, in
days, and I{x} is the number of total reported cases at time I{t}.

With the exception of the residual subplot in the middle, the plot
shows cumulative numbers of total cases C{x(t)}, not the differential
model function C{xd(t, x)} that is fitted to the number of daily
B{new} cases. To produce these plots and the predictions that we are
all interested in, the plots are anchored to the most recently known
value of the number of reported cases. Then I{xd(t, x)} is integrated
in both directions from the date for that most recent known value.

The integration goes backwards to show fit between the model and
historical numbers of total cases, which can be expected to get worse
in percentage terms as the known point gets farther away but probably
not so bad in absolute terms as both modeled and historical numbers
shrink rapidly.

For extrapolations from the most recent known date, the integration is
done forwards from that point. Again, precision will get worse as the
model ventures farther into the future. Extrapolation is dangerous
even for models with a solid theoretical basis (and this one was
produced by a biology+medicine layman), so see the disclaimer.
"""

import re, random
from datetime import date, timedelta

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from twisted.python import failure
from twisted.internet import reactor, defer

from asynqueue import ThreadQueue, ProcessQueue, DeferredTracker
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

    @cvar second_deriv: Set this C{True} to have the model considered
        the second derivative of the cumulative number of cases,
        rather than the first derivative.
    
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
        self.t = np.array(self.t, dtype=float)
    
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
    """
    US, first-order ODE, 6-parameter model: linear combination of
    logistic growth with curve flattening, and linear.

    The default model, because it is fitting the curve really well.

    AICc with 3/31/20 data was -5, much better than any of the other
    models, and curve fit extends further back into the past. Hardly
    any non-normality to the residuals, and very little correlated
    between parameters.
    """
    countryCode = 'US'
    bounds = [
        #--- Logistic Growth with curve flattening (list these first) ---------
        # The initial exponential growth rate
        ('r',   (0.111, 0.129)),
        # Reduction in effective r given current number of cases (when
        # rc*x = 1, number of cases x has reached absolute limit)
        ('rc',   (0.0, 1.2e-6)),
        # Max reduction in effective r from curve flattening effect
        # (>2 would be negative growth)
        ('rf',  (0.75, 1.07)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (2.5, 3.7)),
        # Time at which flattening is occurring fastest (days after 1/22/20)
        ('t0', (61, 64)),
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 20)),
        #----------------------------------------------------------------------
    ]
    k0 = 42


class Covid19Data_US_LGPLL(Covid19Data):
    """
    US, first-order ODE, 7-parameter model: linear combination of
    logistic growth, power law, and linear.

    AICc with 3/31/20 data was +4 after 100 generations.
    """
    countryCode = 'US'
    bounds = [
        #--- Logistic Growth (list these first) -------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (1e5, 5e6)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.03, 0.28)),
        #--- Power-Law (list these second) ------------------------------------
        # Scaling coefficient
        ('a',   (1500, 5000)),
        # Power-law exponent
        ('n',   (0.01, 0.8)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (55, 58)),
        # Decay time constant (two years is 730 days)
        ('t0',  (10, 1e6)),
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 400)),
        #----------------------------------------------------------------------
    ]
    k0 = 50


class Covid19Data_US_LGL(Covid19Data):
    """
    US, first-order ODE, 3-parameter model: logistic growth and
    linear.

    AICc with 3/31/20 data was +1.7 after 50 generations, but the
    upper limit seems unreasonably low (optimistic).
    """
    countryCode = 'US'
    summaryPosition = 'E'
    bounds = [
        #--- Logistic Growth (list these first) -------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (2e5, 7e5)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.2, 0.35)),
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 1e3)),
        #----------------------------------------------------------------------
    ]
    k0 = 50

class Covid19Data_US_PLL(Covid19Data):
    """
    US, first-order ODE, 5-parameter model: linear combination of
    power law and linear.

    AICc with 3/31/20 data was -1.8 after 50 generations. That is
    significantly better than US7 or US3, but the model doesn't track
    known data much earlier than a week ago.

    The I{t0} parameter goes absurdly high with no significant effect
    on model fitness; there is no apparently no exponential decay to
    be seen in the power-law behavior, here or in US7.
    """
    countryCode = 'US'
    bounds = [
        #--- Power-Law --------------------------------------------------------
        # Scaling coefficient
        ('a',   (200, 2000)),
        # Power-law exponent
        ('n',   (0.8, 1.7)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (50, 57)),
        # Decay time constant (two years is 730 days)
        ('t0',  (10, 1e6)),
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (200, 1200)),
        #----------------------------------------------------------------------
    ]
    k0 = 50


class Covid19Data_US_2d(Covid19Data):
    """
    US, second-order ODE treating model as the differential in the
    number of number of new daily cases. In other words, as the
    increase of the increase.
    """
    countryCode = 'US'
    second_deriv = True
    bounds_lg = [
        #--- Logistic Growth (list these first) -------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (1e5, 3.3e8)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.12, 0.26)),
    ]
    bounds_pl = [
        #--- Power-Law (list these second) ------------------------------------
        # Scaling coefficient
        ('a',   (20.0, 700.0)),
        # Power-law exponent
        ('n',   (0.0, 1.4)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (42, 58)),
        # Decay time constant (two years is 730 days)
        ('t0',  (2, 600)),
    ]
    bounds_l = [
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0.0, 0.1)),
        #----------------------------------------------------------------------
    ]
    bounds = bounds_lg + bounds_pl + bounds_l
    k0 = 40


class Covid19Data_Italy(Covid19Data):
    countryCode = 'Italy'
    summaryPosition = 'NW'
    bounds = [
        #--- Logistic Growth --------------------------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (2e4, 1e6)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.05, 0.3)),
        #--- Power-Law Component ----------------------------------------------
        # Scaling coefficient
        ('a',   (10, 1000)),
        # Power-law exponent
        ('n',   (0.02, 1.0)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (15, 52)),
        # Decay time constant
        ('t0',  (1e1, 1e5)),
        #--- Linear -----------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 10)),
    ]
    k0 = 40


class Covid19Data_SouthKorea(Covid19Data):
    countryCode = 'Korea, South'
    summaryPosition = 'E'
    bounds = [
        #--- Logistic Growth --------------------------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (1.0, 500)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.01, 0.4)),
        #--- Power-Law Component ----------------------------------------------
        # Scaling coefficient
        ('a',   (2.0, 500.0)),
        # Power-law exponent
        ('n',   (0.01, 3.5)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (17, 38)),
        # Decay time constant
        ('t0',  (0.3, 10.0)),
        #--- Linear -----------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 0.8)),
    ]
    k0 = 20


class Covid19Data_Finland(Covid19Data):
    countryCode = 'Finland'
    summaryPosition = 'NW'
    bounds = [
        #--- Logistic Growth --------------------------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (1.0, 1e6)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.01, 0.2)),
        #--- Power-Law Component ----------------------------------------------
        # Scaling coefficient
        ('a',   (0.0, 100.0)),
        # Power-law exponent
        ('n',   (0.01, 1.8)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (30, 60)),
        # Decay time constant
        ('t0',  (10, 1e4)),
        #--- Linear -----------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 15)),
    ]
    k0 = 40


class Covid19Data_Singapore(Covid19Data):
    countryCode = 'Singapore'
    summaryPosition = 'NW'
    bounds = [
        #--- Logistic Growth --------------------------------------------------
        # Total cases after exponential growth completely stopped
        ('L',   (1e4, 3e8)),
        # The growth rate, proportional to the maximum number of
        # new cases being reported per day from logistic growth 
        ('r',   (0.005, 0.15)),
        #--- Power-Law Component ----------------------------------------------
        # Scaling coefficient
        ('a',   (0.0, 40)),
        # Power-law exponent
        ('n',   (0.005, 1.9)),
        # Start of local country/region epidemic (days after 1/22/20)
        ('ts',  (40, 60)),
        # Decay time constant
        ('t0',  (1, 1e5)),
        #--- Linear -----------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 10)),
    ]
    k0 = 40


class Results(object):
    """
    I am a simple container to hold the results of modeling with one
    set of parameters.
    """
    __slots__ = ['t', 'XD', 'X', 'R', 'SSE']

    def __init__(self, t0, t1):
        self.t = np.arange(t0, t1, np.sign(t1-t0))

    def flip(self):
        for name in self.__slots__:
            Z = getattr(self, name, None)
            if Z is None: continue
            setattr(self, name, np.flip(Z))


class Evaluator(Picklable):
    """
    I evaluate fitness of one of two different models, or both in
    linear combination, against the number of new COVID-19 cases
    reported each day.

    One model is the logistic (Verhulst) model: "A biological
    population with plenty of food, space to grow, and no threat from
    predators, tends to grow at a rate that is proportional to the
    population....Of course, most populations are constrained by
    limitations on resources--even in the short run--and none is
    unconstrained forever."
    U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html}

    It requires the integration of a first-order differential
    equation:::

        xd(t, x) = x*r*(1 - x/L)

    Or, to use the modification the author of this code has made, this
    equation for logistic growth with curve flattening:::

        xd(t, x) = x*r*(2 - rc*x - rf*tanh(0.55*(t-t0)/th))
    
    The other is the power-law model with exponential cutoff
    function of Vasquez (2006):::

        xd(t) = a * (t-ts)^n * exp(-(t-ts)/t0)

    Finally, you can add a linear component to either model or the
    combination of both with the single parameter I{b}:::

        xd(t) += b*t
    
    The data in my 1-D array I{XD} contains daily numbers of B{new}
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

    @ivar XD: The actual number of new cases per day, each day, for
        the selected country or region since the first reported case
        in the Johns Hopkins dataset on January 22, 2020, B{after} my
        L{transform} has been applied.
    """
    scale_SSE = 1e-2
    f_text = None

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __len__(self):
        return len(self.data.t)
    
    def kt(self, t):
        """
        Returns an index to my I{t} and I{X} vectors for the specified
        number of days I{t} after 1/22/20.

        If I{t} is an array, an array of integer indices will be
        returned.
        """
        K = np.searchsorted(self.data.t, t)
        N = len(self)
        return np.clip(K, 0, N-1)
    
    def Xt(self, t):
        """
        Returns the value of I{X} at the specified number of days I{t}
        after 1/22/20. If the days are beyond the limits of the data
        on hand, the latest data value will be used.
        """
        return self.data.X[self.kt(t)]

    def setup(self, klass, daysAgo=0):
        """
        Call with a subclass I{klass} of L{Covid19Data} with data and
        bounds for evaluation of my model.

        Computes a differential vector I{XD} that contains the
        differences between a particular day's cumulative cases and
        the previous day's. The length this vector is the same as
        I{X}, with a zero value for the first day.
        
        Returns a C{Deferred} that fires with two equal-length
        sequences, the names and bounds of all parameters to be
        determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.

        If the supplied I{klass} has a I{second_deriv} attribute set
        C{True}, the model is considered the second derivative of the
        cumulative number of cases, rather than the first
        derivative. In other words, what gets modeled is the increase
        of the increase.

        That will cause I{X} will be returned from the initial value
        problem done by L{curve}, rather than C{f(t, X)}. Seems weird,
        but this unlikely (and, admittedly, accidental) version of the
        model worked very effectively at predicting at making
        near-term predictions in previous weeks.

        @keyword daysAgo: Set to a positive number of days ago
            to limit the latter end of the John Hopkins data. Useful
            for back-testing or when the current day's data dump is
            believed to be not yet complete.
        @type daysAgo: int
        """
        def done(null):
            # Calculate differential and show data on console
            self.XD = np.zeros_like(self.X)
            msg("Cumulative and new cases reported thus far for {}",
                self.countryCode, '-')
            xPrev = None
            for k, x in enumerate(self.X):
                xd = 0 if xPrev is None else x-xPrev
                xPrev = x
                msg("{:03d}\t{}\t{:d}\t{:d}",
                    k, self.dayText(k), int(x), int(xd))
                self.XD[k] = self.transform(xd)
            # Set up curve for all fitting and plotting
            self.curve_def()
            # Done, return names and bounds to caller
            return names, bounds

        if not issubclass(klass, Covid19Data):
            raise TypeError("You must supply a subclass of Covid19Data")
        data = self.data = klass()
        names = []; bounds = []
        self.second_deriv = getattr(klass, 'second_deriv', False)
        self.f_residuals = self.residuals_2d \
            if self.second_deriv else self.residuals_1d
        for name, theseBounds in data.bounds:
            names.append(name)
            bounds.append(theseBounds)
        return data.setup(daysAgo).addCallbacks(done, oops)

    def transform(self, XD, inverse=False):
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
            return np.sign(XD) * XD**2
        return np.sign(XD) * np.sqrt(np.abs(XD))
    
    def dayText(self, k):
        """
        Returns text indicating the date I{k} days after the first
        reported case.
        """
        firstDay = self.dates[0]
        return (firstDay + timedelta(days=k)).strftime("%m/%d")

    
    xd_logistic_flattened_text = "x*r*(2 - rc*x - rf*tanh(0.55*(t-t0)/th))"
    def xd_logistic_flattened(self, t, x, r, rc, rf, th, t0):
        """
        Modified logistic growth model (Verhulst model),
        U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html},
        with I{L} replaced by I{rc} and curve flattening (my own
        non-expert addition, caveat emptor).
        
        Given a scalar time (in days) followed by arguments defining
        curve parameters, returns the number of B{new} Covid-19
        cases expected to be reported on that day.::

            xd(t, x) = x*r*(2 - rc*x - rf*tanh(0.55*(t-t0)/th))
            
        This requires integration of a first-order differential
        equation, which is performed in L{curve}.

        The hyperbolic tangent has been a very useful function for use
        in the nonlinear models I've developed for MOSFET circuit
        simulation. it provides a smooth transition from one
        simulation regime to another. In the case of COVID-19
        infections, it seemed to me that there was such a smooth
        transition happening as the effect of social distancing
        measures have become felt in various countries/regions.

        There are three main parameters to a hyperbolic tangent
        component of a model: The maximum magnitude (never quite
        achieved, only as a limit), the rate of transition, and the
        time at which transition is happening most rapidly. In this
        modified logistic growth model, those three terms appear as
        I{rf} (max flattening reduction to I{r}), I{th} (days for
        about 1/2 of flattening effect to occur), and I{t0} (days when
        flattening effect being felt fastest).

        One jarring aspect of this modified model is that the
        effective growth rate can go negative with high enough values
        of I{rc} and I{rf}. Let's take a look at each of those
        parameters in turn.

        In a conventional logistic growth model, the cumulative number
        of population members I{x} starts to level off as C{x/L}
        starts to approach 1.0. The function C{f(t, x)} never goes
        negative because its value tapers off to zero as I{x}
        approaches I{L}. The value of I{x} never can exceed I{L}. In
        the author's modification, however, there is another term that
        subtracts from the constant (2.0 instead of 1.0):::
        
            rf*tanh(0.55*(t-t0)/th)

        Once I{t} passes I{t0} (a parameter value that is evolving to
        days in the single digits with U.S. data as of 4/1/20).
        
        The Jacobian with respect to I{x} is provided by
        L{jacobian_logistic_flattened}.
        """
        X = np.array(x)
        return r*X*(2 - rc*X - rf*np.tanh(0.55*(t-t0)/th))

    def xd_logistic_flattened_jac(self, t, x, r, rc, rf, th, t0):
        """
        Jacobian for L{xd_logistic_flattened}, with respect to I{x}::

            xd(t, x) = x*r*(2 - rc*x - rf*tanh(0.55*(t-t0)/th))
            xd(t, x) = -r*rc*x^2 - r*(2 + rf*tanh(0.55*(t-t0)/th))*x

            x2d(t, x) = -2*r*rc*x - r*(2 + rf*tanh(0.55*(t-t0)/th))
        """
        X = np.array(x)
        return -2*r*rc*X - r*(2 + rf*np.tanh(0.55*(t-t0)/th))

    
    xd_logistic_text = "x*r*(1 - x/L)"
    def xd_logistic(self, t, x, L, r):
        """
        Logistic growth model (Verhulst model),
        U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html}
        
        Given a scalar time (in days) followed by arguments defining
        curve parameters, returns the number of B{new} Covid-19
        cases expected to be reported on that day.::

            xd(t, x) = x*r*(1 - x/L)
            
        This requires integration of a first-order differential
        equation, which is performed in L{curve}.

        The Jacobian with respect to I{x} is provided by
        L{jacobian_logistic}.
        """
        x = np.array(x)
        return r*x*(1 - x/L)

    def xd_logistic_jac(self, t, x, L, r):
        """
        Jacobian for L{xd_logistic}, with respect to I{x}::

            xd(t, x) = r*x - r*x^2/L
            x2d(t, x) = r*(1 - 2*x/L)
        """
        x = np.array(x)
        return r*(1 - 2*x/L)
    
    
    xd_powerlaw_text = "a*(t-ts)^n*exp(-(t-ts)/t0)"
    def xd_powerlaw(self, t, x, a, n, ts, t0):
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

        The Jacobian with respect to I{x} is zero.
        """
        def xd(t, x):
            return a*t**n * np.exp(-t/t0) * np.ones_like(x)

        # NOTE: Using t -= ts changed t in place, which wasn't cool
        t = t - ts
        XD = np.zeros_like(x)
        if isinstance(t, np.ndarray):
            K = np.flatnonzero(t > 0)
            XD[K] = xd(t[K], x[K])
        elif t > 0:
            XD = xd(t, x)
        return XD

    def xd_powerlaw_jac(self, t, x, a, n, ts, t0):
        """
        Jacobian for L{xd_powerlaw} with respect to I{x} is constant zero,
        since it is independent of I{x}.
        """
        return np.zeros_like(x)

    
    xd_linear_text = "b"
    def xd_linear(self, t, x, b):
        """
        Linear component: Constant number of new cases reported each day.

        The Jacobian with respect to I{x} is zero.
        """
        return b * np.ones_like(x)

    def xd_linear_jac(self, t, x, b):
        """
        Jacobian for L{xd_linear} with respect to I{x} is constant zero,
        since it is constant with respect not just to I{x} but also I{t}.
        """
        return np.zeros_like(x)


    def curve_def(self):
        """
        Sets up my model function and its Jacobian.
        """
        def has_param(name):
            for thisName, bound in self.bounds:
                if thisName == name:
                    return True
        
        def curve_text():
            """
            Returns a string with the right side of the equation M{xd(t, x) =
            ...}, or M{x2d(t, x) = ...} if my I{second_deriv} is
            C{True}.
            """
            text = sub("{}(t, x) = ", "x2d" if self.second_deriv else "xd")
            text += " + ".join(self.ftList)
            return text

        def append(xd_name, N, k):
            """
            Appends my method with I{xd_name} to my I{fList}, with its
            Jacobian and a slice of the I{N} parameters following and
            including I{k} assigned to it. Appends its name to my
            I{ftList}.

            Returns the new value of I{k}.
            """
            k1 = k + N
            s = slice(k, k1)
            self.fList.append((
                getattr(self, xd_name),
                getattr(self, xd_name + '_jac'), s))
            self.ftList.append(getattr(self, xd_name + '_text'))
            return k1
                    
        if self.f_text is None:
            k = 0
            self.fList = []
            self.ftList = []
            N = len(self.bounds)
            if has_param('rf'):
                # Logistic growth (with curve flattening)
                k = append('xd_logistic_flattened', 5, k)
            elif has_param('r'):
                # Logistic growth (conventional)
                k = append('xd_logistic', 2, k)
            if has_param('n'):
                # Power law
                k = append('xd_powerlaw', 4, k)
            if has_param('b'):
                # Linear (constant differential)
                k = append('xd_linear', 1, k)
            self.f_text = curve_text()
    
    def curve(self, t0, t1, values):
        """
        Call with a starting time I{t0}, the number of days from
        first reported case, an ending time I{t1}, also in days
        from first report, and a list of parameter I{values}.

        Unless there was a problem with the ODE solver, returns an
        instance of L{Results} with the following attributes:

            - I{t}, a 1-D array containing the number of days since
              first report for each modeled day, in ascending order

            - I{XD}, the modeled number of new reported cases per
              day for each day in I{t}, and

            - I{X}, the modeled total number of reported cases for
              each day in I{t}.

        If there was a problem with the ODE solver, returns C{None}
        instead.

        Uses the logistic growth model, the power law model, or both,
        with or without a linear (constant xd) term.

        Obtains modeled numbers of new cases and total cases by
        integrating the first-order ODE C{xd(t, x)} starting at
        C{t=t0} and ending at C{t=t1}. If I{t1} is less than I{t0},
        the integration will go backwards using a sign-inverted output
        of C{xd(t, x)}. All of that is transparent to the user, and
        I{t} is in ascending order either way.
        
        If there are 2 parameters, I{L} and I{r}, only logistic growth
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
            XD = np.zeros_like(x)
            for fc, jac, ks in self.fList:
                XD_this = fc(t, x, *values[ks])
                XD += XD_this
            return XD

        def jac(t, x):
            X2D = np.zeros_like(x)
            for fc, jac, ks in self.fList:
                X2D_this = jac(t, x, *values[ks])
                X2D += X2D_this
            return [X2D]
        
        # Solve the initial value problem to get X, which in turns
        # lets us obtain values for XD, dependent as it probably is
        # (due to logistic component likely included) on X
        r = Results(t0, t1)
        sol = solve_ivp(
            f, [t0, t1], [self.Xt(t0)],
            method='Radau', t_eval=r.t, vectorized=True, jac=jac)
        if not sol.success:
            return
        r.X = sol.y[0]
        if t1 < t0: r.flip()
        r.XD = f(r.t, r.X)
        return r

    def residuals_1d(self, values):
        """
        Computes a 1-D array of transformed residuals to I{XD} from my
        L{curve}, fixed to the last known value with other derivatives
        computed backwards through my time vector I{t}.

        The residual array is left-trimmed to start with my first
        valid day I{k0}.

        Sets I{XD} and I{R} attributes to the instance of L{Results}
        that is obtained from calling L{curve} and returns the
        instance, unless a C{None} object was obtained because of an
        ODE problem, in which case C{None} is returned.
        """
        t0 = self.data.t[-1]
        t1 = self.data.t[self.k0]
        r = self.curve(t0, t1, values)
        if r is None: return
        r.XD = self.transform(r.XD)
        K = [self.kt(x) for x in r.t]
        # During setup, self.XD got its transform done for all time
        r.R = r.XD - self.XD[K]
        return r

    def residuals_2d(self, values):
        """
        Computes a 1-D array of transformed residuals to I{XD} from my
        L{curve}, fixed to 0 at 1/22/20, with other values of I{XD}
        computed via forward integration of the model.

        Unlike L{residuals}, this method integrates the model before
        subtracting the known I{XD}. Thus, the model is actually for
        the second derivative.
        
        The residual array is left-trimmed to start with my first
        valid day I{k0}.

        Sets I{XD} and I{R} attributes to the instance of L{Results}
        that is obtained from calling L{curve} and returns the
        instance, unless a C{None} object was obtained because of an
        ODE problem, in which case C{None} is returned.
        """
        t0 = self.data.t[0]
        t1 = self.data.t[-1]
        r = self.curve(t0, t1, values)
        if r is None: return 
        # Yes, we want r.X, not r.XD, because this is a
        # second-derivative model
        r.XD = self.transform(r.X)
        K = [self.kt(x) for x in r.t]
        # During setup, self.XD got its transform done for all time
        r.R = r.XD - self.XD[K]
        return r

    def residuals(self, values):
        """
        Obtains a L{Results} instance with a residuals vector I{R}
        computed between the modeled and actual values, and sets its
        I{SSE} to the sum of squared error of those residuals.

        Returns the instance with I{R} and I{SSE} set, as well as
        I{t}, I{XD}, and I{X}. If computation of the modeled values
        failed, returns C{None}.
        """
        r = self.f_residuals(values)
        if r is not None:
            r.SSE = self.scale_SSE * np.sum(np.square(r.R))
        return r
    
    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}.

        If the ODE is unsuccessful and calling L{residuals} results in
        a C{None} object, returns a crazy bad SSE.
        """
        r = self.residuals(values)
        return getattr(r, 'SSE', 1e9)


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

    @cvar N: Number of random points in extrapolated scatter plot.
    """
    plotFilePath = "covid19.png"
    minShown = 10
    daysForward = 30
    
    def __init__(self, evaluator, population):
        """
        C{Reporter(evaluator, population)}
        """
        self.ev = evaluator
        self.ev.curve_def()
        self.p = population
        self.prettyValues = population.pm.prettyValues
        self.pt = Plotter(
            3, filePath=self.plotFilePath, width=10, height=14, h2=[0, 2])
        self.pt.use_grid()
        ImageViewer(self.plotFilePath)

    @property
    def kToday(self):
        """
        Returns the current number of days since the date of first
        reported case.
        """
        firstDay = self.ev.dates[0]
        seconds_in = (date.today() - firstDay).total_seconds()
        return int(seconds_in / 3600 / 24)

    @property
    def kForToday(self):
        """
        Property: A 2-tuple with today's index (the current number of days
        since first reported case) and k_offset.
        
        The value of k_offset will be zero if latest data includes
        today, higher for each day the data is stale.
        """
        kToday = self.kToday
        return kToday, kToday - len(self.ev) + 1
    
    def clipLower(self, X):
        return np.clip(X, self.minShown, None)

    def curvePoints(self, values, k0, k1):
        """
        Obtains modeled numbers of cumulative reported cases from I{k0} to
        I{k1} days after first reported case.
        
        Given a sequence of parameter I{values} and a first I{k0} and
        second I{k1} number of days from first reported case, returns
        (1) a 1-D array I{t} with days from first reported case, in
        ascending order between I{k0} and I{k1}, inclusive, and (2) a
        1-D array I{X} with the expected number of B{cumulative} cases
        reported on each of those days.

        The value of I{X} corresponding to time I{k0} is anchored to
        the known number of reported cases then. Integration for
        extrapolation will have I{k1} greater than I{k0}. For
        backwards integration (to do historical comparisons), set
        I{k1} smaller than I{k0}. In any event, the returned I{t} and
        I{X} 1-D arrays will be in ascending order of time.
        """
        r = self.ev.curve(float(k0), float(k1), values)
        if r is None: return
        return r.t, r.X

    def add_model(self, ax, t, X):
        ax.semilogy(t, X, color='red', marker='o', linewidth=2, markersize=3)

    def add_scatter(self, ax, k0, k1, k_offset=0):
        """
        Adds points, using randomly selected parameter combos in present
        population, at times that are plotted with a slight random
        dither around each day in the future.
        """
        Nt = k1 - k0
        Ni = len(self.p)
        tc = np.empty((Nt, Ni))
        Xc = np.empty_like(tc)
        X0 = self.ev.Xt(k0)
        for ki in range(Ni):
            t, X = self.curvePoints(list(self.p[ki]), k0, k1)
            tc[:,ki], Xc[:,ki] = t+0.1*stats.norm.rvs(size=Nt), X
        tc = tc.flatten() - k_offset
        Xc = self.clipLower(Xc.flatten())
        ax.semilogy(tc, Xc, color='black', marker=',', linestyle='')

    @staticmethod
    def cases(X, k):
        return int(round(X[k]))
        
    def annotate_past(self, sp, X, k):
        """
        Given a 1-D array I{X} of case numbers that will be displayed as
        the first line in the supplied subplot I{sp}, adds an
        annotation for the actual number I{k} days after first report.

        The last item in array I{X} must be the most recent actual
        number of cases.
        """
        # Length of X, just actual data to be plotted up to most
        # recent
        N = len(X)
        # Convert index from all data to just data to be plotted
        kX = k - (len(self.ev) - N)
        if kX >= N: return
        if kX < 0: return
        sp.add_annotation(
            kX, "{}: {:,.0f}", self.ev.dayText(k), self.cases(X, kX))
    
    def model_past(self, sp, values):
        """
        Plots the past data against the best-fit model in subplot I{sp},
        given the supplied parameter I{values}, with the model curve
        anchored at the right to today's actual reported cases.
        """
        def annotate_past(k):
            self.annotate_past(sp, X_data, k)
        
        def annotate_error(k):
            xc = X_curve[k-k0]
            x = X_data[k-k0]
            xe = (xc - x)/x
            sp.add_annotation(
                k-k0, "{}: {:+.0f}%",
                self.ev.dayText(k), int(round(100*xe)), kVector=1)
            
        k0 = self.ev.k0
        kToday, k_offset = self.kForToday
        if self.ev.second_deriv:
            t, X_curve = self.curvePoints(values, k0, kToday)
        else: t, X_curve = self.curvePoints(values, kToday, k0)
        t -= k_offset
        X_data = self.clipLower(self.ev.X[self.ev.kt(t)])
        # Date of first reported case number plotted
        annotate_past(k0)
        # Error in expected vs actual reported cases, going back
        # several days starting with today
        for k in range(kToday-1, k0-1, -1):
            annotate_error(k)
        sp.add_axvline(-1)
        ax = sp.semilogy(t, X_data)
        # Add the best-fit model values for comparison
        self.add_model(ax, t, X_curve)
    
    def model_future(self, sp, values):
        """
        Plots the past data against the best-fit model in subplot I{sp},
        given the supplied parameter I{values}.

        The model curve is anchored at the most recent data point in
        the actual reported cases, and then extends beyond for
        extrapolation.

        If the most recent data point is not from today, the
        annotation will actually start in the past. The number of days
        between today (when this is run) and the date of the last data
        point is subtracted from the "days after 1/22/20" index before
        being applied to the method L{dayText}.
        """
        def annotate_past(k):
            self.annotate_past(sp, X_data, k)

        def annotate_future(daysInFuture):
            """
            Add annotation for I{daysInFuture}, returning C{True} if it
            actually was added.
            """
            k_curve = daysInFuture + k_offset
            if k_curve < 0 or k_curve >= len(X_curve):
                return
            sp.add_annotation(
                k_curve, "{}: {:,.0f}",
                self.ev.dayText(k0+daysInFuture),
                self.cases(X_curve, k_curve), kVector=1)
            return True
            
        N_back = 4
        k0, k_offset = self.kForToday
        k1 = k0 + self.daysForward
        t_data, X_data = [
            getattr(self.ev, name)[-N_back:] for name in ('t', 'X')]
        t, X_curve = self.curvePoints(values, k0, k1)
        # Today + previous few days
        for k in range(k0-N_back, k0+1):
            annotate_past(k)
        # This next week
        for k in range(7):
            annotate_future(k)
        # Days after next week, in one-week intervals
        for weeks in range(1, 10):
            if not annotate_future(7*weeks):
                break
        # Start with a few of the most recent actual data points
        sp.add_axvline(self.ev.t[-1])
        ax = sp.semilogy(t_data, X_data)
        # Add the best-fit model extrapolation
        self.add_model(ax, t-k_offset, X_curve)
        # Add scatterplot sorta-probalistic predictions
        self.add_scatter(ax, k0, k1, k_offset)

    def AICc(self, r):
        """
        Computes the Akaike Information Criterion, corrected for small
        sample size (AICc) for the residuals I{R} in the supplied
        instance I{r} of L{Results}.

        The residuals were previou between transformed model and actual
        values. The AICc is defined as follows:::

            AIC = 2*k + N*ln(SSE/N)
        
            AICc = AIC + 2*(k^2 + k)/(N-k-1)

        where SSE is the sum-of-squared error (i.e., the "residual
        sums-of-squares," as Brandmaier puts it at
        U{https://www.researchgate.net/post/What_is_the_AIC_formula},
        already computed and stored in I{r} as I{SSE}, I{N} is the
        number of samples (i.e., residual vector length), and I{k} is
        the number of model parameters, not to be confused with the
        commonly used index variable of the same name.

        Although it provides no absolute indication of fitness, the
        lower the AICc, the better the model is considered to be. Use
        it for deciding what components of the model should be
        included for a particular country/region.
        
        Returns a 3-tuple with AICc, N, k
        """
        N = len(r.R)
        k = self.p.Nd
        AIC = 2*k + N*np.log(r.SSE/N)
        AICc = AIC + 2*(k**2 + k)/(N-k-1)
        return AICc, N, k
    
    def fit_info(self, sp, r):
        """
        Adds info to subplot I{sp} about the goodness of fit for residuals
        I{R} in the supplied instance I{r} of L{Results}.
        """
        def tb(*args):
            sp.add_textBox('SW', *args)

        # Significance of non-normality
        tb("Non-normality: p < {:.4f}", stats.normaltest(r.R)[1])
        # AICc
        AICc, N, k = self.AICc(r)
        tb("AICc is {:+.2f} with SSE={:.5g}, N={:d}, k={:d}",
           AICc, r.SSE, N, k)
        
    def subplot_upper(self, sp, values):
        """
        Does the upper subplot with model fit vs data.

        Returns the I{t} vector for the actual-data dates being
        plotted.
        """
        sp.add_line('-', 2)
        sp.set_tickSpacing('x', 7.0, 1.0)
        sp.add_textBox('NW', self.ev.f_text)
        for k, nb in enumerate(self.ev.bounds):
            sp.add_textBox('SE', "{}: {:.5g}", nb[0], values[k])
        # Data vs best-fit model
        self.model_past(sp, values)

    def subplot_middle(self, sp, values, t):
        """
        Does a middle subplot with residuals between the data I{X_data}
        and modeled data I{X_curve}.
        """
        r = self.ev.residuals(values)
        if r is None: return
        sp.set_ylabel("Residual")
        sp.set_xlabel("Modeled (fitted) new cases/day (square-root transform)")
        sp.set_zeroLine(color="red", linestyle='-', linewidth=3)
        sp.add_line('')
        sp.add_marker('o', 4)
        sp.set_colors("purple")
        sp.add_textBox(
            'NW', "Residuals: Modeled vs actual new cases/day (transformed)")
        self.fit_info(sp, r)
        sp(r.XD, r.R, zorder=3)
        
    def subplot_lower(self, sp, values):
        """
        Does the lower subplot with extrapolation, starting at my I{k0}
        days from first report.
        """
        sp.add_line('-', 2)
        sp.set_tickSpacing('x', 7.0, 1.0)
        self.model_future(sp, values)
        
    def __call__(self, values, counter, SSE):
        """
        Prints out a new best parameter combination and its curve vs
        observations, with lots of extrapolation to the right.
        """
        def tb(*args):
            if len(args[0]) < 3:
                pos = args[0]
                args = args[1:]
            else: pos = self.ev.summaryPosition
            sp.add_textBox(pos, *args)
        
        # Make a frozen local copy of the values list to work with, so
        # outdated stuff doesn't get plotted
        values = list(values)
        msg(0, self.prettyValues(
            values, "SSE={:.5g} on eval {:d}:", SSE, counter), 0)
        self.pt.set_title(
            "Modeled (red) vs Actual (blue) Reported Cases of COVID-19: {}",
            self.ev.countryCode)
        self.pt.set_ylabel("Reported Cases")
        self.pt.set_xlabel("Days after January 22, 2020")
        self.pt.use_minorTicks('x', 1.0)
        with self.pt as sp:
            tb('S', "Reported cases in {} vs days after first case.",
               self.ev.countryCode)
            tb('S', "Annotations show residuals between model and data.")
            t_data = self.subplot_upper(sp, values)
            self.subplot_middle(sp, values, t_data)
            tb("Expected cases reported in {} vs days after first",
               self.ev.countryCode)
            tb("case. Dots show daily model predictions for each")
            tb("of a final population of {:d} evolved parameter", len(self.p))
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
            self.N_cores = 1
            self.q = ThreadQueue(returnFailure=True)
        else:
            self.N_cores = args.N if args.N else ProcessQueue.cores()-1
            self.q = ProcessQueue(self.N_cores, returnFailure=True)
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
    def reEvaluate(self):
        """
        Re-evaluates loaded individuals, keeping the previous history if
        not too much has changed SSE-wise.
        """
        def done(i, k, SSE_prev):
            SSE = float(i.SSE)
            if not SSE or np.isinf(SSE_prev) or np.isinf(SSE):
                info = ": !!!"
            else:
                rd = (SSE-SSE_prev) / SSE_prev
                if abs(rd) > 1E-3:
                    info = sub(": {:+.1f}%", 100*rd)
                    iChanged.add(i)
                else: info = ""
            msg(2, "{:03d}  {:>8.1f} --> {:>8.1f}  {}", k, SSE_prev, SSE, info)
        
        msg(-1, "Re-evaluating {:d} loaded Individuals...", len(self.p), '-')
        iChanged = set()
        dt = DeferredTracker(interval=0.05)
        yield self.p.history.purgePop()
        for k, i in enumerate(self.p):
            SSE_prev = float(i.SSE)
            d = i.evaluate().addCallback(done, k, SSE_prev)
            dt.put(d)
            yield dt.deferUntilFewer(self.N_cores)
        yield dt.deferToAll()
        msg("")
    
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
        reporter = Reporter(self.ev, self.p)
        self.p.addCallback(reporter)
        if len(self.p):
            yield self.reEvaluate()
        elif self.p.running is None:
            OK = yield self.p.setup()
            if not OK:
                yield self.shutdown()
        self.p.initialize()
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
