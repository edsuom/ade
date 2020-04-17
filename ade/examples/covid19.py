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

    @cvar re_region_yes: Set this to a compiled regular expression
        object for the region name to only include matching regions.

    @cvar re_region_no: Set this to a regular expression object for
        the region name to B{exclude} matching regions.
    
    @see: The L{Data} base class.
    """
    basename = "covid19"
    reDate = re.compile(r'([0-9]+)/([0-9]+)/([0-9]+)')
    modelPosition = 'NW'
    summaryPosition = 'NW'

    re_region_yes = None
    re_region_no = None

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
            if self.re_region_no and self.re_region_no.search(rvl[0]):
                continue
            if self.re_region_yes and not self.re_region_yes.search(rvl[0]):
                msg("Not included: {}, {}", rvl[0], rvl[1])
                continue
            yield np.array([int(x) for x in rvl[4:] if x])
                
    def parseValues(self, result, daysAgo):
        """
        Parses the date and reported case numbers from the lists of
        text-values in the I{result}, limiting the latest data to the
        specified number of I{daysAgo}.

        To not limit latest data, set I{daysAgo} to zero.
        """
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
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (2e6, 3.3e8)),
        # The initial exponential growth rate
        ('r',   (0.32, 0.85)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.90, 1.0)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (12, 25)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (51, 67)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 250)),
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
        # Upper limit to number of total cases
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
        ('tc',  (10, 1e6)),
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
        # Upper limit to number of total cases
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

    The I{tc} parameter goes absurdly high with no significant effect
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
        ('tc',  (10, 1e6)),
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
        ('tc',  (2, 600)),
    ]
    bounds_l = [
        #--- Linear (list this last) ------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0.0, 0.1)),
        #----------------------------------------------------------------------
    ]
    bounds = bounds_lg + bounds_pl + bounds_l
    k0 = 40


class Covid19Data_Spain(Covid19Data):
    countryCode = 'Spain'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (3e5, 4.7e7)),
        # The initial exponential growth rate
        ('r',   (0.15, 0.23)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.65, 1.0)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (5, 12)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (65, 69)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 40)),
        #----------------------------------------------------------------------
    ]
    k0 = 51


class Covid19Data_Italy(Covid19Data):
    countryCode = 'Italy'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is <
        # population of 6e7)
        ('L',   (2e5, 5.5e7)),
        # The initial exponential growth rate
        ('r',   (0.1, 0.3)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.83, 0.98)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (9, 17)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (55, 64)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 400)),
        #----------------------------------------------------------------------
    ]
    k0 = 51

    
class Covid19Data_Germany(Covid19Data):
    countryCode = 'Germany'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (3e5, 8.4e7)),
        # The initial exponential growth rate
        ('r',   (0.22, 0.55)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.88, 1.0)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (10, 21)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (52, 65)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 130)),
        #----------------------------------------------------------------------
    ]
    k0 = 40


class Covid19Data_France(Covid19Data):
    # As of 4/4, there was such a huge jump that a good curve fit
    # seemed implausible. Residuals were significantly non-normal.
    countryCode = 'France'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (3e5, 6.5e7)),
        # The initial exponential growth rate
        ('r',   (0.05, 0.4)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.2, 0.9)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (1, 15)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0',  (48, 79)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 50)),
        #----------------------------------------------------------------------
    ]
    k0 = 40


class Covid19Data_Iran(Covid19Data):
    countryCode = 'Iran'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound <
        # population of 8.4e7)
        ('L',   (8e4, 5e5)),
        # The initial exponential growth rate
        ('r',   (0.05, 0.35)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.1, 0.95)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (1, 10)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (37, 85)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 200)),
        #----------------------------------------------------------------------
    ]
    k0 = 46


class Covid19Data_UK(Covid19Data):
    countryCode = 'United Kingdom'
    re_region_no = re.compile('.')
    summaryPosition = 'E'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound <<
        # population of 6.8e7)
        ('L',   (5e4, 2e5)),
        # The initial exponential growth rate
        ('r',   (0.20, 0.27)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.10, 0.35)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (1, 3)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (65, 67)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 40)),
        #----------------------------------------------------------------------
    ]
    k0 = 42

    
class Covid19Data_SouthKorea(Covid19Data):
    countryCode = 'Korea, South'
    modelPosition = 'M'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (5e4, 5e7)),
        # The initial exponential growth rate
        ('r',   (0.1, 0.5)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.95, 1.0)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (3, 7)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (37, 43)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 150)),
        #----------------------------------------------------------------------
    ]
    k0 = 34


class Covid19Data_Singapore(Covid19Data):
    countryCode = 'Singapore'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (2e3, 5.9e6)),
        # The initial exponential growth rate
        ('r',   (0.08, 0.25)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.3, 0.7)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (0.5, 6)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (56, 69)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 2)),
        #----------------------------------------------------------------------
    ]
    k0 = 40


class Covid19Data_Finland(Covid19Data):
    countryCode = 'Finland'
    bounds = [
        #--- Logistic Growth with curve flattening (L, r, rf, th, t0) ---------
        # Upper limit to number of total cases (upper bound is population)
        ('L',   (2e3, 5.5e6)),
        # The initial exponential growth rate
        ('r',   (0.05, 0.3)),
        # Max fractional reduction in effective r from curve flattening effect
        # (0.0 for no flattening, 1.0 to completely flatten to zero growth)
        ('rf',  (0.2, 0.7)),
        # Time for flattening to have about half of its full effect (days)
        ('th',  (1, 6)),
        # Time (days after 1/22/20) at the middle of the transition
        # from regular logistic-growth behavior to fully flattened
        ('t0', (58, 73)),
        #--- Linear (b) -------------------------------------------------------
        # Constant number of new cases reported each day since beginning
        ('b',   (0, 10)),
        #----------------------------------------------------------------------
    ]
    k0 = 53


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


class Model(object):
    """
    I define a model for the number of new daily reported cases for
    each day after 1/22/20. The model is a linear combination of one
    or more sub-models:

        - One of two mutually exclusive logistic growth models, a
          conventional one and the author's own curve-flattened version,
    
        - A power-law model with exponential decay, and

        - A linear model (constant number of new daily reported cases).

    The conventional logistic growth model is the logistic (Verhulst)
    model: "A biological population with plenty of food, space to
    grow, and no threat from predators, tends to grow at a rate that
    is proportional to the population....Of course, most populations
    are constrained by limitations on resources--even in the short
    run--and none is unconstrained forever."
    U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html}

    It requires the integration of a first-order differential
    equation:::

        xd(t, x) = x*r*(1 - x/L)

    The author of this code (EAS) proposes a modification to that
    model with a C{flatten} function reducing the growth rate with a
    smooth transition to a curve-flattened regime. Here is the
    equation for logistic growth with curve flattening:::

        xd(t, x) = r*x*(1 - x/L)*flatten(t)
        flatten(t) = 0.5*rf*(1 - tanh(1.1*(t-t0)/th)) - rf + 1
    
    where

        - I{L} is the maximum number of possible cases,

        - I{rf} is the fractional reduction in growth rate at the
          conclusion of the flattened-curve regime,

        - I{th} is the time interval (in days) over which the middle
          half of a smooth transition occurs from regular
          logistic-growth behavior to a fully flattened curve with
          I{rf} the non-flattened growth rate, and

        - I{t0} is the time (in days after 1/22/20) at the middle of
          the transition from regular logistic-growth behavior to a
          fully flattened curve.
    
    The other model is the power-law model with exponential cutoff
    function of Vasquez (2006):::

        xd(t) = a * (t-ts)^n * exp(-(t-ts)/t0)

    Finally, you can add a linear component to either model or the
    combination of both with the single parameter I{b}:::

        xd(t) += b*t
    """
    def __init__(self, names, second_deriv=False):
        """
        C{Model(names, second_deriv=False)}
        
        Sets up my model function and its Jacobian with a list of
        parameter I{names}, in the order that their I{values} will
        appear in L{__call__}.

        @keyword second_deriv: Set C{True} to have the model consider
            the second derivative of the number of reported cases
            instead of the first. (Not supported or recommended.)
        """
        def has_param(name):
            return name in names
        
        def curve_text():
            """
            Returns a string with the right side of the equation M{xd(t, x) =
            ...}, or M{x2d(t, x) = ...} if my I{second_deriv} is
            C{True}.
            """
            text = sub("{}(t, x) = ", "x2d" if second_deriv else "xd")
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
                    
        k = 0
        self.fList = []
        self.ftList = []
        self.second_deriv = second_deriv
        N = len(names)
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
        
    #--- Logistic Growth Model with Curve Flattening --------------------------
    
    xd_logistic_flattened_text \
        = "r*x*(1 - x/L)*(0.5*rf*(1 - tanh(1.1*(t-t0)/th)) - rf + 1)"

    def flatten(self, t, rf, th, t0):
        """
        Impements the flattening function on time vector I{t} (in days)
        that scales growth rate:::

            flatten(t, rf, th, t0) = 0.5*rf*(1 - tanh(1.1*(t-t0)/th)) - rf + 1
        """
        return 0.5*rf*(1 - np.tanh(1.1*(t-t0)/th)) - rf + 1
    
    def xd_logistic_flattened(self, t, x, L, r, rf, th, t0):
        """
        Modified logistic growth model (Verhulst model),
        U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html},
        with curve flattening (my own non-expert addition, caveat
        emptor).
        
        Given a scalar time (in days) followed by arguments defining
        curve parameters, returns the number of B{new} Covid-19
        cases expected to be reported on that day.::

            xd(t, x) = r*x*(1 - x/L)*flatten(t, rf, th, t0)
            flatten(t, rf, th, t0) = 0.5*rf*(1 - tanh(1.1*(t-t0)/th)) - rf + 1
            
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
        time at which transition is happening most rapidly, i.e., the
        middle. In this modified logistic growth model, those three
        terms appear as I{rf} (max fractional flattening reduction to
        I{r}), I{th} (days for about 1/2 of flattening effect to
        occur), and I{t0} (days after 1/22/20 for the middle, when
        flattening effect being felt fastest).

        Let's take a look at each of those parameters in turn.

        In a conventional logistic growth model, the cumulative number
        of population members I{x} starts to level off as C{x/L}
        starts to approach 1.0. The conventional logistic-growth
        function C{f(t, x)} never goes negative because its value
        tapers off to zero as I{x} approaches I{L}. The value of I{x}
        never can exceed I{L}. In the author's modification, there is
        another term, C{flatten(t)} that further scales the growth as
        a function of time:::
        
            flatten(t, rf, th, t0) = 0.5*rf*(1 - tanh(1.1*(t-t0)/th)) - rf + 1

        Once I{t} passes I{t0} (a parameter value that is evolving to
        days in the single digits with U.S. data as of 4/1/20), the
        growth will have been reduced half of the way from its
        unmodified value to a full reduction of I{rf}, where I{rf} is
        the fractional amount of the final reduction. For example, if
        I{rf} is 0.2, the growth rate at I{t0} will be 10% lower than
        it would have been without any curve flattening, and then will
        approach the limit of a 20% growth rate reduction.
        
        The Jacobian with respect to I{x} is provided by
        L{jacobian_logistic_flattened}.
        """
        X = np.array(x)
        return r*X*(1 - X/L)*self.flatten(t, rf, th, t0)
    
    def xd_logistic_flattened_jac(self, t, x, L, r, rf, th, t0):
        """
        Jacobian for L{xd_logistic_flattened}, with respect to I{x}::

            xd(t, x) = r*x*(1 - r*x/L)*flatten(t)
            xd(t, x) = r*flatten(t)*x - r/L*flatten(t)*x^2

            x2d(t, x) = -2*r/L*flatten(t)*x - r*flatten(t)
            x2d(t, x) = -r*flatten(t)*(2*x/L + 1)
        """
        X = np.array(x)
        return -r*self.flatten(t, rf, th, t0)*(2*X/L + 1)
    
    #--- Logistic Growth Model (conventional) ---------------------------------
    
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

    #--- Power Law with Exponential Decay -------------------------------------
    
    xd_powerlaw_text = "a*(t-ts)^n*exp(-(t-ts)/tc)"
    def xd_powerlaw(self, t, x, a, n, ts, tc):
        """
        Power law model,
        U{https://www.medrxiv.org/content/10.1101/2020.02.16.20023820v2.full.pdf}
        
        Given a scalar time vector (in days) followed by arguments
        defining curve parameters, returns the number of B{new}
        Covid-19 cases expected to be reported on that day.::

            xd(t, x) = a * (t-ts)^n * exp(-(t-ts)/tc)

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
            return a*t**n * np.exp(-t/tc) * np.ones_like(x)

        # NOTE: Using t -= ts changed t in place, which wasn't cool
        t = t - ts
        XD = np.zeros_like(x)
        if isinstance(t, np.ndarray):
            K = np.flatnonzero(t > 0)
            XD[K] = xd(t[K], x[K])
        elif t > 0:
            XD = xd(t, x)
        return XD

    def xd_powerlaw_jac(self, t, x, a, n, ts, tc):
        """
        Jacobian for L{xd_powerlaw} with respect to I{x} is constant zero,
        since it is independent of I{x}.
        """
        return np.zeros_like(x)

    #--- Linear (constant number of new daily cases) --------------------------
    
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

    #--------------------------------------------------------------------------
    
    def __call__(self, values, t0, t1, X0):
        """
        Call my instance with a list of parameter I{values}, starting
        I{t0} and ending times I{t1}, both in days from first report,
        and the fixed (known) model value I{X0} at I{t0}.

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
            f, [t0, t1], [X0],
            method='Radau', t_eval=r.t, vectorized=True, jac=jac)
        if not sol.success:
            return
        r.X = sol.y[0]
        if t1 < t0: r.flip()
        r.XD = f(r.t, r.X)
        return r
    
            
class Evaluator(Picklable):
    """
    I evaluate fitness of the model defined by my L{Model} instance
    I{model} against the number of new COVID-19 cases reported each
    day.

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
        bounds for evaluation of the instance I{model} of L{Model}
        that I construct.

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
            # Set up model for all fitting and plotting
            self.model = Model(names, second_deriv)
            # Done, return names and bounds to caller
            return names, bounds

        if not issubclass(klass, Covid19Data):
            raise TypeError("You must supply a subclass of Covid19Data")
        data = self.data = klass()
        names = []; bounds = []
        second_deriv = getattr(klass, 'second_deriv', False)
        self.f_residuals = self.residuals_2d \
            if second_deriv else self.residuals_1d
        for name, theseBounds in data.bounds:
            names.append(name)
            bounds.append(theseBounds)
        return data.setup(daysAgo).addCallbacks(done, oops)

    def transform(self, XD=None, inverse=False):
        """
        Applies a transform to the numbers of new cases per day each day
        I{XD}, real or modeled. Set I{inverse} C{True} to apply the
        inverse of the transform.

        To use your own (presumably modeled) I{XD}, supply it via that
        keyword. Otherwise, I will do an inverse transform on my own
        I{XD} attribute. In that case, the I{inverse} keyword is
        ignored because it is assumed to be C{True}.

        The crude transform currently used is just a square root of
        the absolute magnitude, with sign preserved.  The seems like a
        reasonable initial compromise between a log transform (useful
        for a purely exponential model), and not transforming at
        all. Will investigate Cox-Box as an option.
        """
        if XD is None:
            XD = self.XD
            inverse = True
        if inverse:
            return np.sign(XD) * XD**2
        return np.sign(XD) * np.sqrt(np.abs(XD))
    
    def dayText(self, k=None):
        """
        Returns text indicating the date I{k} days after the first
        reported case.

        To get text for the last (most recent) reported case, leave
        I{k} C{None}.
        """
        if k is None:
            day = self.dates[-1]
        else:
            firstDay = self.dates[0]
            day = firstDay + timedelta(days=k)
        return day.strftime("%m/%d")

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
        r = self.model(values, t0, t1, self.Xt(t0))
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
        r = self.model(values, t0, t1, self.Xt(t0))
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
    
    def __init__(self, evaluator, population, includeRatio=False):
        """
        C{Reporter(evaluator, population, includeRatio=False)}
        """
        self.ev = evaluator
        self.p = population
        self.includeRatio = includeRatio
        self.prettyValues = population.pm.prettyValues
        if includeRatio:
            N = 4
            height = 19
        else:
            N = 3
            height = 17
        self.pt = Plotter(
            1, N,
            filePath=self.plotFilePath, width=12, height=height, h2=[0, 2])
        self.pt.use_grid()
        self.pt.set_fontsize('textbox', 12)
        ImageViewer(self.plotFilePath)

    @property
    def kToday(self):
        """
        Returns the current (as of the last known date) number of days
        since the date of first reported case.

        Calling this "kToday" is a holdover from when there was no
        option to right-trim the dataset with the '-d' command-line
        option for backtesting, when the most recent known date was
        always assumed to be the current date.
        """
        firstDay = self.ev.dates[0]
        # Unused code from the previous stupid version where current
        # date was used, not last known, left in as a comment for
        # historical reference
        # ---------------------------------------------------------------------
        # seconds_in = (date.today() - firstDay).total_seconds()
        # ---------------------------------------------------------------------
        seconds_in = (self.ev.dates[-1] - firstDay).total_seconds()
        return int(seconds_in / 3600 / 24)

    @property
    def kForToday(self):
        """
        Property: A 2-tuple with the last known date's index (the current
        number of days since first reported case) and k_offset.
        
        The value of k_offset will be zero if latest data includes
        today, higher for each day the data is stale.

        Calling this "kForToday" is a holdover from when there was no
        option to right-trim the dataset with the '-d' command-line
        option for backtesting, when the most recent known date was
        always assumed to be the current date.
        """
        kToday = self.kToday
        return kToday, kToday - len(self.ev) + 1
    
    def clipLower(self, X):
        return np.clip(X, self.minShown, None)

    def curvePoints(self, values, k0, k1):
        """
        Obtains modeled numbers of new daily cases and cumulative reported
        cases from I{k0} to I{k1} days after first reported case.
        
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
        I{k1} smaller than I{k0}. In any event, the I{t} and I{X} 1-D
        arrays in the returned I{r} object will be in ascending order
        of time.

        Returns an instance I{r} of I{Results} with the modeled arrays.
        """
        t0, t1 = [float(x) for x in [k0, k1]]
        r = self.ev.model(values, t0, t1, self.ev.Xt(t0))
        if r is None: return
        return r

    def add_model(self, ax, t, X, semilog=False):
        f = getattr(ax, 'semilogy' if semilog else 'plot')
        f(t, X, color='red', marker='o', linewidth=2, markersize=3)

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
            r = self.curvePoints(list(self.p[ki]), k0, k1)
            tc[:,ki], Xc[:,ki] = r.t+0.1*stats.norm.rvs(size=Nt), r.X
        tc = tc.flatten() - k_offset
        Xc = self.clipLower(Xc.flatten())
        ax.semilogy(tc, Xc, color='black', marker=',', linestyle='')

    @staticmethod
    def cases(X, k):
        return int(round(X[k]))
        
    def annotate_past(self, sp, X, k=None):
        """
        Given a 1-D array I{X} of case numbers that will be displayed as
        the first line in the supplied subplot I{sp}, adds an
        annotation for the actual number I{k} days after first report.

        The last item in array I{X} must be the most recent actual
        number of cases.

        To just annotate the last item in I{X}, leave I{k} C{None}.
        """
        if k is None:
            kX = -1
        else:
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

        Returns a 4-tuple of equal-length 1-D arrays with (1) the
        actual time I{t}, (2) the actual cumulative numbers of cases
        I{X_data}, (3) the actual numbers of new daily cases, and (4)
        the modeled number of new daily cases.
        """
        def annotate_error(k):
            kk = k - k0 - 1
            xc = X_curve[kk]
            x = X_data[kk]
            xe = (xc - x)/x
            sp.add_annotation(
                kk, "{}: {:+.1f}%",
                self.ev.dayText(k), int(round(100*xe)), kVector=1)
            
        k0 = self.ev.k0
        kToday, k_offset = self.kForToday
        if self.ev.model.second_deriv:
            r = self.curvePoints(values, k0, kToday)
        else: r = self.curvePoints(values, kToday, k0)
        t, X_curve = r.t, r.X
        t -= k_offset
        K = self.ev.kt(t)
        X_data = self.clipLower(self.ev.X[K])
        XD = self.ev.transform()[K]
        # Date of first reported case number plotted
        self.annotate_past(sp, X_data, k0+1)
        # Error in expected vs actual reported cases, going back
        # several days starting with today
        kList = range(kToday, k0, -1)
        Nfd = len(kList) / 15
        N_data = len(X_data)
        for kk, k in enumerate(kList):
            k_data = N_data - kk - 1
            if not k_data % 7:
                sp.add_axvline(k_data)
            if kk > 10:
                if Nfd and kk % Nfd: continue
            annotate_error(k)
        ax = sp.semilogy(t, X_data)
        # Add the best-fit model values for comparison
        self.add_model(ax, t, X_curve, semilog=True)
        return t, X_data, XD, r.X, r.XD
    
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

        Returns the modeled time I{t}, cumulative cases I{X_curve},
        and new daily cases I{XD}.
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
        r = self.curvePoints(values, k0, k1)
        t = r.t - k_offset
        X_curve = r.X
        # Vertical line "today"
        t0 = self.ev.t[-1]
        sp.add_axvline(t0)
        # "Today" + previous few days
        for k in range(k0-N_back, k0+1):
            annotate_past(k)
        # This next week
        for k in range(1, 7):
            annotate_future(k)
        # Days after next week, in one-week intervals
        for weeks in range(1, 10):
            if not annotate_future(7*weeks):
                break
            sp.add_axvline(t0 + 7*weeks)
        # Start with a few of the most recent actual data points
        sp.add_axvline(self.ev.t[-1])
        ax = sp.semilogy(t_data, X_data)
        # Add the best-fit model extrapolation
        self.add_model(ax, t, X_curve, semilog=True)
        # Add scatterplot sorta-probalistic predictions
        self.add_scatter(ax, k0, k1, k_offset)
        return t, X_curve, r.XD
    
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
        tb("Non-normality: p = {:.4f}", stats.normaltest(r.R)[1])
        # AICc
        AICc, N, k = self.AICc(r)
        tb("AICc is {:+.2f} with SSE={:.5g}, N={:d}, k={:d}",
           AICc, r.SSE, N, k)
        
    def subplot_upper(self, sp, values):
        """
        Does the upper subplot with model fit vs data.

        Returns the I{t} I{X} vectors for the actual data being
        plotted, the I{XD} vector for the actual data, the I{X} vector
        for the modeled data, and the I{XD} vector for the modeled
        data.
        """
        sp.add_line('-', 2)
        sp.set_tickSpacing('x', 7.0, 1.0)
        sp.add_textBox(self.ev.modelPosition, self.ev.model.f_text)
        for k, nb in enumerate(self.ev.bounds):
            sp.add_textBox('SE', "{}: {:.5g}", nb[0], values[k])
        # Data vs best-fit model
        return self.model_past(sp, values)

    def subplot_middle(self, sp, values, t):
        """
        Does a middle subplot with residuals between the data I{X_data}
        and modeled data I{X_curve}, given model parameter I{values}
        and evaluation times (days after 1/22/20) I{t}.
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

        Returns the I{t} and I{X} vectors for the future modeled data
        being plotted, plus the modeled I{XD} vector.
        """
        sp.add_line('-', 2)
        sp.set_tickSpacing('x', 7.0, 1.0)
        return self.model_future(sp, values)

    def subplot_bottom(self, sp, ta, Ra, tm, Rm):
        """
        Draws a fourth subplot on the very bottom showing the modeled
        ratio between new and total cases, both past (actual) and
        future (modeled).

        Call with actual-data time and ratio vectors I{ta}, I{Ra} and
        modeled time and ratio vectors I{tm}, I{Rm}.
        """
        Ra = 100 * Ra
        Rm = 100 * Rm
        sp.add_axvline(-1)
        sp.add_annotation(
            ta[-1], "{} had {:.1f}% of all\ncase reports thus far",
            self.ev.dayText(), Ra[-1])
        sp.add_textBox(
            'NE', "New cases each day as a percentage of total cases thus far")
        sp.set_ylabel("% New")
        ax = sp(ta, Ra)
        self.add_model(ax, tm, Rm)
    
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
            ta, Xa, XDa, Xam, XDam = self.subplot_upper(sp, values)
            self.subplot_middle(sp, values, ta)
            tb("Expected cases reported in {} vs days after",
               self.ev.countryCode)
            tb("first case. Dots show daily model predictions for each")
            tb("of a final population of {:d} evolved parameter", len(self.p))
            tb("combinations. Annotations show actual values in")
            tb("past, best-fit projected values in future.")
            for line in DISCLAIMER.split('\n'):
                sp.add_textBox("SE", line)
            tm, Xm, XDm = self.subplot_lower(sp, values)
            if self.includeRatio:
                tm = np.concatenate([ta, tm])
                Xm = np.concatenate([Xam, Xm])
                XDm = np.concatenate([XDam, XDm])
                self.subplot_bottom(sp, ta, XDa/Xa, tm, XDm/Xm)
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
        reporter = Reporter(self.ev, self.p, args.r)
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
args('-m', '--maxiter', 75, "Maximum number of DE generations to run")
args('-e', '--bitter-end', "Keep working to the end even with little progress")
args('-p', '--popsize', 40, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.7, "DE Crossover rate CR")
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
args('-r', '--include-ratio',
     "Include subplot with ratio of new vs cumulative cases")
args("<Country/State Name> [<pickle file>]")
args(main)
