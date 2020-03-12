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
linear+exponential model of the number of reported cases vs time in
hours.

This example reads a CSV file you may obtain from John Hopkins (one
that was obtained as of the date of this commit/version). Each line
contains the number of reported cases for one region of the world.

The datafile C{time_series_19-covid-Confirmed.csv} is for educational
purposes and reproduced under the Terms of Use for the data file,
which is as follows:

    This GitHub repo and its contents herein, including all data,
    mapping, and analysis, copyright 2020 Johns Hopkins University,
    all rights reserved, is provided to the public strictly for
    educational and academic research purposes.  The Website relies
    upon publicly available data from multiple sources, that do not
    always agree. The Johns Hopkins University hereby disclaims any
    and all representations and warranties with respect to the
    Website, including accuracy, fitness for use, and merchantability.
    Reliance on the Website for medical guidance or use of the Website
    in commerce is strictly prohibited.

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

    f(t) = a*t + b*exp(c*t) + d

where I{f} is the number of reported cases and I{t} is the time, in
hours, since the first report in this dataset.
"""

import re
from datetime import date, timedelta

import numpy as np

from twisted.internet import reactor, defer

from asynqueue.process import ProcessQueue
from yampex.plot import Plotter

from ade.population import Population
from ade.de import DifferentialEvolution
from ade.image import ImageViewer
from ade.util import *

from data import Data


class Covid19Data(Data):
    """
    Run L{setup} on my instance to decompress and load the
    voc.csv.bz2 CSV file.

    The CSV file isn't included in the I{ade} package and will
    automatically be downloaded from U{edsuom.com}. Here's the privacy
    policy for my site (it's short, as all good privacy policies
    should be)::

        Privacy policy: I don’t sniff out, track, or share anything
        identifying individual visitors to this site. There are no
        cookies or anything in place to let me see where you go on the
        Internetthat’s creepy. All I get (like anyone else with a web
        server), is plain vanilla server logs with “referral” info
        about which web page sent you to this one.

    @see: The L{Data} base class.
    """
    basename = "covid19"
    reDate = re.compile(r'([0-9]+)/([0-9]+)/([0-9]+)')

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
            thisDate = date(g2i(3), g2i(1), g2i(2))
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
            if rvl[0] not in self.regionCodes:
                continue
            if rvl[1] not in self.countryCodes:
                continue
            print rvl
            yield np.array([int(x) for x in rvl[4:]])
            
    def parseValues(self, result):
        self.parseDates(result)
        self.X = np.zeros(len(self))
        for Y in self.valuerator(result):
            self.X += Y
                   
    def setup(self):
        """
        Calling this gets you a C{Deferred} that fires when setup is done
        and my I{t} and I{X} ivars are ready.
        """
        return self.load().addCallback(self.parseValues)


class Covid19Data_US(Covid19Data):    
    countryCodes = ['US']
    regionCodes = [
        'Alabama',              'Alaska',       'Arizona',      'Arkansas',
        'California',           'Colorado',     'Connecticut',  'Delaware',
        'Florida',              'Georgia',      'Hawaii',
        'Idaho',                'Illinois',     'Indiana',      'Iowa',
        'Kansas',               'Kentucky',     'Louisiana',    'Maine',
        'Maryland',             'Massachusetts',
        'Michigan',             'Minnesota',    'Mississippi',  'Missouri',
        'Montana',              'Nebraska',     'Nevada',       'New Hampshire',
        'New Jersey',           'New Mexico',   'New York',     'North Carolina',
        'North Dakota',         'Ohio',         'Oklahoma',     'Oregon',
        'Pennsylvania',         'Rhode Island', 'South Carolina',
        'South Dakota',         'Tennessee',    'Texas',        'Utah',
        'Vermont',              'Virginia',     'Washington',
        'West Virginia',        'Wisconsin',    'Wyoming',
    ]


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
    """
    plotFilePath = "voc.png"
    N_curve_plot = 200
    extrapolationMultiple = 3

    def __init__(self, evaluator, population):
        """
        C{Reporter(evaluator, population)}
        """
        self.ev = evaluator
        self.prettyValues = population.pm.prettyValues
        self.pt = Plotter(
            1, filePath=self.plotFilePath, width=16, height=9)
        self.pt.use_grid()
        ImageViewer(self.plotFilePath)
    
    def __call__(self, values, counter, SSE):
        """
        Prints out a new best parameter combination and its curve vs
        observations, with lots of extrapolation to the right.
        """
        def titlePart(*args):
            titleParts.append(sub(*args))

        SSE_info = sub("SSE={:g}", SSE)
        titleParts = []
        titlePart("Cases vs Time (hr)")
        titlePart(SSE_info)
        titlePart("k={:d}", counter)
        msg(0, self.prettyValues(values, SSE_info+", with"), 0)
        with self.pt as sp:
            t = self.ev.t
            sp.set_title(", ".join(titleParts))
            # Model versus observations
            sp.add_line('-', 1)
            sp.set_xlabel("Hours")
            sp.set_ylabel("N")
            sp.set_zeroLine(values[-1])
            sp.add_annotation(0, self.ev.dates[0])
            sp.add_annotation(-1, self.ev.dates[-1])
            ax = sp(t, self.ev.X)
            tm = np.linspace(
                t.min(), self.extrapolationMultiple*t.max(), self.N_curve_plot)
            X_curve = self.ev.curve(tm, *values)
            ax.plot(tm, X_curve, color='red', marker='o', markersize=2)
        self.pt.show()


class Evaluator(Picklable):
    """
    I evaluate fitness of the function::

        x(t) = a*t + b*exp(c*t) + d

    against data for daily numbers of reported COVID-19 cases, where
    I{x} is the number of expected reported cases and I{t} is the time
    since the first observation in hours.
    
    Construct an instance of me, run the L{setup} method, and wait (in
    non-blocking Twisted-friendly fashion) for the C{Deferred} it
    returns to fire. Then call the instance a bunch of times with
    parameter values for a L{curve} to get a (deferred)
    sum-of-squared-error fitness of the curve to the actual data.
    """
    scale_SSE = 100
    bounds = {
        # Linear term (increase per hour in expected number of new
        # daily cases)
        'a':    (0.2,    10),
        # Exponential term (expected number of exponentially
        # increasing new daily cases at outset)
        'b':    (0.03,   20),
        # Scaling coefficient (number of daily case doublings expected
        # per ... hrs)
        'c':    (0.001,  0.02),
        # Constant term (number of reported cases at outset)
        'd':    (1,     200),
    }

    def setup(self):
        """
        Returns a C{Deferred} that fires with two equal-length sequences,
        the names and bounds of all parameters to be determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.
        """
        def done(null):
            for name in ('t', 'X', 'dates'):
                setattr(self, name, getattr(data, name))
            return names, bounds

        bounds = []
        names = sorted(self.bounds.keys())
        for name in names:
            bounds.append(self.bounds[name])
        # The data
        data = Covid19Data_US()
        return data.setup().addCallbacks(done, oops)

    def curve(self, t, *args):
        """
        Given a 1-D time vector followed by arguments defining curve
        parameters, returns a 1-D vector of expected Covid-19 cases to
        be reported over that time, in hours.

        The model implements this equation:

        M{x = a*t + b*np.exp(c*t) + d}
        """
        return args[0]*t + args[1]*np.exp(args[2]*t) + args[3]
    
    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}.
        """
        X_curve = self.curve(self.t, *values)
        return self.scale_SSE * np.sum(np.square(X_curve - self.X))

        
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
        t0 = time.time()
        args = self.args
        names_bounds = yield self.ev.setup().addErrback(oops)
        self.p = Population(
            self.evaluate,
            names_bounds[0], names_bounds[1], popsize=args.p)
        yield self.p.setup().addErrback(oops)
        reporter = Reporter(self.ev, self.p)
        self.p.addCallback(reporter)
        F = [float(x) for x in args.F.split(',')]
        de = DifferentialEvolution(
            self.p,
            CR=args.C, F=F, maxiter=args.m,
            randomBase=not args.b, uniform=args.u,
            adaptive=not args.n, bitterEnd=args.e, logHandle=self.fh)
        yield de()
        yield self.shutdown()
        msg(0, "Final population:\n{}", self.p)
        msg(0, "Elapsed time: {:.2f} seconds", time.time()-t0, 0)
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

    Downloads a compressed CSV file of real VOC data points from
    edsuom.com to the current directory (if it's not already
    present). The data points and the current best-fit curves are
    plotted in the PNG file (also in the current directory)
    pfinder.png. You can see the plots, automatically updated, with
    the Linux command "qiv -Te thermistor.png". (Possibly that other
    OS may have something that works, too.)

    Press the Enter key to quit early.
    """
)
args('-m', '--maxiter', 800, "Maximum number of DE generations to run")
args('-e', '--bitter-end', "Keep working to the end even with little progress")
args('-p', '--popsize', 20, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.8, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-b', '--best', "Use DE/best/1 instead of DE/rand/1")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")
args('-l', '--logfile',
     "Write results to logfile 'covid19.log' instead of STDOUT")
args(main)
