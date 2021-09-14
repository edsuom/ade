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
Example script I{voc.py}: Identifying coefficients for the
open-circuit voltage of an AGM lead-acid battery over time.

This example reads a two-item-per-line CSV file. Each line
contains: (B{1}) the time in seconds from some arbitrary starting
time, (B{2}) the battery voltage with no charge or discharge current.

Then it uses asynchronous differential evolution to efficiently find a
nonlinear best-fit curve.
"""

import time

import numpy as np
from scipy import signal

from twisted.internet import reactor, defer

from asynqueue.process import ProcessQueue
from yampex.plot import Plotter

from ade.population import Population
from ade.de import DifferentialEvolution
from ade.image import ImageViewer
from ade.util import *

from data import TimeData


class BatteryData(TimeData):
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
    basename = "voc"


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
            2, filePath=self.plotFilePath, width=15, height=10)
        self.pt.use_grid()
        self.pt.use_timex()
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
        titlePart("Voltage vs Time (sec)")
        titlePart(SSE_info)
        titlePart("k={:d}", counter)
        msg(0, self.prettyValues(values, SSE_info+", with"), 0)
        with self.pt as sp:
            sp.set_title(", ".join(titleParts))
            t = self.ev.t
            V = self.ev.X[:,0]
            # Model versus observations
            sp.add_line('-', 1)
            sp.set_ylabel("V")
            sp.set_zeroLine(values[-1])
            sp.add_annotation(0, "Battery disconnect")
            sp.add_annotation(-1, "Last observation")
            sp.add_textBox("NE", "Estimated VOC: {:.2f} V", values[-1])
            ax = sp(t, V)
            tm = np.linspace(
                t.min(), self.extrapolationMultiple*t.max(), self.N_curve_plot)
            V_curve = self.ev.curve(tm, *values)
            ax.plot(tm, V_curve, color='red', marker='o', markersize=2)
            # Residuals
            res = self.ev.curve(t, *values) - V
            sp.set_ylabel("dV")
            sp.set_zeroLine()
            k = np.argmax(np.abs(res[2:])) + 2
            resPercentage = 100 * res[k]/V[k]
            sp(t, res)
        self.pt.show()


class Evaluator(Picklable):
    """
    I evaluate battery VOC model fitness.
    
    Construct an instance of me, run the L{setup} method, and wait (in
    non-blocking Twisted-friendly fashion) for the C{Deferred} it
    returns to fire. Then call the instance a bunch of times with
    parameter values for a L{curve} to get a (deferred)
    sum-of-squared-error fitness of the curve to the thermistor data.
    """
    scale_SSE = 100
    bounds = {
        # Initial rapid drop with up to 10 minute time constant
        'a1':   (0,      40),
        'b1':   (1,      10*60),
        'c1':   (1,      200),
        # Middle drop with 20 min to 2 hour time constant
        'a2':   (0,      600),
        'b2':   (20*60,  2*3600),
        'c2':   (50,     1000),
        # Slow settling with 1-12 hour time constant
        'a3':   (0,      800),
        'b3':   (3600,   12*3600),
        'c3':   (100,    4000),
        # A bit beyond the extremes for VOC of an AGM lead acid battery
        'voc':  (45,     54),
    }

    def setup(self):
        """
        Returns a C{Deferred} that fires with two equal-length sequences,
        the names and bounds of all parameters to be determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.
        """
        def done(null):
            for name in ('t', 'X'):
                setattr(self, name, getattr(data, name))
            return names, bounds

        bounds = []
        names = sorted(self.bounds.keys())
        for name in names:
            bounds.append(self.bounds[name])
        # The data
        data = BatteryData()
        return data.setup().addCallbacks(done, oops)

    def curve(self, t, *args):
        """
        Given a 1-D time vector followed by arguments defining curve
        parameters, returns a 1-D vector of battery voltage over that
        time with with no charge or discharge current, with one
        particular but unknown SOC.

        The model implements this equation:

        M{V = a1*exp(-t/b1+c1) + ... ak*exp(-t/bk+ck) + voc}
        """
        V = args[-1]
        for k in range(3):
            a, b, c = args[3*k:3*k+3]
            V += a*np.exp(-(t+c)/b)
        return V
    
    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}.
        """
        V = self.X[:,0]
        V_curve = self.curve(self.t, *values)
        return self.scale_SSE * np.sum(np.square(V_curve - V))

        
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
        self.fh = open("voc.log", 'w') if args.l else True
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
    Parameter finder for AGM lead-acid battery open-circuit voltage
    model using Differential Evolution.

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
args('-l', '--logfile', "Write results to logfile 'voc.log' instead of STDOUT")
args(main)
