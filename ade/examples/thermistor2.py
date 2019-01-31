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
Example script I{thermistor2.py}: Identifying coefficients for the
resistance versus temperature model of six thermistors.

This example reads an eight-item-per-line CSV file. Each line
contains: (B{1}) the time in seconds from some arbitrary starting
time, (B{2}) the temperature in degrees C, as read by a trusted
temperature sensor, and (B{3-8}) the resistances of six thermistors,
measured at the known temperature.

Then it uses asynchronous differential evolution to efficiently find a
nonlinear best-fit curve, with digital filtering to match thermal time
constants using the model implemented by L{Evaluator.curve}.
"""

import time

import numpy as np
from scipy import signal

from twisted.internet import reactor, defer

from asynqueue import ThreadQueue
from asynqueue.process import ProcessQueue
from yampex.plot import Plotter

from ade.util import *
from ade.population import Population
from ade.de import DifferentialEvolution

from data import Data


class TemperatureData(Data):
    """
    Run L{setup} on my instance to decompress and load the
    tempdump.csv.bz2 CSV file.

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

    @ivar weights: A 1-D array of weights for each row in that array.

    @see: The L{Data} base class.
    """
    basename = "tempdump"
    ranges = [(0, 213), (442, 2920), (3302, 6000), (6550, 7000),
              (7200, 8210), (8280, 8345), (8380, None)]

    dTdt_halfWeight = 0.05 / 10
    dTdt_power = 3
    
    def setWeights(self):
        """
        Call this to set my I{weights} attribute to a 1-D Numpy array to
        weights that are larger for colder temperature readings, since
        there are fewer of them.
        """
        T_filt = signal.lfilter([1, 1], [2], self.X[:,0])
        dTdt = np.diff(T_filt) / np.diff(self.t)
        weights = np.power(
            self.dTdt_halfWeight / (np.abs(dTdt) + self.dTdt_halfWeight),
            self.dTdt_power)
        self.weights = np.pad(weights, (1, 0), 'constant')
        # Hack to weight cold readings more, since fewer of them.
        self.weights = np.where(
            self.X[:,0] < 4.0, 10*self.weights, self.weights)
        # Even more so for really cold readings
        self.weights = np.where(
            self.X[:,0] < -5.0, 3*self.weights, self.weights)


class Evaluator(Picklable):
    """
    I evaluate thermistor curve fitness.
    
    Construct an instance of me, run the L{setup} method, and wait (in
    non-blocking Twisted-friendly fashion) for the C{Deferred} it
    returns to fire. Then call the instance a bunch of times with
    parameter values for a L{curve} to get a (deferred)
    sum-of-squared-error fitness of the curve to the thermistor data.
    """
    T_kelvin_offset = +273.15 # deg C
    driftPenalty = 0.05
    prefixes = "ABCD"
    prefix_bounds = {
        # Common to all thermistors
        'A':   (5E-4,     2E-3),
        'B':   (1.5E-4,   5E-4),
        'C':   (1E-8,     3E-8),
        'D':   (8E-9,     2.5E-8),
        # Per-thermistor relative variation
        'a':   (0.7,      1.2),
        'b':   (0.7,      1.2),
        'c':   (0.1,      10.0),
        'd':   (0.1,      10.0),
    }

    def setup(self):
        """
        Returns a C{Deferred} that fires with two equal-length sequences,
        the names and bounds of all parameters to be determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.
        """
        def done(null):
            for name in ('t', 'X', 'weights'):
                setattr(self, name, getattr(data, name))
            return names, bounds

        names = []
        bounds = []
        self.indices = {}
        prefixes = sorted(self.prefix_bounds.keys())
        for k in range(7):
            for prefix in prefixes:
                if k == 0 and prefix in self.prefixes:
                    name = prefix
                elif k > 0 and prefix in self.prefixes.lower():
                    name = self.prefix2name(prefix, k)
                else: continue
                names.append(name)
                self.indices[name] = len(bounds)
                bounds.append(self.prefix_bounds[prefix])
        # The data
        data = TemperatureData()
        return data.setup().addCallbacks(done, oops)

    def prefix2name(self, prefix, k):
        return sub("{}{:d}", prefix, k)
    
    def values2args(self, values, k):
        """
        Returns a subset list of parameter values to use as args in a call
        to L{curve} for the specified Yocto-MaxThermistor input I{k}
        (1-6).
        """
        names = [x for x in self.prefixes]
        for prefix in [x.lower() for x in names]:
            names.append(self.prefix2name(prefix, k))
        return [values[self.indices[x]] for x in names]
    
    def curve(self, R, *args):
        """
        Given a 1-D vector of resistances measured for one thermistor,
        followed by arguments defining curve parameters, returns a 1-D
        vector of temperatures (degrees C) for those resistances.

        The model implements this equation:

        M{T = 1 / (A*b + B*b*ln(R) + C*c*ln(R)^2 + D*d*ln(R)^3 - 273.15}

        where I{R} is in Ohms and I{T} is in degrees C. Uppercase
        coefficients are for all thermistors and lowercase are for
        just the thermistor in question.
        """
        lnR = np.log(R)
        a, b, c, d = [args[k]*args[k+4] for k in range(4)]
        T_kelvin = 1.0 / (a + b*lnR + c*(lnR**2) + d*(lnR**3))
        return T_kelvin - self.T_kelvin_offset

    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}.

        Applies a penalty if the geometric mean of the scaling factors
        (values after the first six, lowercase param names) deviates
        from 1.0, to counteract genetic drift.
        """
        SSE = 0
        T = self.X[:,0]
        for k in range(1, 7):
            R = self.X[:,k]
            args = self.values2args(values, k)
            T_curve = self.curve(R, *args)
            squaredResiduals = self.weights * np.square(T_curve - T)
            SSE += np.sum(squaredResiduals)
        # Apply drift penalty
        values = np.array(values)
        for prefix in self.prefixes:
            # Add a penalty for drift of each parameter separately
            K = [self.indices[x]
                 for x in self.indices if x.startswith(prefix.lower())]
            ratios = values[K]
            drift = np.sum(np.log(ratios))
            SSE *= 1 + self.driftPenalty * drift**2
        # Done
        return SSE

        
class Runner(object):
    """
    I run everything to fit a curve to thermistor data using
    asynchronous differential evolution.

    Construct an instance of me with an instance of L{Args} that has
    parsed command-line options, then have the Twisted reactor call
    the instance when it starts. Then start the reactor and watch the
    fun.
    """
    plotFilePath = "thermistor2.png"
    N_curve_plot = 200
    # Set much lower because real improvements are made even with low
    # improvement scores. I think that behavior has something to do
    # with all the independent parameters for six thermistors.
    targetFraction = 0.005
    
    def __init__(self, args):
        """
        C{Runner(args)}
        """
        self.args = args
        self.ev = Evaluator()
        N = args.N if args.N else ProcessQueue.cores()-1
        self.q = ThreadQueue(
            raw=True) if args.l else ProcessQueue(N, returnFailure=True)

    @defer.inlineCallbacks
    def shutdown(self):
        msg("Shutting down...")
        if self.q is not None:
            yield self.q.shutdown()
            msg("Task Queue is shut down")
            self.q = None
        msg("Goodbye")
    
    def plot(self, X, Y, p, xName):
        p.set_xlabel(xName)
        return p(X, Y)

    def titlePart(self, *args):
        if not args or not hasattr(self, 'titleParts'):
            self.titleParts = []
        if not args:
            return
        self.titleParts.append(sub(*args))

    def report(self, values, counter, SSE):
        if not hasattr(self, 'pt'):
            self.pt = Plotter(
                3, 2, filePath=self.plotFilePath, width=15, height=10)
            self.pt.set_grid(); self.pt.add_marker(',')
        SSE_info = sub("SSE={:g}", SSE)
        msg(0, self.p.pm.prettyValues(values, SSE_info+", with"), 0)
        T = self.ev.X[:,0]
        self.titlePart()
        self.titlePart("Temp vs Voltage")
        self.titlePart(SSE_info)
        self.titlePart("k={:d}", counter)
        with self.pt as p:
            for k in range(1, 7):
                xName = sub("R{:d}", k)
                R = self.ev.X[:,k]
                # Scatter plot of temp and resistance readings
                ax = self.plot(R, T, p, xName)
                # Plot current best-fit curve, with a bit of extrapolation
                R = np.linspace(R.min()-10, R.max()+10, self.N_curve_plot)
                T_curve = self.ev.curve(
                    R, *self.ev.values2args(values, k))
                ax.plot(R, T_curve, 'r-')
        self.pt.set_title(", ".join(self.titleParts))
        self.pt.show()
        
    def evaluate(self, values):
        values = list(values)
        if self.q: return self.q.call(self.ev, values)
    
    @defer.inlineCallbacks
    def __call__(self):
        msg(True)
        t0 = time.time()
        args = self.args
        names_bounds = yield self.ev.setup().addErrback(oops)
        self.p = Population(
            self.evaluate,
            names_bounds[0], names_bounds[1],
            popsize=args.p, targetFraction=self.targetFraction)
        yield self.p.setup().addErrback(oops)
        self.p.addCallback(self.report)
        F = [float(x) for x in args.F.split(',')]
        de = DifferentialEvolution(
            self.p,
            CR=args.C, F=F, maxiter=args.m,
            randomBase=args.r, uniform=args.u, adaptive=not args.n,
            bitterEnd=args.b
        )
        yield de()
        yield self.shutdown()
        msg(0, "Final population:\n{}", self.p)
        msg(0, "Elapsed time: {:.2f} seconds", time.time()-t0, 0)
        reactor.stop()


args = Args(
    """
    Thermistor Temp vs resistance parameter finder using
    Differential Evolution.

    Downloads a compressed CSV file of real thermistor data points
    from edsuom.com to the current directory (if it's not already
    present). The data points and the current best-fit curves are
    plotted in the PNG file (also in the current directory)
    pfinder.png. You can see the plots, automatically updated, with
    the Linux command "qiv -Te thermistor.png". (Possibly that other
    OS may have something that works, too.)
    """
)
args('-m', '--maxiter', 2000, "Maximum number of DE generations to run")
args('-p', '--popsize', 10, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.6, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-b', '--bitter-end', "Keep working to the end even with little progress")
args('-r', '--random-base', "Use DE/rand/1 instead of DE/best/1")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")
args('-l', '--local-queue', "Use the local ThreadQueue, no subprocesses")


def main():
    """
    Called when this module is run as a script.
    """
    if args.h:
        return
    r = Runner(args)
    reactor.callWhenRunning(r)
    reactor.run()


if __name__ == '__main__':
    main()
