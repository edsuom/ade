#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018 by Edwin A. Suominen,
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
Example script for the I{ade} package: thermistor.py

Reads a three-item-per-line CSV file containing temperatures (as read
by a YoctoTemp, in degrees C) inside an outdoor equipment shed, and
the voltage at inputs 1 and 2 of a YoctoVolt with thermistors
connecting to 23V. Then uses asynchronous differential evolution to
efficiently find a nonlinear best-fit curve, with digital filtering to
match thermal time constants.
"""

import os.path, bz2, time

import numpy as np
from twisted.internet import reactor, defer
from twisted.web import client

from asynqueue import ThreadQueue
from asynqueue.process import ProcessQueue
from yampex.plot import Plotter

from ade.util import *
from ade.population import Population
from ade.de import DifferentialEvolution


class Data(Picklable):
    """
    Run L{setup} on my instance to load (possibly downloading and
    decompressing first) the CSV file.
    """
    csvPath = "tempdump.csv.bz2"
    csvURL = "http://edsuom.com/ade-tempdump.csv.bz2"
    firstColumn = 1
    ranges = (0, 213), (442, 2920), (3302, 6000), (6550, 7000), (7200, None)
    
    @defer.inlineCallbacks
    def setup(self):
        """
        "Returns" a (deferred) 2-D Numpy array with the first column
        temperature readings and the following columns thermistor
        resistance readings, and a 1-D array of weights for each row
        in that array.
        """
        if not os.path.exists(self.csvPath):
            print "Downloading tempdump.csv.bz2 data file from edsuom.com...",
            yield client.downloadPage(self.csvURL, self.csvPath)
            print "Done"
        value_lists = []; T_counts = {}
        print "Decompressing and parsing tempdump.csv.bz2...",
        with bz2.BZ2File(self.csvPath, 'r') as bh:
            while True:
                line = bh.readline().strip()
                if not line:
                    if value_lists:
                        break
                    else: continue
                if line.startswith('#'):
                    continue
                value_list = [float(x.strip()) for x in line.split(',')]
                value_lists.append(value_list)
        print "Done"
        print "Doing array conversions...",
        value_lists.sort(None, lambda x: x[0])
        selected_value_lists = []
        for k, value_list in enumerate(value_lists):
            for k0, k1 in self.ranges:
                if k >= k0 and (k1 is None or k < k1):
                    T = value_list[1]
                    T_counts[T] = T_counts.get(T, 0) + 1
                    selected_value_lists.append(value_list[1:])
                    break
        print sub(
            "Read {:d} of {:d} data points", len(selected_value_lists), k+1)
        X = np.array(selected_value_lists)
        N = len(selected_value_lists)
        weights = np.zeros(N)
        for k in range(N):
            weights[k] = 1.0 / T_counts[X[k,0]]
        print "Done"
        defer.returnValue((X, weights))

    def plot(self, X):
        """
        Just plot everything in I{X}, with annotations.
        """
        T = X[:,0]
        N = len(T)
        pp = Plotter(3, 2, width=12, height=10)
        pp.add_marker('.')
        pp.add_plotKeyword('markersize', 1)
        with pp as p:
            for k in range(6):
                R = X[:,k+1]
                p.set_title(sub("Thermistor #{:d}", k+1))
                p(R, T)
        pp.show()


class Evaluator(Picklable):
    """
    Construct an instance of me, run the L{setup} method and wait for
    the C{Deferred} it returns to fire, and then call the instance a
    bunch of times with parameter values for a curve to get (deferred)
    sum-of-squared-error fitness of the curve to the thermistor data.
    """
    T_kelvin_offset = +273.15 # deg C
    prefix_bounds = {
        'a':   (1E-4, 1E-2),
        'b':   (1E-5, 1E-3),
        'c':   (1E-7, 1E-6),
        'd':   (-1E-5, 0),
        'x':   (0.75, 1.25),
        'y':   (0.75, 1.25),
        'z':   (0.75, 1.25),
    }

    def setup(self):
        """
        Returns a C{Deferred} that fires with two equal-length sequences,
        the names and bounds of all parameters to be determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.
        """
        def done(stuff):
            #data.plot(stuff[0])
            self.X, self.weights = stuff
            return names, bounds

        names = []
        bounds = []
        self.indices = {}
        prefixes = sorted(self.prefix_bounds.keys())
        for k in range(7):
            for prefix in prefixes:
                if k == 0 and prefix in 'abcd':
                    name = prefix
                elif k > 0 and prefix in 'xyz':
                    name = self.prefix2name(prefix, k)
                else: continue
                names.append(name)
                self.indices[name] = len(bounds)
                bounds.append(self.prefix_bounds[prefix])
        # The data
        data = Data()
        return data.setup().addCallbacks(done, oops)

    def prefix2name(self, prefix, k):
        return sub("{}{:d}", prefix, k)
    
    def values2params(self, values, k):
        """
        Returns a list of parameter values to use in a call to L{curve}
        for the specified Yocto-MaxThermistor input I{k} (1-6).
        """
        names = ['a', 'b', 'c', 'd']
        for prefix in ('x', 'y', 'z'):
            names.append(self.prefix2name(prefix, k))
        return [values[self.indices[x]] for x in names]
    
    def curve(self, R, a, b, c, d, x, y, z):
        """
        Given a 1-D vector of thermistor resistances followed by arguments
        defining curve parameters, returns a 1-D vector of
        temperatures (degrees C) for those resistances.
        """
        a *= x
        b *= y
        c *= z
        lnR = np.log(R)
        T_kelvin = 1.0 / (a + b*lnR + c*(lnR**3) + d*R)
        return T_kelvin - self.T_kelvin_offset

    def __call__(self, values, xSSE=None):
        SSE = 0
        T = self.X[:,0]
        for k in range(1, 7):
            R = self.X[:,k]
            params = self.values2params(values, k)
            T_curve = self.curve(R, *params)
            squaredResiduals = self.weights * np.square(T_curve - T)
            SSE += np.sum(squaredResiduals)
            if xSSE and SSE > xSSE:
                break
        return SSE

        
class Runner(object):
    """
    I run everything to fit a curve to thermistor data using
    asynchronous differential evolution. Construct an instance of me
    with an instance of L{Args} that has parsed command-line options,
    then have the Twisted reactor call the instance when it
    starts. Then start the reactor and watch the fun.
    """
    plotFilePath = "thermistor2.png"
    N_curve_plot = 200
    
    def __init__(self, args):
        self.args = args
        self.ev = Evaluator()
        N = args.N if args.N else ProcessQueue.cores()-1
        self.q = None if args.l else ProcessQueue(N, returnFailure=True)
        self.qLocal = ThreadQueue(raw=True)
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)

    @defer.inlineCallbacks
    def shutdown(self):
        if hasattr(self, 'triggerID'):
            reactor.removeSystemEventTrigger(self.triggerID)
            del self.triggerID
        if self.q is not None:
            yield self.q.shutdown()
            msg("ProcessQueue is shut down")
            self.q = None
        if self.qLocal is not None:
            yield self.qLocal.shutdown()
            msg("Local ThreadQueue is shut down")
            self.qLocal = None
    
    def plot(self, X, Y, p, xName):
        p.set_xlabel(xName)
        return p(X, Y)

    def titlePart(self, *args):
        if not args or not hasattr(self, 'titleParts'):
            self.titleParts = []
        if not args:
            return
        self.titleParts.append(sub(*args))

    def report(self, values, counter):
        def gotSSE(SSE):
            msg(0, self.p.pm.prettyValues(values, "SSE={:g} with", SSE), 0)
            T = self.ev.X[:,0]
            self.titlePart()
            self.titlePart("Temp vs Voltage")
            self.titlePart("SSE={:g}", SSE)
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
                        R, *self.ev.values2params(values, k))
                    ax.plot(R, T_curve, 'r-')
            self.pt.set_title(", ".join(self.titleParts))
            self.pt.show()
        if not hasattr(self, 'pt'):
            self.pt = Plotter(
                3, 2, filePath=self.plotFilePath, width=15, height=10)
            self.pt.set_grid(); self.pt.add_marker(',')
        return self.qLocal.call(self.ev, values).addCallbacks(gotSSE, oops)
        
    def evaluate(self, values, xSSE):
        values = list(values)
        q = self.qLocal if self.q is None else self.q
        return q.call(self.ev, values, xSSE).addErrback(oops)
    
    @defer.inlineCallbacks
    def __call__(self):
        msg(True)
        t0 = time.time()
        args = self.args
        names_bounds = yield self.ev.setup().addErrback(oops)
        self.p = Population(
            self.evaluate,
            names_bounds[0], names_bounds[1], popsize=args.p)
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
args('-m', '--maxiter', 100, "Maximum number of DE generations to run")
args('-p', '--popsize', 15, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.7, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-b', '--bitter-end', "Keep working to the end even with little progress")
args('-r', '--random-base', "Use DE/rand/1 instead of DE/best/1")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")
args('-l', '--local-queue', "Use the local ThreadQueue, no subprocesses")


def main():
    if args.h:
        return
    r = Runner(args)
    reactor.callWhenRunning(r)
    reactor.run()


if __name__ == '__main__':
    main()
