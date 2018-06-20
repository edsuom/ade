#!/usr/bin/python

"""
Reads a three-item-per-line CSV file containing the temp (as read
by YoctoTemp, in degrees C), and the voltage at inputs 1 and 2 of the
YoctoVolt with thermistors connecting to 23V. Then finds a nonlinear
best-fit curve (with digital filtering to match thermal time
constants) using differential evolution.
"""

import sys, os.path, bz2
from copy import copy

import numpy as np
from scipy import signal, interpolate, stats
from twisted.internet import reactor, defer
from twisted.web import client

from asynqueue import ThreadQueue
from asynqueue.process import ProcessQueue
from yampex.plot import Plotter

from ade.util import *
from ade.population import Population
from ade.de import DifferentialEvolution

# For providing some limited info about unhandled Deferred failures
from twisted.logger import globalLogPublisher
from twisted.logger._levels import LogLevel
def analyze(event):
    if event.get("log_level") == LogLevel.critical:
        print sub("\nERROR: {}\n", event)
        #reactor.stop()
globalLogPublisher.addObserver(analyze)


class IIR(Picklable):
    """
    First-order IIR LPF section to filter a [V1, V2] sample and make
    each tiny little thermistor's small amount of thermal capacitance
    better approximate that of the YoctoTemp sensor mounted on a small
    chunk of PCB in a ventilated housing.
    """
    # One sample every ten seconds
    ts = 10.0
    # Settling precision
    sp = 0.01

    def setup(self, tc, x0):
        self.a = 0 if tc == 0 else np.exp(-self.ts / tc)
        self.y = np.zeros(2)
        while True:
            y = self(x0)
            if np.any(x0 == 0):
                if np.all(y < self.sp):
                    break
            elif np.all(1.0 - abs(y/x0) < self.sp):
                break
    
    def __call__(self, x):
        self.y = x + self.a*self.y
        return (1.0-self.a) * self.y


class Data(Picklable):
    """
    Run L{setup} on my instance once to load (possibly downloading and
    decompressing first) the CSV file. Then call the instance as many
    times as you like with one or more thermal time constants to
    obtain a 3-column array of temp and filtered V1, V2 values.
    """
    csvPath = "tempdump.log.bz2"
    csvURL = "http://edsuom.com/ade-tempdump.log.bz2"
    
    @defer.inlineCallbacks
    def setup(self):
        if not os.path.exists(self.csvPath):
            print "Downloading tempdump.log.bz2 data file from edsuom.com...",
            yield client.downloadPage(self.csvURL, self.csvPath)
            print "Done"
        txy = []
        print "Decompressing and parsing tempdump.log.bz2...",
        with bz2.BZ2File(self.csvPath, 'r') as bh:
            while True:
                line = bh.readline().strip()
                if txy and not line:
                    break
                parts = line.split(',')[-3:]
                try:
                    this_txy = [float(x.strip()) for x in parts]
                except:
                    continue
                txy.append(this_txy)
        self.txy = np.array(txy)
        self.N = len(self.txy)
        print "Done"
            
    def __call__(self, tcs):
        """
        Builds a ladder of first-order IIR sections and filters V1, V2
        samples through them using the time constants supplied in the
        single sequence-like argument.
        """
        cascade = []
        for tc in tcs:
            section = IIR()
            section.setup(tc, self.txy[0,1:3])
            cascade.append(section)
        txy = np.copy(self.txy)
        for section in cascade:
            for k in range(self.N):
                # Very slow to have this looping per sample in Python,
                # but at least V1 and V2 are done efficiently at the
                # same time in Numpy
                txy[k,1:3] = section(txy[k,1:3])
        return txy


class Evaluator(Picklable):
    """
    """
    curveParam_names = [
        "a0",
        "a1",
        "a2",
        "v0",
    ]
    curveParam_bounds = [
        (2.0,  10.0),
        (-5.0, 10.0),
        (3.0,  20.0),
        (-5.0, 2.5),
    ]
    timeConstant_bounds = [
        (0, 30),
        (150, 400),
    ]

    def setup(self):
        """
        Returns two equal-length sequences, the names and bounds of all
        parameters to be determined.
        """
        def done(*null):
            return names, bounds
        
        # The parameters
        names, bounds = [], []
        for prefix in ("v1_", "v2_"):
            for name, bound in zip(
                    self.curveParam_names, self.curveParam_bounds):
                names.append(prefix+name)
                bounds.append(bound)
        self.kTC = len(names)
        self.N_CP = self.kTC/2
        for k, bound in enumerate(self.timeConstant_bounds):
            names.append(sub("tc{:d}", k))
            bounds.append(bound)
        # The data
        self.data = Data()
        return self.data.setup().addCallbacks(done, oops)

    def curve(self, v, a0, a1, a2, v0):
        """
        Given a 1-D vector of actual voltages followed by arguments
        defining curve parameters, returns a 1-D vector of
        temperatures (degrees C) for those voltages.
        """
        if np.any(v <= v0):
            return np.zeros_like(v)
        return a0 + a1*v + a2*np.log(v-v0)

    def curve_k(self, values, k):
        """
        """
        kCP = (k-1) * self.N_CP
        curveParams = values[kCP:kCP+self.N_CP]
        return self.curve(self.txy[:,k], *curveParams)
    
    def __call__(self, values):
        SSE = 0
        self.txy = self.data(values[self.kTC:])
        for k in (1, 2):
            t_curve = self.curve_k(values, k)
            squaredResiduals = np.square(t_curve - self.txy[:,0])
            # TODO: Remove outliers
            SSE += np.sum(squaredResiduals)
        return SSE
        
        
class Runner(object):
    """
    """
    plotFilePath = "thermistor.png"
    
    def __init__(self, args):
        self.args = args
        self.ev = Evaluator()
        N = args.N if args.N else ProcessQueue.cores()-1
        self.q = ProcessQueue(N, returnFailure=True)
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
        p.clear_annotations()
        for k in (np.argmin(X), np.argmax(X)):
            p.add_annotation(
                k, sub("({:.3g}, {:.3g})", X[k], Y[k]))
        p.set_xlabel(xName)
        p.set_ylabel("Deg C")
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
            pt = Plotter(2, 1, filePath=self.plotFilePath)
            pt.set_grid(); pt.add_marker(',')
            T = self.ev.txy[:,0]
            self.titlePart()
            self.titlePart("Temp vs Voltage")
            self.titlePart("SSE={:g}", SSE)
            with pt as p:
                for k in (1, 2):
                    V = self.ev.txy[:,k]
                    I = np.argsort(V)
                    V = V[I]
                    xName = sub("V{:d}", k)
                    ax = self.plot(V, T[I], p, xName)
                    t_curve = self.ev.curve_k(values, k)
                    ax.plot(V, t_curve[I], 'r-')
            pt.figTitle(", ".join(self.titleParts))
            pt.show()
        return self.qLocal.call(self.ev, values).addCallbacks(gotSSE, oops)
        
    def evaluate(self, values, xSSE):
        values = list(values)
        return self.q.call(self.ev, values).addErrback(oops)
        #return self.qLocal.call(self.ev.evaluate, values).addErrback(oops)
        #return defer.maybeDeferred(self.ev, values).addErrback(oops)
    
    @defer.inlineCallbacks
    def __call__(self):
        msg(True)
        args = self.args
        names_bounds = yield self.ev.setup().addErrback(oops)
        self.p = Population(
            self.evaluate, names_bounds[0], names_bounds[1], popsize=args.p)
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
        reactor.stop()


args = Args(
    """
    Thermistor Temp vs Voltage curve parameter finder using
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
args('-p', '--popsize', 4, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.6, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-b', '--bitter-end', "Keep working to the end even with little progress")
args('-r', '--random-base', "Use DE/rand/1 instead of DE/best/1")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")


def main():
    if args.h:
        return
    import pdb, traceback, sys
    r = Runner(args)
    try:
        reactor.callWhenRunning(r)
        reactor.run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == '__main__':
    main()
