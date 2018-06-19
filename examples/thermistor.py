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
from twisted.web import client

from asynqueue.process import ProcessQueue

from ade.util import sub, Args


class IIR(object):
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


class Data(object):
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
            yield client.downloadPage(self.csvURL, self.csvPath)
        txy = []
        with bz2.BZ2File(self.csvPath, 'r') as bh:
            while True:
                line = bh.readline().strip()
                if not line: break
                parts = line.split(',')[-3:]
                try:
                    this_txy = [float(x.strip()) for x in parts]
                except:
                    continue
                txy.append(this_txy)
        self.txy = np.array(txy)
        self.N = len(self.txy)
            
    def __call__(self, tcs):
        """
        Builds a ladder of first-order IIR sections and filters V1, V2
        samples through them using the time constants supplied in the
        single sequence-like argument.
        """
        if not tcs:
            return self.txy
        cascade = []
        for tc in tcs:
            section = IIR(tc)
            section.setup(self.txy[0,1:3])
            cascade.append(section)
        txy = np.copy(self.txy)
        for section in enumerate(cascade):
            for k in range(self.N):
                # Very slow to have this looping per sample in Python,
                # but at least V1 and V2 are done efficiently at the
                # same time in Numpy
                txy[k,1:3] = section(txy[k,1:3])
        return txy


class Evaluator(object):
    """
    """
    curveParam_names = [
        "a0",
        "a1",
        "a2",
    ]
    curveParam_bounds = [
        (-40.0, 5.0),
        (-20.0, 20.0),
        (3.0,   20.0),
    ]
    timeConstant_bounds = [
        (0, 80),
        (90, 240),
    ]

    def setup(self):
        """
        Returns two equal-length sequences, the names and bounds of all
        parameters to be determined.
        """
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
        return self.data.setup().addCallback(lambda _: names, bounds)

    def curve(self, v, a0, a1, ae, vc, v0):
        """
        Given a 1-D vector of actual voltages followed by arguments
        defining curve parameters, returns a 1-D vector of
        temperatures (degrees C) for those voltages.
        """
        return a0 + a1*v + a2*np.log(v)
    
    def __call__(self, *values):
        SSE = 0
        txy = self.data(values[self.kTC:])
        for k in (0, 1):
            kCP = k * self.N_CP
            curveParams = values[kCP:kCP+self.N_CP]
            t = self.curve(txy[:,k+1], *curveParams)
            squaredResiduals = np.square(t - txy[:,0])
            # TODO: Remove outliers
            SSE += np.sum(squaredResiduals)
        return SSE
        
        
class Runner(object):
    p = None
    X = None
    smoothingScale = 1E-5
    csvFile = "tempdump.log"

    class FakePlotter(object):
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def show(self, null):
            pass
    
    def __init__(self, args):
        indices = self.str2list(args.k) if args.k else [0]
        if len(indices) == 1:
            kI = [int(indices[0]), None]
        else: kI = [int(x) for x in indices]
        indices = self.str2list(args.K) if args.K else None
        kE = [] if indices is None else [int(x) for x in indices]
        self.r = Reader(kI, kE, args.M, args.f)
        with open(self.csvFile) as fh:
            self.r.load(fh)
        self.r.makeVectors()
        if args.p:
            self.lines = args.l
            self.plotSetup(args.t)
        else:
            self.p = self.FakePlotter()

    def str2list(self, text, what=str, sep=','):
        result = []
        parts = text.strip().split(sep)
        if len(parts) > 1 and parts[0] == '':
            result.append(what("0"))
        for part in parts:
            part = part.strip()
            if part:
                result.append(what(part))
        return result
        
    def plotSetup(self, timeSeries=False):
        # Define subplot structure
        self.p = Plotter(1, 1) if timeSeries else Plotter(2, 1)
        self.p.set_grid()
        if not timeSeries and not self.lines:
            self.p.add_marker('.')
    
    def plot(self, X, Y, p, r, xName, yName):
        if not p: return
        p.clear_annotations()
        for k in (np.argmin(X), np.argmax(X)):
            p.add_annotation(
                k, sub("({:.3g}, {:.3g})", X[k], Y[k]))
        p.set_xlabel(xName)
        p.set_ylabel(yName)
        return p(X, Y)

    def titlePart(self, proto, *args):
        if not hasattr(self, 'titleParts'):
            self.titleParts = []
        self.titleParts.append(sub(proto, *args))
    
    def __call__(self):
        cf = CurveFitter()
        r = self.r
        if not self.lines:
            I = np.argsort(r.T)
            T = r.T[I];
        else: T = r.T;
        self.titlePart("Voltage vs Temp")
        self.titlePart(
            "{:d} decimated samples from {:d} total", len(T), len(r))
        self.titlePart("M={:d}", r.M)
        self.titlePart("Tc={:.1f} sec", r.Tc)
        with self.p as p:
            for k in (0, 1):
                V = getattr(r, 'X' if k == 0 else 'Y')
                if not self.lines:
                    V = V[I]
                yName = sub("V{:d}", k+1)
                ax = self.plot(
                    T, V, p, r, "Temp (Deg C)", yName)
                if not self.lines:
                    cf(T, V, ax)
        self.p.figTitle(", ".join(self.titleParts))
        self.p.show()
        np.save("tempdump.npy", np.column_stack((r.T, r.X, r.Y)))


args = Args("Temp Dump File Reader")
args('-k', '--indices', "", "Index of first[,last] sample(s) to acquire")
args('-K', '--omit-indices', "", "Index of first[,last] sample(s) to omit")
args('-M', '--M', 4, "Decimation rate, CSV entries to filtered data points")
args('-p', '--plot', "Plot the vectors")
args('-l', '--lines', "Connect unsorted datapoints with lines (implies -p)")
args('-t', '--time', "Time series plot instead of XY")
args('-f', '--filter', 180.0, "Time constant for 1st-order IIR LPF on V1, V2")


def main():
    if args.h:
        return
    import pdb, traceback, sys
    try:
        Runner(args)()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == '__main__':
    main()
