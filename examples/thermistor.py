#!/usr/bin/python

"""
Reads a three-item-per-line CSV file containing the temp (as read
by YoctoTemp, in degrees C), and the voltage at inputs 1 and 2 of the
YoctoVolt with thermistors connecting to 23V. Then finds a nonlinear
best-fit curve (with digital filtering to match thermal time
constants) using differential evolution.
"""

import sys, os.path

import numpy as np
from scipy import signal, interpolate, stats

from asynqueue.process import ProcessQueue

from ade.util import sub, Args


class IIR(object):
    """
    First-order IIR LPF section to filter a V1 or V2 sample and make
    the tiny little thermistor's small amount of thermal capacitance
    better approximate that of the YoctoTemp sensor mounted on a small
    chunk of PCB in a ventilated housing.
    """
    # One sample every ten seconds
    ts = 10.0
    # Settling precision
    sp = 0.01
    
    def __init__(self, tc):
        self.a = 0 if tc == 0 else np.exp(-self.ts / tc)
        self.y = 0

    def settle(self, x):
        self.y = 0
        while True:
            y = self(x)
            if x == 0:
                if y < self.sp:
                    break
            elif 1.0 - abs(y/x) < self.sp:
                break
    
    def __call__(self, x):
        self.y = x + self.a*self.y
        return (1.0-self.a) * self.y


class Data(object):
    """
    Reads the CSV file the first time called.
    """
    def __init__(self, fh):
        self.txy = None
        txy = []
        for t, x, y in self.txyerator(fh):
            txy.append([t, x, y])
        self.txy = np.array(txy)
        self.N = len(self.txy)

    def txyerator(self, fh):
        for line in fh:
            parts = line.split(',')[-3:]
            try:
                txy = [float(x.strip()) for x in parts]
            except:
                continue
            yield txy
            
    def __call__(self, *tcs):
        """
        Builds a ladder of first-order IIR sections and filters V1, V2
        samples through them.
        """
        if not tcs:
            return self.txy
        cascades = []
        for k in (1, 2):
            cascade = []
            for tc in tcs:
                section = IIR(tc)
                section.settle(self.txy[0, k])
                cascade.append(section)
            cascades.append(cascade)
        txy = np.copy(self.txy)
        for k, cascade in enumerate(cascades):
            for section in cascade:
                for kk in range(self.N):
                    # Very slow to have this looping per sample in Python
                    txy[kk, k+1] = section(txy[kk, k+1])
        return txy


class CurveFitter(object):
    p0 = [0.45, 0.05, 0.01, 25.0]
    #bounds = (
    #    [0.1, 0.01, 0.002, 15.0],
    #    [0.7, 0.10, 0.050, 40.0],
    #)
    
    def func(self, x, *p):
        return p[0] + p[1]*x + p[2]*x*np.exp((x-p[3])/15)

    def sigma(self, X):
        from scipy.signal import hann
        N = len(X)
        return 0.1 + 0.5*hann(N) + 0.1*np.power(np.linspace(1.0, 0.0, N), 3)
    
    def __call__(self, X, Y, ax):
        popt, pcov = curve_fit(
            self.func, X, Y,
            p0=self.p0, #bounds=self.bounds,
            sigma=self.sigma(X), absolute_sigma=False)
        perr = np.sqrt(np.diag(pcov))
        Yc = self.func(X, *popt)
        print sub("\nParameters: [{}]", ", ".join([
            sub("{:f}", x) for x in popt]))
        print sub("Sum of squared residuals: {:f}\n{}", np.sum(
            np.square(Y-Yc)), "-"*79)
        print popt
        print perr
        ax.plot(X, Yc, 'r-')

        
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
