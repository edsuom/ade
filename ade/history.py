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
A L{History} class for maintaining a history of L{Individual}
objects.
"""

from cStringIO import StringIO

import numpy as np

from yampex.plot import Plotter

from util import *


class Analysis(object):
    """
    I let you analyze the parameter values of a L{Population}.
    
    Construct an instance of me with a sequence of parameter I{names},
    a 2-D Numpy array I{X} of values (in columns) for each of those
    parameters (in rows), a sequence I{K} of row indices, and a
    sequence I{SSEs} of SSEs, one corresponding to each row index in
    I{K}.

    The values of I{SSEs} must be sorted in ascending order. Each
    index in I{K} points to the row of I{X} with the parameter values
    for that SSE.
    """
    fmms = (
        (0.05, 'o', 3.0),
        (0.10, 'o', 2.0),
        (0.20, '.', 1.5),
        (0.50, '.', 1.0),
        (0.70, '.', 0.5),
        (1.00, '.', 0.0),
    )
    
    def __init__(self, names, X, K, SSEs, Kp):
        self.names = names
        self.X = X
        self.K = np.array(K)
        self.SSEs = np.array(SSEs)
        self.Kp = np.array(Kp)

    def corr(self, k1, k2):
        """
        Returns the Pearson correlation coefficient between parameter
        values of column I{k1} and column I{k2} in my I{X} array.
        """
        kk = [k1, k2]
        return np.corrcoef(np.transpose(self.X[self.K][:,kk]))[0,1]

    def correlator(self):
        """
        Iterates over combinations of parameters, from most correlated to
        least. Each iteration yields column indices of the parameter
        pair and their correlation coefficient.
        """
        kkR = []
        combos = []
        Nc = self.X.shape[1]
        for k1 in range(Nc):
            for k2 in range(Nc):
                if k2 == k1: continue
                combo = {k1, k2}
                if combo in combos: continue
                combos.append(combo)
                kkR.append([k1, k2, self.corr(k1, k2)])
        kkR.sort(key=lambda row: abs(row[2]), reverse=True)
        for k1, k2, R in kkR:
            yield k1, k2, R

    def Kf12(self, f1, f2):
        """
        Returns a 1-D Numpy array of row indices to my I{X} array whose
        SSEs are from fractional value I{f1} to I{f2} between minimum
        and maximum SSE.
        """
        def fSSE(f):
            minSSE = self.SSEs[0]
            maxSSE = self.SSEs[-1]
            mmSSE = maxSSE - minSSE
            return minSSE + f*mmSSE
        I = np.logical_and(self.SSEs >= fSSE(f1), self.SSEs < fSSE(f2))
        return self.K[np.flatnonzero(I)]

    def Kp12(self, p1, p2):
        """
        Returns a 1-D Numpy array of row indices to my I{X} array whose
        SSEs are from fractional portion I{p1} to I{p2} between
        minimum and maximum SSE.
        """
        N = len(self.SSEs)
        return self.K[slice(int(np.floor(p1*N)), int(np.ceil(p2*N)))]
    
    def args2names(self, args):
        """
        Converts args to a list of parameter names:

            - With no args, returns my full list of parameter I{names}.

            - With one or more strings, returns a list of the matching
              names.

            - With integer arguments, creates a slice and returns that
              slice of the entries of my parameter I{names} list.
        """
        if not args:
            return sorted(self.names)
        for arg in args:
            if not isinstance(arg, int): return args
        return sorted(self.names)[slice(*args)]

    def value_vs_SSE(self, *names, **kw):
        """
        Returns a 1-D Numpy array of the SSEs of my individuals and
        matching 1-D Numpy arrays for each of the named parameter
        values.

        @keyword maxRatio: Set this to specify a maximum ratio between
            an included individual's SSE and the best individual's
            SSE.
        """
        names = self.args2names(names)
        maxRatio = kw.get('maxRatio', 1000)
        SSE_best = self.SSEs[0]
        K = np.flatnonzero(self.SSEs/SSE_best <= maxRatio)
        result = [self.SSEs[K]]
        K = self.K[K]
        for name in names:
            k = self.names.index(name)
            result.append(self.X[K,k])
        return result

    def plot(self, *names, **kw):
        """
        Plots the values versus SSE for each of the named
        parameters. Accepts keywords used for L{value_vs_SSE}.

        If there are two integer args, they are used to select a range
        of my I{names}.
        """
        names = self.args2names(names)
        XY = self.value_vs_SSE(*names, **kw)
        N = len(XY) - 1
        kList = range(N)
        while kList:
            N = min([9, len(kList)])
            kkList = kList[:N]; kList = kList[N:]
            Nc = 1 if N == 1 else 3 if N > 6 else 2
            pt = Plotter(N, Nc=Nc)
            pt.add_marker('.', 2.0); pt.add_line(""); pt.use_grid()
            with pt as sp:
                for k in kkList:
                    name = names[k]
                    sp.set_title(name)
                    sp(XY[0], XY[k+1])
        pt.showAll()

    def plotXY(self, arg1, arg2, sp=None, useFraction=False):
        """
        Plots the values of the parameter at column I{k2} of my I{X} array
        versus the values of the parameter at column I{k1}, with a
        rough indication of the SSEs involved.
        """
        def plot(sp):
            sp.set_xlabel(self.names[k1])
            sp.set_ylabel(self.names[k2])
            ax = sp()
            f1 = 0.0
            kw = {'color': "blue"}
            for f2, m, ms in self.fmms:
                if ms:
                    K = self.Kf12(f1, f2) if useFraction else self.Kp12(f1, f2)
                    kw['marker'] = m
                    kw['markersize'] = ms
                    ax.plot(self.X[K,k1], self.X[K,k2], **kw)
                f1 = f2

        k1 = arg1 if isinstance(arg1, int) else self.names.index(arg1)
        k2 = arg2 if isinstance(arg2, int) else self.names.index(arg2)
        if sp is None:
            pt = Plotter(1)
            pt.add_line(""); pt.use_grid()
            with pt as sp:
                plot(sp)
            pt.show()
            return
        plot(sp)

    def plotCorrelated(self, N=4):
        """
        Plots values of four pairs of parameters with the highest
        correlation. The higher the SSE for a given combination of
        values, the less prominent the point will be in the plot.

        This actually has been of surprisingly little use in my own
        work, which is probably a good sign that my parameters have
        all been pretty independent and thus independently
        worthwhile. Seeing a very high correlation in one of these
        plots is an indication that you should somehow consolidate the
        correlated parameters or at least make them explicitly
        dependent on each other at the outset, so DE doesn't waste
        effort searching all the deserted fitness landscape outside
        the narrow ellipse of their correlated values.
        """
        pt = Plotter(N)
        pt.add_line(""); pt.use_grid()
        with pt as sp:
            for k, stuff in enumerate(self.correlator()):
                if k == N: break
                k1, k2, R = stuff
                sp.add_textBox("SE" if R > 0 else "NE", "R={:+.3f}", R)
                self.plotXY(k1, k2, sp)
        pt.show()
        

class History(object):
    """
    I maintain a roster of the parameter values and SSEs of
    I{Individual} objects that a L{Population} has had and possibly
    replaced.

    @keyword N_max: The most records I can have in my roster. When the
        roster is full, adding a non-duplicative I{Individual} will
        bump the highest-SSE one currently in the roster to make room.
    
    @ivar names: A sequence of my individuals' parameter names,
        supplied as the sole constructor argument.
    
    @ivar X: A 2-D Numpy array of parameter values, each row being the
        parameter values for one individual.

    @ivar K: A list of indices to rows of I{X}, each entry in the list
        corresponding to an entry of I{SSEs}.

    @ivar SSEs: A list of SSEs, one for each row of parameter values
        in I{X}, as referenced by the indices in I{K}.

    @ivar Kp: A list of indices to I{K} (and I{SSEs}) for individuals
        currently in the L{Population} that I serve.

    @cvar minFracDiff: The minimum fractional distance for an
        I{Individual} to be considered non-duplicative of the
        I{Individual} with the closest SSE. The figure is small, but
        it actually represents a pretty high bar for disregarding an
        addition to the history, considering that everything (SSE and
        B{all} parameter values) has to be that close to the nearest
        neighbor to not be added.
    """
    N_max = 2500
    minFracDiff = 0.03
    
    def __init__(self, names, N_max=None):
        self.names = names
        if N_max: self.N_max = N_max
        self.X = np.zeros((self.N_max, len(names)), dtype='f4')
        self.K = []; self.SSEs = []; self.Kp = []

    @staticmethod
    def seq2str(X, dtype=None):
        """
        Converts the supplied sequence I{X} to a string, returned. If the
        sequence is not already a Numpy array, supply an efficient
        I{dtype} for the array version that will be created for it.
        """
        if dtype: X = np.array(X, dtype=dtype)
        fh = StringIO()
        np.save(fh, X)
        X = fh.getvalue()
        fh.close()
        return X

    @staticmethod
    def str2array(state, name):
        """
        Converts the string with key I{name} in I{state} into a Numpy
        array, which gets returned.
        """
        if name not in state: return np.array([])
        text = state[name]
        fh = StringIO(text)
        X = np.load(fh)
        fh.close()
        return X
        
    def __getstate__(self):
        """
        For storage-efficient pickling.
        """
        return {
            'names': self.names,
            'N_max': self.N_max,
            'X': self.seq2str(self.X),
            'K': self.seq2str(self.K, 'u2'),
            'SSEs': self.seq2str(self.SSEs, 'f4'),
            'Kp': self.seq2str(self.Kp, 'u2'),
        }
    
    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.names = state['names']
        self.N_max = state['N_max']
        self.X = self.str2array(state, 'X')
        self.K = list(self.str2array(state, 'K'))
        self.SSEs = list(self.str2array(state, 'SSEs'))
        self.Kp = list(self.str2array(state, 'Kp'))
    
    @property
    def a(self):
        if not hasattr(self, '_analysis'):
            self._analysis = Analysis(
                self.names, self.X, self.K, self.SSEs, self.Kp)
        return self._analysis
            
    def __len__(self):
        """
        My length is the number of records in my roster.
        """
        return len(self.SSEs)

    def __getitem__(self, k):
        """
        Access the parameter values corresponding to index I{k} of my
        I{SSEs} list.
        """
        k = self.K[k]
        return self.X[k,:]

    def __iter__(self):
        """
        I iterate over 1-D Numpy arrays of parameter values in ascending
        order of the SSEs they resulted in.
        """
        for k in self.K:
            yield self.X[k,:]
    
    def clear(self):
        del self.K[:]
        del self.SSEs[:]

    def isDuplicative(self, k, SSE, values):
        """
        Returns C{True} if the I{SSE} and I{values} being proposed for
        insertion into to my roster are so close to those of an
        adjacent entry as to not be meaningful.
        """
        def diffEnough(a, b, bMin, bMax):
            minDiff = self.minFracDiff * (bMax-bMin)
            return abs(a-b) > minDiff

        def valuesDifferent(k):
            for kk, value in enumerate(values):
                X = self.X[self.K,kk]
                minMax = X.min(), X.max()
                if diffEnough(value, X[k], *minMax):
                    # This value was different enough from the
                    # neighbor's value
                    return True
        
        if k == 0:
            kList = [0]
        else:
            kList = [k-1]
            if k < len(self): kList.append(k)
        # Check if SSE is different enough from neighbors
        minMax = self.SSEs[0], self.SSEs[-1]
        for k in kList:
            if not diffEnough(SSE, self.SSEs[k], *minMax):
                # The SSE wasn't different enough from this neighbor's
                # to disregard values
                if not valuesDifferent(k):
                    # SSE and values not substantially different
                    return True
        
    def add(self, i):
        """
        Adds the SSE and parameter values of the supplied individual I{i}
        to my roster, possibly bumping the worst one already there to
        make room.
        """
        SSE = i.SSE
        values = np.array(i.values)
        N = len(self)
        if N == 0:
            # First item, addition is simple and guaranteed
            self.K.append(0)
            self.SSEs.append(SSE)
            self.Kp.append(0)
            self.X[0,:] = values
            return
        if N >= self.N_max:
            # Roster is full...
            if SSE > self.SSEs[-1]:
                # ...and this is worse than any there, so no addition
                return
            # ...so this is the row index of what values might get
            # replaced
            kr = self.K[-1]
            trim = True
        else:
            # Roster isn't yet full, so we will (probably) just add
            # values at the next row index
            kr = N
            trim = False
        # The new SSE entry would go here
        k = np.searchsorted(self.SSEs, SSE)
        if k == 0 and trim:
            # It's the best, so it will definitely be added. But the
            # roster is full, so if the current best one is
            # duplicative of it, that will get discarded.
            if self.isDuplicative(0, SSE, values):
                self.SSEs.pop(0)
                kk = self.K.pop(0)
                if kk in self.Kp: self.Kp.remove(kk)
                # That takes care of the trimming
                trim = False
        elif k and self.isDuplicative(k, SSE, values):
            # It's not the best and is duplicative, so it won't be
            # added
            return
        self.K.insert(k, kr)
        self.SSEs.insert(k, SSE)
        self.Kp.append(k)
        self.X[kr,:] = values
        # Trim off any excess
        if trim:
            self.SSEs.pop()
            kk = self.K.pop()
            if kk in self.Kp: self.Kp.remove(kk)

    def notInPop(self, i):
        """
        Call this with a reference to an I{Individual} when it has been
        removed from the L{Population} I serve.
        """
        iSSE = i.SSE
        K = [k for k, SSE in enumerate(self.SSEs) if SSE == iSSE]
        N = len(K)
        if not N: return
        if N == 1: k = K[0]
        else:
            iValues = np.array(i.values)
            for k in K:
                if np.all(np.equal(values, iValues)):
                    break
            else: return
        if k in self.Kp: self.Kp.remove(k)


    
                
        
