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

import numpy as np

from yampex.plot import Plotter

from util import *


class Analysis(object):
    """
    
    """
    fmms = (
        (0.05, 'o', 3.0),
        (0.10, 'o', 2.0),
        (0.20, '.', 1.5),
        (0.50, '.', 1.0),
        (0.70, '.', 0.5),
        (1.00, '.', 0.0),
    )
    
    def __init__(self, names, X, K, SSEs):
        self.names = names
        self.X = X
        self.K = np.array(K)
        self.SSEs = np.array(SSEs)

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

    def plotXY(self, k1, k2, sp=None, useFraction=False):
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
        
        if sp is None:
            pt = Plotter(1)
            pt.add_line(""); pt.use_grid()
            with pt as sp:
                plot(sp)
            pt.show()
            return
        plot(sp)

    def plotCorrelated(self, N=4):
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

    @keyword N_max: The most records I can have in my roster. If more
        are added when it's full, the highest-SSE one is bumped to
        make room.
    
    @ivar names: A sequence of my individuals' parameter names,
        supplied as the sole constructor argument.
    
    @ivar X: A 2-D Numpy array of parameter values, each row being the
        parameter values for one individual.

    @ivar K: A list of indices to rows of I{X}, each entry in the list
        corresponding to an entry of I{SSEs}.

    @ivar SSEs: A list of SSEs, one for each row of parameter values
        in I{X}, as referenced by the indices in I{K}.
    """
    N_max = 2000
    minFracDiff = 0.005
    
    def __init__(self, names, N_max=None):
        self.names = names
        if N_max: self.N_max = N_max
        self.X = np.zeros((self.N_max, len(names)), dtype='f4')
        self.K = []; self.SSEs = []

    def __getstate__(self):
        return {
            'names': self.names,
            'N_max': self.N_max,
            'X': self.X,
            'K': self.K,
            'SSEs': self.SSEs
        }
        
    def __setstate__(self, state):
        for name in state:
            setattr(self, name, state[name])

    @property
    def a(self):
        if not hasattr(self, '_analysis'):
            self._analysis = Analysis(self.names, self.X, self.K, self.SSEs)
        return self._analysis
            
    def __len__(self):
        """
        My length is the number of records in my roster.
        """
        return len(self.SSEs)

    def __getitem__(self, k):
        k = self.K[k]
        return self.X[k,:]

    def __iter__(self):
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
        
        kList = [k-1]
        if k < len(self)-1: kList.append(k+1)
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
            self.SSEs.append(SSE)
            self.K.append(0)
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
        if k and self.isDuplicative(k, SSE, values):
            # Except that it's not the best and is duplicative, so it
            # doesn't
            return
        self.SSEs.insert(k, SSE)
        self.K.insert(k, kr)
        self.X[kr,:] = values
        # Trim off any excess
        if trim:
            self.SSEs.pop()
            self.K.pop()


