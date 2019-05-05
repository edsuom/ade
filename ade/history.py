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
    a 2-D Numpy array I{X} of values (in columns) for the SSEs (first
    column) and then each of those parameters (remaining columns), and
    a sequence I{K} of row indices.

    Each index in I{K} points to a row of I{X} with one SSE and the
    parameter values for that SSE, with the indices of I{K} sorted in
    ascending order of the SSE they point to.
    """
    fmms = (
        (0.05, 'o', 3.0),
        (0.10, 'o', 2.0),
        (0.20, '.', 1.5),
        (0.50, '.', 1.0),
        (0.70, '.', 0.5),
        (1.01, '.', 0.0),
    )
    
    def __init__(self, names, X, K, Kp=set()):
        self.names = names
        self.X = X
        self.K = np.array(K)
        self.Kp = Kp

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
        for k1 in range(1, Nc):
            for k2 in range(1, Nc):
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
            mmSSE = SS[-1] - SS[0]
            return SS[0] + f*mmSSE

        # 1-D array of SSEs, sorted
        SS = self.X[self.K,0]
        I = np.logical_and(SS >= fSSE(f1), SS < fSSE(f2))
        return self.K[np.flatnonzero(I)]

    def Kp12(self, p1, p2):
        """
        Returns a 1-D Numpy array of row indices to my I{X} array whose
        SSEs are from fractional portion I{p1} to I{p2} between
        minimum and maximum SSE.

        The fractional portion is how far along the indices you are,
        not how far along the values you are. If the SSEs increased
        linearly, they would be the same.
        """
        N = len(self.K)
        return self.K[slice(int(np.floor(p1*N)), int(np.floor(p2*N))+1)]
    
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

    def name2k(self, name):
        """
        Returns the column index in my I{X} array for the values of
        parameter I{name}.
        """
        return self.names.index(name) + 1
    
    def value_vs_SSE(self, *names, **kw):
        """
        Returns a 1-D Numpy array of the SSEs of my individuals and
        matching 1-D Numpy arrays for each of the named parameter
        values.

        @keyword inPop: Set C{True} to only include individuals in the
            population.
        
        @keyword notInPop: Set C{True} to exclude individuals in the
            population.

        @keyword maxRatio: Set this to specify a maximum ratio between
            an included individual's SSE and the best individual's
            SSE.
        """
        names = self.args2names(names)
        maxRatio = kw.get('maxRatio', 1000)
        SSE_best = self.X[self.K[0],0]
        if kw.get('inPop', False):
            K = [k for k in self.K if k in self.Kp]
        elif kw.get('notInPop', False):
            K = [k for k in self.K if k not in self.Kp]
        else: K = self.K
        KK = np.flatnonzero(self.X[K,0]/SSE_best <= maxRatio)
        K = np.array(K)[KK]
        result = [self.X[K,0]]
        for name in names:
            result.append(self.X[K,self.name2k(name)])
        return result

    def plot(self, *names, **kw):
        """
        Plots the values versus SSE for each of the named
        parameters. Accepts keywords used for L{value_vs_SSE}.

        If there are two integer args, they are used to select a range
        of my I{names}.
        """
        names = self.args2names(names)
        kw['inPop'] = True
        XYp = self.value_vs_SSE(*names, **kw)
        kw['inPop'] = False
        kw['notInPop'] = True
        XYn = self.value_vs_SSE(*names, **kw)
        N = len(XYp) - 1
        kList = range(N)
        while kList:
            N = min([9, len(kList)])
            kkList = kList[:N]; kList = kList[N:]
            Nc = 1 if N == 1 else 3 if N > 6 else 2
            pt = Plotter(N, Nc=Nc)
            pt.add_marker('.', 2.5); pt.add_color('red')
            pt.add_marker('.', 2.0); pt.add_color('blue')
            pt.add_line(""); pt.use_grid()
            with pt as sp:
                for k in kkList:
                    name = names[k]
                    sp.set_title(name)
                    ax = sp(XYp[0], XYp[k+1])
                    ax.plot(Xyn[0], XYn[k+1])
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

        k1 = arg1 if isinstance(arg1, int) else self.name2k(arg1)
        k2 = arg2 if isinstance(arg2, int) else self.name2k(arg2)
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
    
    @ivar X: A 2-D Numpy array of SSEs (first column) and parameter
        values (remaining columns) of one individual.

    @ivar K: A list of indices to rows of I{X}, each entry in the list
        corresponding to a row of I{X}.

    @ivar Kp: A set of the values (not indices) of I{K} that are for
        individuals currently in the population.

    """
    N_max = 2000
    
    def __init__(self, names, N_max=None):
        self.names = names
        if N_max: self.N_max = N_max
        self.X = np.zeros((self.N_max, len(names)+1), dtype='f4')
        self.K = []
        self.Kp = set()

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
            'Kp': self.seq2str(list(self.Kp), 'u2'),
        }
    
    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.names = state['names']
        self.N_max = state['N_max']
        self.X = self.str2array(state, 'X')
        self.K = list(self.str2array(state, 'K'))
        self.Kp = set(self.str2array(state, 'Kp'))
    
    @property
    def a(self):
        if not hasattr(self, '_analysis'):
            self._analysis = Analysis(self.names, self.X, self.K, self.Kp)
        return self._analysis
            
    def __len__(self):
        """
        My length is the number of records in my roster.
        """
        return len(self.K)

    def __getitem__(self, k):
        """
        Access the parameter values corresponding to index I{k} of my I{K}
        list.
        """
        kr = self.K[k]
        return self.X[kr,1:]

    def __iter__(self):
        """
        I iterate over 1-D Numpy arrays of parameter values in ascending
        order of the SSEs they resulted in.
        """
        for kr in self.K:
            yield self.X[kr,1:]
    
    def clear(self):
        del self.K[:]
        self.Kp.clear()

    def sim(self, a, b):
        """
        Returns a scalar between 0.0 and 1.0 that indicates the similarity
        between 1-D arrays I{a} and I{b}.
        """
        dp = np.dot(a, b)
        m = [np.sqrt(np.sum(np.square(x))) for x in [a, b]]
        s = dp/np.prod(m)
        s = min(m)/max(m)
        return s
        
    def mostExpendable(self):
        """
        Returns the row index to my I{X} array of the SSE+values
        combination that is most expendable (closest to another one,
        and not currently in the population).

        If all are currently in the population, something went wrong,
        but returns the row index for the worst SSE as a fail-safe.

        B{TODO}: Optimize by selecting a subset array from I{X} and
        computing dot products and magnitudes of all its rows at once.
        """
        # TODO: Optimize
        X = self.X / np.sum(self.X, axis=0, initial=1E-30)
        # Weight of SSE column is equal to weight of all other columns
        # put together
        X[:,0] *= X.shape[1] - 1
        sMost = 0
        for kr in self.K:
            for kr_other in self.K:
                if kr == kr_other or kr in self.Kp:
                    continue
                s = self.sim(X[kr,:], X[kr_other,:])
                if s > sMost:
                    kMost = kr
                    sMost = s
        return kMost
    
    def add(self, i):
        """
        Adds the SSE and parameter values of the supplied individual I{i}
        to my roster.

        If the roster is already full, bumps the record deemed most
        expendable by L{mostExpendable} before adding a record for
        I{i}.

        Returns the row index to my SSE+values array I{X} of the
        record for I{i}.
        """
        N = len(self.K)
        SV = np.array([i.SSE] + list(i.values))
        if N == 0:
            # First addition
            self.X[0,:] = SV
            self.K.append(0)
            self.Kp.add(0)
            return 0
        if N == self.N_max:
            # Roster is full, we will need to bump somebody before adding
            kr = self.mostExpendable()
            self.K.remove(kr)
        # Find the index in K of the row index for the closest
        # recorded SSE above i.SSE
        k = np.searchsorted(self.X[self.K,0], i.SSE)
        # Pick a row index for the new record
        for kr in self.K:
            if kr > 0 and kr-1 not in self.K:
                kr -= 1
                break
            if kr < N-1 and kr+1 not in self.K:
                kr += 1
                break
        else: kr = N
        self.X[kr,:] = SV
        self.K.insert(k, kr)
        self.Kp.add(kr)
        return kr

    def notInPop(self, kr):
        self.Kp.remove(kr)


