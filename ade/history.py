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

import random
from cStringIO import StringIO

import numpy as np
from twisted.internet import defer, task

from yampex.plot import Plotter

from util import *


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
        self.K = K
        self.Kp = Kp

    def corr(self, k1, k2):
        """
        Returns the correlation coefficient between parameter values of
        column I{k1} and column I{k2} in my I{X} array.

        B{TODO}: Make SSE-weighted (lower=more weight).
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
        return np.array(self.K)[np.flatnonzero(I)]

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
        the specified parameter I{name}. The reverse of L{k2name}.
        """
        return self.names.index(name) + 1

    def k2name(self, k):
        """
        Returns the parameter I{name} for the the specified column index
        I{k} in my I{X} array. The reverse of L{name2k}.
        """
        return self.names[k-1]
    
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
                    ax.plot(XYn[0], XYn[k+1])
        pt.showAll()

    def plotXY(self, arg1, arg2, sp=None, useFraction=False):
        """
        Plots the values of the parameter at column I{k2} of my I{X} array
        versus the values of the parameter at column I{k1}, with a
        rough indication of the SSEs involved.
        """
        def plot(sp):
            sp.set_xlabel(self.k2name(k1))
            sp.set_ylabel(self.k2name(k2))
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
        Np = self.X.shape[1] - 1
        Ncombos = Np*(Np-1)/2
        if Ncombos == 0: return
        if Ncombos < N: N = Ncombos
        pt = Plotter(N)
        pt.add_line(""); pt.use_grid()
        with pt as sp:
            for k, stuff in enumerate(self.correlator()):
                if k == N: break
                k1, k2, R = stuff
                sp.add_textBox("SE" if R > 0 else "NE", "R={:+.3f}", R)
                self.plotXY(k1, k2, sp)
        pt.show()
        

class ClosestPairFinder(object):
    """
    I determine which of two rows are most similar of a Numpy 1-D
    array I{X} I maintain having I{Nr} rows and I{Nc} columns.

    The array's first column contains the SSEs of individuals and the
    remaining columns contain the parameter values that resulted in
    each SSE. The array values are normalized such that the average of
    of each of the columns is 1.0.

    @ivar Nc: The number of columns, SSE + parameter values.

    @ivar X: My Numpy 1-D array having up to I{Nr} active rows and
        exactly I{Nc} columns of SSE+values combinations.

    @ivar S: A Numpy 1-D array having a scaling factor for the
        sum-of-squared-differences calculated by L{__call__}. The
        scaling factor is reciprocal of the variance of all active
        rows in I{X}, or C{None} if the variance needs to be
        (re)computed.

    @ivar K: A set of indices to the active rows in I{X}.

    @cvar Np_max: The maximum number of row pairs to examine for
        differences in L{__call__}.
    """
    Np_max = 10000
    
    def __init__(self, Nr, Nc):
        """
        C{ClosestPairFinder(Nr, Nc)}
        """
        self.Nr = Nr
        self.Nc = Nc
        self.X = np.empty((Nr, Nc))
        self.clear()

    def clear(self):
        """
        Sets my I{K} to an empty set and I{S} to C{None}, returning me to
        a virginal state.
        """
        self.K = set()
        self.S = None
    
    def setRow(self, k, Z):
        """
        Call with the row index to my I{X} array and a 1-D array I{Z} with
        the SSE+values that are to be stored in that row.

        Nulls out my I{S} scaling array to force re-computation of the
        column-wise variances when L{__call__} runs next, because the
        new row entry will change them.
        """
        self.X[k,:] = Z
        self.K.add(k)
        self.S = None
    
    def clearRow(self, k):
        """
        Call with the row index to my I{X} array to have me disregard the
        SSE+values that are to be stored in that row.

        Nulls out my I{S} scaling array to force re-computation of the
        column-wise variances when L{__call__} runs next, because
        disregarding the row entry.
        """
        if k in self.K:
            self.K.remove(k)
            self.S = None

    def pairs_sampled(self, N):
        """
        Returns a 2-D Numpy array of I{N} pairs of separate row indices to
        my I{X} array, randomly sampled from my set I{K} with
        replacement.

        The second value in each row of the returned array must be
        greater than the first value. (There may be duplicate rows,
        however.) Sampling of I{K} continues until there are enough
        suitable rows.
        """
        Nr = len(self.K)
        mult = 2.1*Nr/(Nr-1)
        Ns = int(mult*N)
        K = np.random.choice(list(self.K), (Ns, 2))
        K = K[np.flatnonzero(K[:,1] > K[:,0]),:][:N]
        Ns = K.shape[0]
        if Ns < N:
            K = np.row_stack([K, self.pairs_sampled(N-Ns)])
        return K

    def pairs_all(self):
        """
        Returns a 2-D Numpy array of all pairs of separate row indices to
        my I{X} array where the second value in each pair is greater
        than the first value.

        The returned array will have M{N*(N-1)/2} rows and two
        columns, where I{N} is the length of my I{K} set of row indices.
        """
        K1, K2 = np.meshgrid(list(self.K), list(self.K), indexing='ij')
        K12 = np.column_stack([K1.flatten(), K2.flatten()])
        return K12[np.flatnonzero(K12[:,1] > K12[:,0])]
    
    def __call__(self, Np=None, K=None):
        """
        Returns a C{Deferred} that fires with the row index to my I{X}
        array of the SSE+values combination that is most expendable
        (closest to another one, and not currently in the population).

        If I have just a single SSE+value combination, returns its row
        index in I{X}.

        If the maximum number of pairs I{Np} to examine (default
        I{Np_max}) is greater than M{N*(N-1)/2}, where I{N} is the
        length of my I{K} set of row indices, L{pairs_all} is called
        to examine all suitable pairs.

        Otherwise, L{pairs_sampled} is called instead and examination
        is limited to a random sample of I{Np} suitable pairs. With
        the default I{Np_max} of 10000, this occurs at C{N>142}. With
        I{Np_max} of 1000, it occurs with C{N>45}. Since the I{N_max}
        of L{History} has a default of 1000, L{pairs_sampled} is
        what's going to be used in all practical situations.

        The similarity is determined from the sum of squared
        differences between two rows, divided by the column-wise
        variance of all (active) rows.

        @keyword Np: Set to the maximum number of pairs to
            examine. Default is I{Np_max}.

        @keyword K: For unit testing only: Supply a 2-D Numpy array of
            pairs of row indices, and the C{Deferred} will fire with
            just the sum-of-squares difference between each pair.
        """
        def calcerator(K):
            """
            Iterates over computationally intensive chunks of processing.
            """
            if K is None:
                if Nr*(Nr-1)/2 < Np:
                    KK = self.pairs_all()
                else: KK = self.pairs_sampled(Np)
                yield
            else: KK = K
            if self.S is None:
                XK = self.X[list(self.K),:]
                self.S = 1.0 / (np.var(XK, axis=0) + 1E-20)
                yield
            X = self.X[KK[:,0]]
            yield
            X -= self.X[KK[:,1]]
            yield
            D = np.square(X)
            yield
            D *= self.S
            yield
            D = np.sum(D, axis=1)
            yield
            if K is None:
                kr = KK[np.argmin(D),0]
                result(kr)
            else: result(D)
            
        result = Bag()
        Nr = len(self.K)
        if Nr == 1:
            return defer.succeed(list(self.K)[0])
        if Np is None: Np = self.Np_max
        d = task.cooperate(calcerator(K)).whenDone()
        d.addCallback(lambda _: result())
        return d


class History(object):
    """
    I maintain a roster of the parameter values and SSEs of
    I{Individual} objects that a L{Population} has had and possibly
    replaced.

    @keyword N_max: The most records I can have in my roster. When the
        roster is full, adding a non-duplicative I{Individual} will
        bump the highest-SSE one currently in the roster to make
        room. The default of 1500 seems like a sensible compromise
        between reasonably compact C{.dat} file size and informative
        plots.
    
    @ivar names: A sequence of my individuals' parameter names,
        supplied as the sole constructor argument.
    
    @ivar X: A 2-D Numpy array of SSEs (first column) and parameter
        values (remaining columns) of one individual.

    @ivar K: A list of indices to rows of I{X}, each entry in the list
        corresponding to a row of I{X}.

    @ivar Kp: A set of the values (not indices) of I{K} that are for
        individuals currently in the population.

    @ivar kr: A dict containing row indices, keyed by the hashes of
        I{Individual} instances.
    """
    N_max = 1500
    
    def __init__(self, names, N_max=None):
        """
        C{History(names, N_max=None)}
        """
        self.names = names
        if N_max: self.N_max = N_max
        self.N_total = 0
        self.X = np.zeros((self.N_max, len(names)+1), dtype='f4')
        self.K = []; self.Kp = set()
        self.kr = {}
        self._initialize()

    def __getstate__(self):
        """
        For storage-efficient pickling.
        """
        return {
            'names': self.names,
            'N_max': self.N_max,
            'N_total': self.N_total,
            'X': seq2str(self.X),
            'K': seq2str(self.K, 'u2'),
            'Kp': seq2str(list(self.Kp), 'u2'),
            'kr': self.kr,
        }
    
    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.names = state['names']
        self.N_max = state['N_max']
        self.N_total = state['N_total']
        self.X = str2array(state, 'X')
        self.K = list(str2array(state, 'K'))
        self.Kp = set(str2array(state, 'Kp'))
        self.kr = state['kr']
        self._initialize()

    def _initialize(self):
        self.a = Analysis(self.names, self.X, self.K, self.Kp)
        self.cpf = ClosestPairFinder(self.N_max, len(self.names)+1)
        for kr in self.K:
            if kr in self.Kp: continue
            self.cpf.setRow(kr, self.X[kr,:])
        self.dLock = defer.DeferredLock()

    def shutdown(self):
        return self.dLock.acquire().addCallback(lambda _: self.dLock.release())

    def __len__(self):
        """
        My length is the number of records in my roster.

        B{Note}: Immediate result, not locked! Mostly for unit testing.
        """
        return len(self.K)

    def __getitem__(self, k):
        """
        Access the SSE and parameter values corresponding to index I{k} of
        my I{K} list.

        B{Note}: Immediate result, not locked! Mostly for unit testing.
        """
        kr = self.K[k]
        return self.X[kr,:]

    def __iter__(self):
        """
        I iterate over 1-D Numpy arrays of parameter values in ascending
        order of the SSEs they resulted in.

        B{Note}: Immediate result, not locked! Mostly for unit testing.
        """
        for kr in self.K:
            yield self.X[kr,1:]
    
    def clear(self):
        """
        Call to have me return to a virginal state with no SSE+values
        combinations recorded or considered for removal, an empty
        population, and an I{N_total} of zero.
        """
        def gotLock():
            del self.K[:]
            self.Kp.clear()
            self.cpf.clear()
            self.kr.clear()
            self.N_total = 0
        return self.dLock.run(gotLock)

    def value_vs_SSE(self, *args, **kw):
        def gotLock():
            return self.a.value_vs_SSE(*args, **kw)
        return self.dLock.run(gotLock)
        
    @defer.inlineCallbacks
    def add(self, i):
        """
        Adds the SSE and parameter values of the supplied individual I{i}
        to my roster.

        If the roster is already full, bumps the record deemed most
        expendable before adding a record for I{i}. That determination
        is made by a call to my L{ClosestPairFinder} instance I{cpf}.

        Returns a C{Deferred} that fires with the row index of the new
        record when it has been written.
        """
        def kkr():
            # Find the index in K of the row index for the closest
            # recorded SSE above i.SSE
            k = np.searchsorted(self.X[self.K,0], i.SSE)
            # Pick a row index for the new record
            for kr in self.K:
                if kr > 0 and kr-1 not in self.K:
                    return k, kr-1
                if kr < N-1 and kr+1 not in self.K:
                    return k, kr+1
            return k, N

        def writeRecord(k, kr):
            SV = np.array([i.SSE] + list(i.values))
            self.X[kr,:] = SV
            self.K.insert(k, kr)
            self.Kp.add(kr)
            # This row is starting out as a population member, so have
            # cpf initially disregard it
            self.cpf.clearRow(kr)
            # Add to the lifetime total count
            self.N_total += 1
            # Finally, add to the individual-row map
            self.kr[hash(i)] = kr
        
        yield self.dLock.acquire()
        N = len(self.K)
        if N == 0:
            # First addition
            k, kr = 0, 0
        elif N < self.N_max:
            # Roster not yet full, no need to search for somebody to
            # bump first
            k, kr = kkr()
        else:
            # Roster is full, we will need to bump somebody (those in
            # the current population are protected and exempt) before
            # adding
            kr = yield self.cpf()
            self.K.remove(kr)
            # Have cpf disregard this row, because it's now gone
            self.cpf.clearRow(kr)
            k, kr = kkr()
        writeRecord(k, kr)
        self.dLock.release()
        defer.returnValue(kr)
    
    def notInPop(self, x=None):
        """
        Call this with an integer row index or an I{Individual} instance
        that was added via L{add} to remove its row of my I{X} array
        from being considered part of the current population.

        If called with nothing, removes all rows from being considered
        part of the current population.
        """
        def gotLock():
            if x is None:
                self.Kp.clear()
                self.kr.clear()
                return
            if isinstance(x, int):
                kr = x
            else:
                # Must be an Individual
                key = hash(x)
                if key not in self.kr: return
                kr = self.kr.pop(key)
            self.Kp.remove(kr)
            # This row is no longer a population member and is thus
            # expendable, so have cpf start considering it
            self.cpf.setRow(kr, self.X[kr,:])
        return self.dLock.run(gotLock)
