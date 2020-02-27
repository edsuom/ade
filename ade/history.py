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
from numpy.polynomial import polynomial as poly
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

    If no such string is present, an empty array is constructed and
    returned.
    """
    text = state.get(name, None)
    if text is None:
        return np.array([])
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

    @ivar names: A sequence of the names of all the parameters.
    """
    fmms = (
        (0.05, 'o', 3.0),
        (0.10, 'o', 2.0),
        (0.20, '.', 1.5),
        (0.50, '.', 1.0),
        (0.70, '.', 0.5),
        (1.01, '.', 0.0),
    )
    
    def __init__(self, names, X, K, Kp=set(), Kn=set()):
        self.names = names
        self.X = X
        self.K = K
        self.Kp = Kp
        self.Kn = Kn

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

        Only combinations where the first column index is lower than
        the second are yielded. This avoids duplication by limiting
        the iteration to the upper right triangle in a 2-D combination
        matrix where the first index is for rows and the second is for
        columns.
        """
        kkR = []
        Nc = self.X.shape[1]
        for k1 in range(1, Nc):
            for k2 in range(1, Nc):
                if k2 > k1: 
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
            mmSSE = SSE[-1] - SSE[0]
            return SSE[0] + f*mmSSE

        # 1-D array of SSEs, sorted
        SSE = self.X[self.K,0]
        I = np.logical_and(SSE >= fSSE(f1), SSE < fSSE(f2))
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
    
    def value_vs_SSE(self, names, **kw):
        """
        Returns a 1-D Numpy array of the SSEs of my individuals and
        matching 1-D Numpy arrays for each of the parameter
        values in I{names}.

        @keyword inPop: Set C{True} to only include individuals in the
            population.
        
        @keyword notInPop: Set C{True} to only include individuals who
            were once but no longer are in the population.

        @keyword neverInPop: Set C{True} to only include individuals
            who were never in the population.

        @keyword maxRatio: Set this to specify a maximum ratio between
            an included individual's SSE and the best individual's
            SSE.
        """
        def subset(kCol):
            return self.X[K,kCol] if len(K) else K
        
        names = self.args2names(names)
        maxRatio = kw.get('maxRatio', 1000)
        SSE_best = self.X[self.K[0],0]
        if kw.get('inPop', False):
            K = [k for k in self.K if k in self.Kp]
        elif kw.get('notInPop', False):
            K = [k for k in self.K if k not in self.Kp and k not in self.Kn]
        elif kw.get('neverInPop', False):
            K = [k for k in self.K if k in self.Kn]
        else: K = self.K
        KK = np.flatnonzero(self.X[K,0]/SSE_best <= maxRatio)
        K = np.array(K)[KK]
        result = [subset(0)]
        for name in names:
            result.append(subset(self.name2k(name)))
        return result

    def lineFit(self, k1, k2, K):
        """
        Returns the slope and y-intercept of a line that has a best fit to
        the SSE-weighted data in column vectors of my I{X} array at
        I{k1} (x) and I{k2} (y), with elements in I{K} selected.

        For the best (lowest SSE) pair, the weight is 1.0. If the
        worst SSE is many times larger, the worst pair's weight is
        approximately M{2*SSE_best/SSE_worst}.
        """
        SSE = self.X[K,0]
        SSE_min = SSE.min()
        W = 2*SSE_min / (SSE + SSE_min)
        b, m = poly.polyfit(self.X[K,k1], self.X[K,k2], 1, w=W)
        return m, b

    def pick_N(self, kList):
        """
        Returns a sensible number of subplots to show in the next figure,
        given a supplied list I{kList} of column indices of remaining
        parameters to show.

        Favors 2x2 and 3x3 plots. Single-subplot figures have too much
        empty space and are visually confusing.
        """
        N_left = len(kList)
        if N_left in (8, 10):
            return 4
        if N_left > 8:
            return 9
        return N_left

    def _makePlotter(self, *args, **kw):
        """
        Returns a L{Plotter} object, constructed with the supplied args
        and/or keywords, with some global options set.
        """
        pt = Plotter(*args, **kw)
        pt.add_line(""); pt.use_minorTicks('y'); pt.use_grid()
        return pt
    
    def plot(self, names, **kw):
        """
        Plots the values versus SSE for each of the parameters in
        I{names}. Accepts keywords used for L{value_vs_SSE} (though
        only I{inPop} is honored in this method), plus I{noShow}.
        
        If there are two integer values in I{names}, they are used to
        select a range of my I{names} sequence. (Seldom used.)

        @keyword noShow: Set C{True} to return the C{Plotter} object
            from the last Matplotlib C{Figure} plotted instead of
            calling C{showAll} on it, thus allowing you to do so at
            your convenience.
        """
        noShow = kw.pop('noShow', False)
        names = self.args2names(names)
        inPop = kw.get('inPop', False)
        kw['inPop'] = True
        XYp = self.value_vs_SSE(names, **kw)
        if not inPop:
            kw['inPop'] = False
            kw['notInPop'] = True
            XYn = self.value_vs_SSE(names, **kw)
            kw['notInPop'] = False
            kw['neverInPop'] = True
            XYr = self.value_vs_SSE(names, **kw)
        # kList is a range of indices to the XYp, XYn, and XYr lists
        # of 1-D Numpy arrays
        N = len(XYp) - 1
        kList = range(N)
        while kList:
            N = self.pick_N(kList)
            kkList = kList[:N]; kList = kList[N:]
            Nc = 1 if N == 1 else 3 if N > 6 else 2
            pt = self._makePlotter(N, Nc=Nc)
            pt.add_marker('o', 2.0); pt.add_color('red')
            if not inPop:
                pt.add_marker('o', 2.0); pt.add_color('blue')
                pt.add_marker('.', 1.5); pt.add_color('#303030')
            with pt as sp:
                for k in kkList:
                    name = names[k]
                    sp.set_title(name)
                    ax = sp(XYp[0], XYp[k+1])
                    if not inPop:
                        ax.plot(XYn[0], XYn[k+1])
                        ax.plot(XYr[0], XYr[k+1])
        if noShow: return pt
        pt.showAll()

    def prettyLine(self, m, b):
        return sub("Y={:+.6g}*X {} {:.6g}", m, "-" if b < 0 else "+", abs(b))
        
    def plotXY(self, arg1, arg2, sp=None, useFraction=False):
        """
        Plots the values of the parameter at column I{k2} of my I{X} array
        versus the values of the parameter at column I{k1}, with a
        rough indication of the SSEs involved.

        Also plots a best-fit line determined by I{lineFit} with the
        pairs having the best 50% of the SSEs.

        Returns a 4-tuple with the x- and y-axis labels and the slope
        and y-intercept of the best-fit line.
        """
        def plot(sp):
            xName = self.k2name(k1)
            yName = self.k2name(k2)
            sp.set_xlabel(xName)
            sp.set_ylabel(yName)
            K =  self.Kp12(0, 0.5)
            m, b = self.lineFit(k1, k2, K)
            sp.add_annotation(0, self.prettyLine(m, b))
            X = self.X[K,k1]
            X = np.array([X.min(), X.max()])
            ax = sp(X, m*X+b, '-r')
            f1 = 0.0
            kw = {'color': "blue"}
            for f2, mk, ms in self.fmms:
                if ms:
                    K = self.Kf12(f1, f2) if useFraction else self.Kp12(f1, f2)
                    kw['marker'] = mk
                    kw['markersize'] = ms
                    X, Y = [self.X[K,x] for x in (k1, k2)]
                    ax.plot(X, Y, **kw)
                f1 = f2
            return xName, yName, m, b

        k1 = arg1 if isinstance(arg1, int) else self.name2k(arg1)
        k2 = arg2 if isinstance(arg2, int) else self.name2k(arg2)
        if sp is None:
            pt = self._makePlotter(1)
            with pt as sp:
                result = plot(sp)
            pt.show()
            return result
        return plot(sp)
    
    def plotCorrelated(self, name=None, N=4, noShow=False, verbose=False):
        """
        Plots values of four pairs of parameters with the highest
        correlation. The higher the SSE for a given combination of
        values, the less prominent the point will be in the plot.

        You can specify one parameter that must be included. Then the
        correlations checked are with everything else.
        
        Seeing a very high correlation in one of these plots is an
        indication that you should somehow consolidate the correlated
        parameters or at least make them explicitly dependent on each
        other at the outset, so DE doesn't waste effort searching all
        the deserted fitness landscape outside the narrow ellipse of
        their correlated values.

        @keyword noShow: Set C{True} to return the C{Plotter} object
            from the last Matplotlib C{Figure} plotted instead of
            calling C{showAll} on it, thus allowing you to do so at
            your convenience.

        """
        Np = self.X.shape[1] - 1
        Ncombos = Np*(Np-1)/2
        if Ncombos == 0: return
        if Ncombos < N: N = Ncombos
        pt = self._makePlotter(N)
        with pt as sp:
            count = 0
            for stuff in self.correlator():
                k1, k2, R = stuff
                if name and name != self.k2name(k1): continue
                corr = sub("R={:+.3f}", R)
                sp.add_textBox("SE" if R > 0 else "NE", corr)
                xName, yName, m, b = self.plotXY(k1, k2, sp)
                if verbose:
                    firstPart = sub("{}:{} ({})", xName, yName, corr)
                    print(sub(
                        "{:>30s}  {}", firstPart, self.prettyLine(m, b)))
                count += 1
                if count == N: break
        if noShow: return pt
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

    @cvar Kn_penalty: The multiplicative penalty to impose on the
        computed difference to favor pairs where at least one member
        has been a population member.
    """
    Np_max = 10000
    Kn_penalty = 2.0
    
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
        Sets my I{K} and I{Kn} to empty sets and I{S} to C{None},
        returning me to a virginal state.
        """
        self.K = set()
        self.Kn = set()
        self.S = None
    
    def setRow(self, k, Z, neverInPop=False):
        """
        Call with the row index to my I{X} array and a 1-D array I{Z} with
        the SSE+values that are to be stored in that row.

        Nulls out my I{S} scaling array to force re-computation of the
        column-wise variances when L{__call__} runs next, because the
        new row entry will change them.

        Never call this with an C{inf} or C{NaN} anywhere in I{Z}. An
        exception will be raised if you try.

        @keyword neverInPop: Set C{True} to indicate that this
            SSE+value was never in the population and thus should be
            less more to be bumped in favor of a newcomer during size
            limiting.
        """
        if not np.all(np.isfinite(Z)):
            raise ValueError("Non-finite value in Z")
        self.X[k,:] = Z
        self.K.add(k)
        if neverInPop: self.Kn.add(k)
        self.S = None
    
    def clearRow(self, k):
        """
        Call with the row index to my I{X} array to have me disregard the
        SSE+values that are to be stored in that row. If the index is
        in my I{Kn} set, discards it from there.

        Nulls out my I{S} scaling array to force re-computation of the
        column-wise variances when L{__call__} runs next, because
        disregarding the row entry.
        """
        if k in self.K:
            self.K.remove(k)
            self.Kn.discard(k)
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

        If I have just a single SSE+value combination, the Deferred
        fires with that combination's row index in I{X}. If there are
        no legit combinations, it fires with C{None}.

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
                    K1 = self.pairs_all()
                else: K1 = self.pairs_sampled(Np)
                yield
            else: K1 = K
            if self.S is None:
                XK = self.X[list(self.K),:]
                self.S = 1.0 / (np.var(XK, axis=0) + 1E-20)
                yield
            # Calculate difference
            X = self.X[K1[:,0]]
            yield
            X -= self.X[K1[:,1]]
            yield
            D = np.square(X)
            yield
            D *= self.S
            yield
            D = np.sum(D, axis=1)
            yield
            # Divide difference by mean SSE to favor lower-SSE history
            SSEs = [self.X[K2,0] for K2 in [K1[:,k] for k in (0, 1)]]
            D /= np.mean(np.column_stack(SSEs), axis=1)
            yield
            # Divide difference by a computed amount when the first
            # item was never in the population, to keep a substantial
            # fraction of the non-population history reserved for
            # those who once were in the population
            penalize = [1 if k1 in self.Kn else 0 for k1, k2 in K1]
            # The penalty increases dramatically if the history comes
            # to have more never-population records than those that
            # have been in the population
            N_neverpop = len(self.Kn)
            if N_neverpop:
                Kn_penalty = 1 + np.exp(12*(N_neverpop/len(D) - 0.4))
                D /= np.choose(penalize, [1, Kn_penalty])
            yield
            if K is None:
                kr = K1[np.argmin(D),0]
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

    @ivar Kn: A set of the values (not indices) of I{K} that are for
        individuals who never were in the population.
    
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
        self.K = []; self.Kp = set(); self.Kn = set()
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
            'Kn': seq2str(list(self.Kn), 'u2'),
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
        self.Kn = set(str2array(state, 'Kn'))
        self.kr = state['kr']
        self._initialize()

    def _initialize(self):
        self.a = Analysis(self.names, self.X, self.K, self.Kp, self.Kn)
        self.cpf = ClosestPairFinder(self.N_max, len(self.names)+1)
        for kr in self.K:
            if kr in self.Kp: continue
            self.cpf.setRow(kr, self.X[kr,:], neverInPop=(kr in self.Kn))
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
            self.Kn.clear()
            self.cpf.clear()
            self.kr.clear()
            self.N_total = 0
        return self.dLock.run(gotLock)

    def value_vs_SSE(self, *args, **kw):
        def gotLock():
            return self.a.value_vs_SSE(*args, **kw)
        return self.dLock.run(gotLock)
        
    @defer.inlineCallbacks
    def add(self, i, neverInPop=False):
        """
        Adds the SSE and parameter values of the supplied individual I{i}
        to my roster, unless it has an SSE of C{inf}, in which case it
        is ignored.

        If the roster is already full, bumps the record deemed most
        expendable before adding a record for I{i}. That determination
        is made by a call to my L{ClosestPairFinder} instance I{cpf}.
        
        Returns a C{Deferred} that fires with the row index of the new
        record when it has been written, or C{None} if no record was
        written.

        @keyword neverInPop: Set C{True} to have the individual added
            without ever having been part of the population.
        """
        def kkr():
            """
            Returns (1) the index I{k} of my I{K} list where the row index of
            the new record should appear in my I{X} array, and (2)
            that row index I{kr}.

            First, index I{k} is obtained, by seeing where the I{K}
            list points to a record with an SSE closest but above the
            new one. Then each row index in the I{K} list is examined
            to see if the previous row of my I{X} array is
            unallocated. If so, that is the row index for the new
            record. Otherwise, is the next row of my I{X} array is
            unallocated, that is used instead. If both adjacent rows
            of I{X} are already allocated, the next row index in the
            I{K} list is examined.

            If there are no row indices in I{K} that point to a row of
            I{X} with an unallocated adjacent row, the row index is
            determined to be the current length of I{k}.
            """
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
            """
            Writes a 1-D Numpy array with SSE+values to row I{kr} of my I{X}
            array, and inserts the row index I{kr} into my I{K} list
            at position I{k}.
            """
            SV = np.array([i.SSE] + list(i.values))
            self.X[kr,:] = SV
            self.K.insert(k, kr)
            self.N_total += 1           # Add to the lifetime total count
            if neverInPop:
                # This row has been added without ever being a
                # population member, so cpf considers it along with
                # the other non-population entries
                self.cpf.setRow(kr, self.X[kr,:], neverInPop=True)
                self.Kn.add(kr)
            else:
                self.Kp.add(kr)
                # This row is starting out as a population member, so
                # have cpf initially disregard it
                self.cpf.clearRow(kr)
                # Add to the individual-row map so it can be removed
                # from the population later
                self.kr[hash(i)] = kr

        if np.isfinite(i.SSE):
            yield self.dLock.acquire()
            N = len(self.K)
            if N == 0:
                # First addition
                k, kr = 0, 0
            elif N < self.N_max:
                # Roster not yet full, no need to search for somebody
                # to bump first
                k, kr = kkr()
            else:
                # Roster is full, we will need to bump somebody (those
                # in the current population are protected and exempt)
                # before adding
                kr = yield self.cpf()
                if kr is not None: self.purge(kr)
                k, kr = kkr()
            writeRecord(k, kr)
            self.dLock.release()
        else: kr = None
        defer.returnValue(kr)

    def purge(self, kr):
        """
        Purges my history of the record at row index I{kr}.

        Removes the row index from my I{K} list and has my I{cpf}
        instance of L{ClosestPairFinder} disregard the row, because
        it's now gone.

        B{Note}: Does not remove the index from the values of my I{kr}
        dict, as that is a time-consuming process and the caller can
        likely just clear the whole thing anyhow.
        """
        if kr not in self.K:
            raise IndexError(sub("No row index {} in my K list!", kr))
        self.K.remove(kr)
        self.cpf.clearRow(kr)
        self.Kp.discard(kr)     # May already have been discarded with .pop
        self.Kn.discard(kr)
        
    def notInPop(self, x):
        """
        Call this with an integer row index or an I{Individual} instance
        that was added via L{add} to remove its row of my I{X} array
        from being considered part of the current population.
        """
        def gotLock():
            if isinstance(x, int):
                kr = x
            else:
                # Must be an Individual
                key = hash(x)
                if key not in self.kr: return
                kr = self.kr.pop(key)
            self.Kp.discard(kr)
            # This row is no longer a population member and is thus
            # expendable, so have cpf start considering it
            self.cpf.setRow(kr, self.X[kr,:])
        return self.dLock.run(gotLock)

    def purgePop(self):
        """
        Purges the history of all members of the current
        population. (Presumably, they will get added back again after
        re-evaluation.)
        """
        def gotLock():
            while self.Kp:
                kr = self.Kp.pop()
                self.purge(kr)
            self.kr.clear()
        return self.dLock.run(gotLock)
        
