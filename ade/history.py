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

from yampex import plot

from util import *


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

    def add(self, i):
        """
        Adds the SSE and parameter values of the supplied individual I{i}
        to my roster, possibly bumping the worst one already there to
        make room.
        """
        SSE = i.SSE
        N = len(self.SSEs)
        if N >= self.N_max:
            if SSE >= self.SSEs[-1]:
                return
            self.SSEs.pop()
            kk = self.K.pop()
        else: kk = N-1                
        k = np.searchsorted(self.SSEs, SSE)
        self.SSEs.insert(k, SSE)
        self.K.insert(k, kk)
        self.X[kk,:] = i.values
    
    def value_vs_SSE(self, *names, **kw):
        """
        Returns a 1-D Numpy array of the SSEs of my individuals and
        matching 1-D Numpy arrays for each of the named parameter
        values.

        @keyword maxRatio: Set this to specify a maximum ratio between
            an included individual's SSE and the best individual's
            SSE.
        """
        SSE_a = np.array(self.SSEs)
        maxRatio = kw.get('maxRatio', 1000)
        SSE_best = SSE_a[0]
        K = np.flatnonzero(SSE_a/SSE_best <= maxRatio)
        result = [SSE_a[K]]
        K = np.array(self.K)[K]
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
        if not names:
            names = sorted(self.names)
        elif [isinstance(x, int) for x in names] == [True, True]:
            names = sorted(self.names)[slice(*names)]
        XY = self.value_vs_SSE(*names, **kw)
        N = len(XY) - 1
        kList = range(N)
        while kList:
            N = min([9, len(kList)])
            kkList = kList[:N]; kList = kList[N:]
            Nc = 1 if N == 1 else 3 if N > 6 else 2
            pt = plot.Plotter(N, Nc=Nc)
            pt.add_marker('.', 2.0); pt.add_line(""); pt.use_grid()
            with pt as sp:
                for k in kkList:
                    name = names[k]
                    sp.set_title(name)
                    sp(XY[0], XY[k+1])
        pt.showAll()

