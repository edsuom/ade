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
    I maintain a roster of the L{Individual} objects that a
    L{Population} has had and possibly replaced.

    @cvar N_max: The most individuals I can have in my roster. If more
        are added when it's full, the highest-SSE one is bumped to
        make room.

    @ivar iList: My roster of individuals, maintained in ascending
        order of SSE.
    """
    N_max = 1000
    
    def __init__(self, names):
        self.names = names
        self.iList = []
        self.SSEs = []

    def __getstate__(self):
        return {'names': self.names, 'iList': self.iList, 'SSEs': self.SSEs}
        
    def __setstate__(self, state):
        self.names = state['names']
        self.iList = state['iList']
        self.SSEs = state['SSEs']

    def __len__(self):
        """
        My length is the number of individuals in my roster.
        """
        return len(self.iList)

    def clear(self):
        del self.iList[:]
        del self.SSEs[:]
    
    def add(self, i):
        """
        Adds an individual to my roster, possibly bumping the worst one
        already there to make room.
        """
        SSE = i.SSE
        k = np.searchsorted(self.SSEs, SSE)
        self.iList.insert(k, i)
        self.SSEs.insert(k, SSE)
        if len(self.iList) > self.N_max:
            self.iList.pop()
            self.SSEs.pop()

    def value_vs_SSE(self, *names, **kw):
        """
        Returns a 1-D Numpy array of the SSEs of my individuals and
        matching 1-D Numpy arrays for each of the named parameter
        values.

        @keyword include: A fraction of my roster to include (the best
            individuals). Default is 0.9, which excludes the worst 10%.

        @keyword maxRatio: Set this to specify a maximum ratio between
            an included individual's SSE and the best individual's
            SSE.
        """
        include = kw.get('include', 0.9)
        kMax = int(np.ceil(include*len(self))) - 1
        SSE_a = np.array(self.SSEs[:kMax])
        iList = self.iList[:kMax]
        maxRatio = kw.get('maxRatio', 1000)
        SSE_best = SSE_a[0]
        K = np.flatnonzero(SSE_a/SSE_best <= maxRatio)
        result = [SSE_a[K]]
        iList = [iList[k] for k in K]
        for name in names:
            k = self.names.index(name)
            result.append(np.array([i[k] for i in iList]))
        return result

    def plot(self, *names, **kw):
        """
        Plots the values versus SSE for each of the named
        parameters. Accepts keywords used for L{value_vs_SSE}.
        """
        XY = self.value_vs_SSE(*names, **kw)
        pt = plot.Plotter(len(XY)-1)
        pt.add_marker('x'); pt.add_line(""); pt.use_grid()
        with pt as sp:
            for name, Y in zip(names, XY[1:]):
                sp.set_title(name)
                sp(XY[0], Y)
        pt.show()

