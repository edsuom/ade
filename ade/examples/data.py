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
The L{Data} object handles downloading (if necessary),
decompressing, and reading data files for the L{thermistor2} example.
"""

import os.path, bz2

import numpy as np

from twisted.internet import defer
from twisted.web import client

from yampex.plot import Plotter

from ade.util import *


class Data(Picklable):
    """
    Run L{setup} on an instance of me to decompress and load the CSV
    file identified by my subclass's I{basename}.

    The CSV file isn't included in the I{ade} package and will
    automatically be downloaded from U{edsuom.com}. Here's the privacy
    policy for my site (it's short, as all good privacy policies
    should be)::

        Privacy policy: I don’t sniff out, track, or share anything
        identifying individual visitors to this site. There are no
        cookies or anything in place to let me see where you go on the
        Internetthat’s creepy. All I get (like anyone else with a web
        server), is plain vanilla server logs with “referral” info
        about which web page sent you to this one.

    @cvar basename: The base name of a CSV file (not including any
        extension)

    @cvar ranges: Your subclass may set this to a list of 2-tuples
        that each define an acceptable range of row indices to include
        from the CSV file. (The first row index is zero, and the last
        row index is C{None}.) An empty list (the default) includes
        all rows.
    
    @ivar t: A 1-D Numpy vector containing the number of seconds
        elapsed from the first reading.

    @ivar X: A 2-D Numpy array with one first column for each CSV item
        after the first (time).

    @ivar csvPath: Path to the bzip2-compressed CSV file.

    @ivar weights: An optional 1-D array of weights, one for each row
        in I{X}.
    """
    basename = None
    urlProto = "http://edsuom.com/ade-{}"
    ranges = None
    weightCutoff = 0.2
    
    @defer.inlineCallbacks
    def setup(self):
        """
        Calling this gets you a C{Deferred} that fires when setup is done
        and my I{t}, I{X}, and possibly my I{weights} ivars are ready.
        """
        if self.basename is None:
            raise AttributeError("No CSV file defined")
        self.csvPath = sub("{}.csv.bz2", self.basename)
        if not os.path.exists(self.csvPath):
            if url is None:
                print(sub(
                    "No CSV file '{}' exists in this directory, "+\
                    "and no download URL was provided!", self.csvPath))
            print(sub("Downloading {} data file from edsuom.com..."))
            yield client.downloadPage(self.csvURL, self.csvPath)
            print("\tDone")
        value_lists = []; T_counts = {}
        print("Decompressing and parsing tempdump.csv.bz2...")
        with bz2.BZ2File(self.csvPath, 'r') as bh:
            while True:
                line = bh.readline().strip()
                if not line:
                    if value_lists:
                        break
                    else: continue
                if line.startswith('#'):
                    continue
                value_list = [float(x.strip()) for x in line.split(',')]
                value_lists.append(value_list)
        print("\tDone")
        print("Doing array conversions...")
        value_lists.sort(None, lambda x: x[0])
        t_list = []
        t0 = value_lists[0][0]
        if self.ranges:
            selected_value_lists = []
            for k, value_list in enumerate(value_lists):
                for k0, k1 in self.ranges:
                    if k >= k0 and (k1 is None or k < k1):
                        t_list.append(value_list[0] - t0)
                        selected_value_lists.append(value_list[1:])
                        break
        else: selected_value_lists = value_lists
        print(sub(
            "\tRead {:d} of {:d} data points", len(selected_value_lists), k+1))
        self.t = np.array(t_list)
        self.X = np.array(selected_value_lists)
        self.setWeights()
        print("\tDone")

    def setWeights(self):
        """
        Override this in your sublcass to set my I{weights} attribute to a
        1-D Numpy array of weights, one for each CSV file row.
        """
        
    def plot(self):
        """
        Plots my data with annotations.
        """
        if self.weights is None:
            I_below = []
            I_above = np.arange(self.X.shape[0])
        else:
            I_below = np.flatnonzero(self.weights < self.weightCutoff))
            I_above = np.flatnonzero(self.weights >= self.weightCutoff)
        T = self.X[I_above, 0]
        T_cutoff = self.X[I_below, 0]
        pp = Plotter(3, 2, width=12, height=10)
        pp.add_plotKeyword('markersize', 1)
        pp.add_marker('.')
        with pp as p:
            for k in range(6):
                R = self.X[I_above, k+1]
                p.set_title(sub("Thermistor #{:d}", k+1))
                ax = p(R, T)
                R = self.X[I_below, k+1]
                ax.plot(R, T_cutoff, 'r.', markersize=1)
        pp.show()