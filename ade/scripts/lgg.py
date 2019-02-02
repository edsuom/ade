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
Parses a log file produced by running a L{DifferentialEvolution}
object to show parameter values leading up to the best solution it
found before terminating.
"""

import sys, os.path, re, pdb, traceback

import numpy as np

from ade.util import *


class RowManager(object):
    """
    I manage rows of parameter combinations, keyed by SSE and added in
    the order they appeared in an I{ade} log file.

    Construct an instance of me with a list of the parameter I{names}
    you're interested in.

    I can iterate over the parameter names, in order.
    """
    maxRows = 40
    reRange = re.compile(r'([a-zA-Z_]+)([0-9]+)\-([0-9]+)$')
    
    def __init__(self, names):
        self.names = []
        for name in names:
            match = self.reRange.match(name)
            if match:
                prefix = match.group(1)
                suffixValues = range(
                    int(match.group(2)), int(match.group(3))+1)
                for x in suffixValues:
                    self.names.append(prefix+str(x))
            else: self.names.append(name)
        self.N = len(self.names)
        self.rp = {}
        self.indices = {}
        for k, name in enumerate(self.names):
            self.indices[name] = k
            self.rp[name] = re.compile(sub(
                r'(^|\s){}\=([\+\-]?[0-9][0-9\.e\E\-\+]+)(\**)(\s|$)', name))
        self.SSE = []
        self.values = []
        self.stars = []

    def __iter__(self):
        for name in self.names:
            yield name

    def __len__(self):
        return self.N

    def add(self, SSE):
        """
        Adds a new SSE entry, returning I{values} and I{stars} lists with
        placeholder C{None} items for each parameter.

        You'll need to set each of those items to something else.
        """
        self.SSE.append(SSE)
        values = [None for x in range(self.N)]
        self.values.append(values)
        stars = [None for x in range(self.N)]
        self.stars.append(stars)
        return values, stars

    def insert(self, container, name, value):
        """
        Inserts I{value} to the supplied list I{container} at the index
        for the parameter with the specified I{name}.
        """
        k = self.indices[name]
        if container[k] is not None:
            raise RuntimeError(sub(
                "A value has already been assigned for '{}'!", name))
        container[k] = value

    def arrays(self):
        """
        Generates a 1-D Numpy array I{SSE} with SSEs, and 2-D arrays of
        I{values} and I{stars} with one row per SSE and one column per
        parameter.
        """
        for name in ('SSE', 'values', 'stars'):
            x = getattr(self, name)
            if None in x:
                raise ValueError("Missing parameter(s)!")
            setattr(self, name, np.array(x))

    def addMatches(self, line, values, stars):
        """
        Adds items for all parameters found in the supplied L{line} to the
        appropriate places in the supplied I{values} and I{stars}
        lists.

        Returns C{True} if there are no items left to add.
        """
        for name in self.rp:
            match = self.rp[name].search(line)
            if not match: continue
            self.insert(values, name, float(match.group(2)))
            self.insert(stars, name, len(match.group(3)))
        return None not in values

    def trim(self, maxRatio=4.0):
        """
        Trims my arrays to only include the best (latest) SSEs.

        Modifies the arrays, so calling this repeatedly will trim them
        some more.
        """
        bestSSE = np.min(self.SSE)
        K = np.flatnonzero(self.SSE <= maxRatio*bestSSE)
        K = K[-self.maxRows:]
        self.SSE = self.SSE[K]
        self.values = self.values[K]
        self.stars = self.stars[K]

    
class Grepper(object):
    """
    I grep your log file for param values.
    """
    colSep = 3
    width = 10
    precision = 5
    reSSE = re.compile(r'SSE=([\d\.eE\-\+]+)')

    def __init__(self, filePath, names):
        self.filePath = filePath
        self.rm = RowManager(names)
    
    def load(self):
        """
        Loads SSEs, and parameter values and stars for those SSEs, from my
        log file.
        """
        with open(self.filePath, 'r') as fh:
            keepLooping = True
            while keepLooping:
                line = fh.readline()
                if not line: break
                match = self.reSSE.match(line)
                if match:
                    SSE = float(match.group(1))
                    values, stars = self.rm.add(SSE)
                    while True:
                        if self.rm.addMatches(line, values, stars):
                            break
                        line = fh.readline()
                        if not line:
                            # This shouldn't happen, because we
                            # shouldn't run out of lines before all
                            # parameter matches have been found for
                            # this SSE
                            keepLooping = False
                            break
        # Have rm make its SSE, values, stars arrays
        self.rm.arrays()
        # Now have it trim the arrays
        self.rm.trim()

    def rj(self, x):
        """
        Returns a right-justified string version of the supplied string or
        float I{x}, with my column I{width}.
        """
        if isinstance(x, str):
            xs = x
        elif x is None:
            return "--"
        else:
            proto = sub("{{:0.{:d}g}}", self.precision)
            xs = sub(proto, x)
        padding = self.width - len(xs)
        return " "*padding + xs
    
    def makeLines(self):
        """
        Returns lines of a text table of parameter values for the best
        SSEs leading up to the end.

        Puts one or two stars ("*" or "**") after the value if they
        appeared after the value in the log file.
        """
        # Heading
        parts = [self.rj("SSE")]
        for name in self.rm:
            parts.append(self.rj(name))
        lines = ["", (" "*self.colSep).join(parts)]
        lines.append("-" * (len(lines[-1])+2))
        # Rows
        prevSSE = None
        for k, SSE in enumerate(self.rm.SSE):
            if prevSSE and np.abs(SSE/prevSSE - 1.0) < 0.01: continue
            prevSSE = SSE
            values = self.rm.values[k,:]
            stars = self.rm.stars[k,:]
            parts = [self.rj(SSE), " "*self.colSep]
            for kk in range(len(self.rm)):
                parts.append(self.rj(values[kk]))
                spaceBetween = "*" * int(stars[kk])
                spaceBetween += " "*(self.colSep-len(spaceBetween))
                parts.append(spaceBetween)
            line = "".join(parts).rstrip()
            lines.append(line)
        lines.append("")
        return lines
        
    def __call__(self):
        self.load()
        lines = self.makeLines()
        print("\n".join(lines))


def main():
    args = list(sys.argv[1:])
    # Parameter names don't have a dot in them, whereas the log file
    # name will
    if '.' in args[0]:
        fileName = args.pop(0)
    else: fileName = "~/pfinder.log"
    filePath = os.path.expanduser(fileName)
    if not os.path.exists(filePath):
        raise RuntimeError(sub("No log file '{}' found", filePath))
    g = Grepper(filePath, args)
    try: g()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
              
if __name__ == '__main__':
    main()

