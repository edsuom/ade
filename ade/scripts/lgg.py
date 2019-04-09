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
C{lgg}: Parses a log file produced by running a
L{DifferentialEvolution} object to show parameter values leading up to
the best solution it found before terminating.

This is very handy for seeing whether you have to redefine bounds for
any of your parameters. It is often useful to tighten bounds for
particular problems. Or you might need to loosen things up if you see
the '*' indicating that the value is close to the lower bound or the
'**' indicating that it's close to the upper bound.

Specify the log file with the first argument. If not specified,
defaults to I{~/pfinder.log}.

Call with no parameters specified to show parameters whose values are
near one edge or another of the bounds in the last few entries of the
log file.

You can use glob patterns when specifying parameters. You can also
specify ranges like I{x1-3}, which matches I{x1}, I{x2}, and I{x3}.
"""

import sys, os.path, re, pdb, traceback, fnmatch
from collections import deque

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
    reNumRange = re.compile(r'([a-zA-Z0-9_]+?)([0-9]+)\-([0-9]+)$')
    reLetterRange = re.compile(r'([a-zA-Z0-9_]+?)([a-z])\-([a-z])$')
    letters = "abcdefghijklmnopqrstuvwxyz"
    
    def __init__(self, names):
        self.setupNames(names)

    def addRange(self, reObj, name, ranger):
        match = reObj.match(name)
        if match:
            prefix = match.group(1)
            suffixValues = ranger(match.group(2), match.group(3))
            for x in suffixValues:
                self.names.append(prefix+str(x))
            return True
        
    def setupNames(self, names):
        def numRange(n1, n2):
            return range(int(n1), int(n2)+1)
        
        def letterRange(c1, c2):
            return self.letters[
                self.letters.index(c1):self.letters.index(c2)+1]
        
        self.names = []
        for name in names:
            if self.addRange(self.reNumRange, name, numRange):
                continue
            if self.addRange(self.reLetterRange, name, letterRange):
                continue
            self.names.append(name)
        self.N = len(self.names)
        self.rp = {}
        self.indices = {}
        for k, name in enumerate(self.names):
            self.indices[name] = k
            self.rp[name] = re.compile(sub(
                r'(^|\s){}\=([\+\-]?[0-9][0-9\.e\E\-\+]*)(\**)(\s|\>|$)', name))
        self.SSE = []
        self.values = []
        self.stars = []

    def __iter__(self):
        for name in self.names:
            yield self.indices[name], name

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
        Adds items for all parameters found in the supplied I{line} to the
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

    def trim(self, maxRatio=2.0, maxDiff=0.01, maxRows=40):
        """
        Trims my arrays to only include the entry with the best (latest)
        SSE and worse (earlier) entries representing a significant
        step in evolution toward the best SSE.

        The following criteria are applied to decide whether entries
        remain:

            1. SSE less than I{maxRatio} (default 2.0) times the best SSE.

            2. SSE or at least one parameter as a relative difference
               of more than I{maxDiff} (default 0.01, i.e., +/-1%)
               from the SSE or parameter of a better entry that is
               closest in SSE.

            3. No more than I{maxRows} displayed (default 40).

        Modifies the arrays, so calling this repeatedly will trim them
        some more.
        """
        def diffEnough(a, b):
            if b == 0:
                return abs(a) > maxDiff
            return abs(a-b)/b > maxDiff
        
        K = np.argsort(self.SSE)
        bestSSE = np.min(self.SSE)
        KS = [K[0]]
        for k in K[1:]:
            SSE = self.SSE[k]
            if SSE > maxRatio*bestSSE: break
            if not diffEnough(SSE, self.SSE[KS[-1]]):
                for kk, value in enumerate(self.values[k]):
                    if diffEnough(value, self.values[KS[-1]][kk]):
                        break
                else: continue
            KS.append(k)
        KS = list(reversed(KS[-maxRows:]))
        self.SSE = self.SSE[KS]
        self.values = self.values[KS]
        self.stars = self.stars[KS]

    
class Grepper(object):
    """
    I grep your log file for param values.
    """
    colSep = 3
    width = 10
    precision = 5
    reSSE = re.compile(r'SSE\=([\d\.eE\-\+]+)')
    reParam = re.compile(
        r'(^|\s)([a-zA-Z]+[_a-zA-Z0-9]*)\='+\
        r'([\+\-]?[0-9][0-9\.e\E\-\+]*)(\**)(\s|\>|$)')

    def __init__(self, filePath, args):
        self.filePath = filePath
        if not args:
            names = self.starredNames()
        else:
            names = []
            for pattern in args:
                for name in self.globNames(pattern):
                    if name not in names:
                        names.append(name)
            names.sort()
        self.rm = RowManager(names)

    def filerator(self):
        """
        Opens my I{filePath} and, for each SSE entry found in it, yields
        2-tuples with the SSE as a float and a dict keyed by name with
        a 2-tuple of (1) parameter value as a string, and (2) stars.
        """
        with open(self.filePath, 'r') as fh:
            keepLooping = True
            lineCount = 0
            while keepLooping:
                line = fh.readline()
                lineCount += 1
                if not line: break
                match = self.reSSE.search(line)
                if not match: continue
                SSE = float(match.group(1))
                pd = {}
                line = line[match.end(0):]
                while True:
                    match = self.reParam.search(line)
                    if not match:
                        line = fh.readline()
                        if line is None:
                            # EOF
                            keepLooping = False
                            break
                        line = line.strip()
                        if line: continue
                        # All done with this SSE entry
                        yield SSE, pd
                        break
                    name, svalue, stars = match.group(2, 3, 4)
                    pd[name] = svalue, stars
                    line = line[match.end(0):]
            # Now parse final population table, if any
            fh.seek(0)
            stage = 0; params = {}; SSEs = []
            N_consecutiveBlank = 0
            while True:
                line = fh.readline()
                if not line:
                    N_consecutiveBlank += 1
                    if N_consecutiveBlank > 4: break
                    continue
                line = line.strip()
                if stage == 0:
                    if line.startswith("Final population"):
                        stage = 1
                    continue
                if stage == 1:
                    if line.startswith("SSE"):
                        stage = 2
                        SSEs = line.split('|')[1].split()
                    continue
                if line.startswith('--'):
                    if params: break
                    continue
                name, values = [x.strip() for x in line.split('|')]
                params[name] = values.split()
            for k, SSE in enumerate(SSEs):
                pd = {}
                parts = [sub("{}={}", name, params[name][k]) for name in params]
                line = " ".join(parts)
                while line:
                    match = self.reParam.search(line)
                    if not match: break
                    name, svalue, stars = match.group(2, 3, 4)
                    pd[name] = svalue, stars
                    line = line[match.end(0):]
                yield float(SSE), pd

    def starredNames(self, N=4):
        """
        Looks through the log file and identifies all names that have
        stars in the I{N} lowest-SSE entries.
        """
        best = {}
        names = []
        for SSE, pd in self.filerator():
            if len(best) < N:
                best[SSE] = pd
                continue
            SSEs = sorted(best.keys())
            if SSE < SSEs[-1]:
                del best[SSEs[-1]]
                best[SSE] = pd
        for pd in best.values():
            for name in pd:
                if pd[name][1] and name not in names:
                    names.append(name)
        return sorted(names)

    def globNames(self, pattern):
        """
        Looks at the first SSE entry of the log file and identifies all
        names that match the glob I{pattern}.
        """
        for SSE, pd in self.filerator():
            return fnmatch.filter(sorted(pd.keys()), pattern)
    
    def load(self):
        """
        Loads SSEs, and parameter values and stars for those SSEs, from my
        log file.
        """
        for SSE, pd in self.filerator():
            values, stars = self.rm.add(SSE)
            for k, name in self.rm:
                vk, sk = pd[name]
                values[k] = float(vk)
                stars[k] = len(sk)
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
        for k, name in self.rm:
            parts.append(self.rj(name))
        lines = ["", (" "*self.colSep).join(parts)]
        lines.append("-" * (len(lines[-1])+2))
        # Rows
        for k, SSE in enumerate(self.rm.SSE):
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
    """
    The C{lgg} script entry point.
    """
    args = list(sys.argv[1:])
    # Parameter names don't have a dot in them, whereas the log file
    # name will
    if args and '.' in args[0]:
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

