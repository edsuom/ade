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
Utility stuff used by most modules of L{ade}. Imports the author's
public-domain convenience class L{Args} from a separate module
L{args}.
"""

import os, sys, re, time

from twisted.python import failure


# Exportable import
from ade.args import Args


def sub(proto, *args):
    """
    Format string prototype I{proto} with I{args}. This really should
    be a built-in function.
    """
    try:
        return proto.format(*args)
    except:
        raise ValueError("Proto '{}' couldn't apply args {}", proto, args)

def oops(failureObj):
    """
    A handy universal errback.

    Prints the failure's error message to STDOUT and then stops
    everything so you can figure out what went wrong.
    """
    if isinstance(failureObj, failure.Failure):
        info = failureObj.getTraceback()
    else: info = str(failureObj)
    print(sub("Failure:\n{}\n{}\n", '-'*40, info))
    #import pdb; pdb.set_trace()
    #os._exit(1)


class CallbackFailureToken:
    """
    An errback can return one of these to indicate that a reporting
    callback had a fatal error.
    """


class Picklable(object):
    """
    Base class for things that can be pickled.
    """
    def __getstate__(self):
        state = {}
        dirClass = dir(self.__class__)
        if '__slots__' in dirClass:
            for name in self.__slots__:
                if hasattr(self, name):
                    state[name] = getattr(self, name)
            return state
        for name in dir(self):
            if name.startswith('_') or name in dirClass:
                continue
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        for name in state:
            setattr(self, name, state[name])

        
class Bag(object):
    """
    Use an instance of me to let functions within methods have easy
    access to stuff defined in the main method body.
    """
    #__slots__ = ['x']
    
    def __init__(self, initialValue=None):
        self.x = initialValue

    def __nonzero__(self):
        return self.x is not None

    def pop(self):
        x = self.x
        self.x = None
        return x
    
    def __call__(self, value=None):
        if value is not None:
            self.x = value
        return self.x


class EvalTimer(Picklable):
    """
    Piecewise evaluation helper.

    If your evaluation (fitness) function is made up of a number of
    separate parts that each contribute to the overall SSE, you can
    call each part via an instance of me and iterate over my instance
    to run the fastest parts first. With the I{xSSE} option enabled
    for L{DifferentialEvolution}, that may avoid the need for the
    slower parts to run as much during challenges.

    The sole constructor argument I{k} identifies the position of a
    hashable argument to L{__call__} (starting with 0 for the first
    arg following I{func}) that serves as an ID for the function part,
    uniquely identifies one flavor of a call to it for the overall
    evaluation.

    For example, consider an evaluation function that involves calling
    the same callable C{f} three times with a different integer ID as
    the first argument followed by an array of parameter values. You
    would set I{k} to 0, since the first argument is the unique ID.
    """
    def __init__(self, k):
        self.k = k
        self.times = {}
        self.Ns = {}

    def __call__(self, func, *args, **kw):
        """
        Call my instance with one part I{func} of the overall fitness
        function with its I{args} and any I{kw}.

        The I{func} must return a C{Deferred} result, and one of its
        arguments must be a unique identifying ID for the part it
        plays.

        Updates my average for elapsed time of the call, using the arg
        at my ID index I{k} as a dict key.
        """
        def done(result):
            ID = args[self.k]
            self.times[ID] = self.times.get(ID, 0) + time.time()
            self.Ns[ID] = self.Ns.get(ID, 0) + 1
            return result
        
        t0 = time.time()
        return func(*args, **kw).addBoth(done)

    def __iter__(self):
        """
        I iterate over my IDs, in ascending order of the average time it
        took their function parts to run.

        Does a sort of IDs by average time each time I iterate, which
        seems inefficient but really isn't. Sorting is very fast, and
        L{__call__} is probably being run more often than L{__iter__},
        so caching wouldn't make sense.

        My I{times} and I{Ns} dicts can update while I'm iterating
        with no problems. I will keep using the ascending order I
        computed before I started yielding IDs.
        """
        pairs = []
        for ID in self.times:
            avgTime = self.times[ID] / self.Ns[ID]
            pairs.append([ID, avgTime])
        for ID, avgTime in sorted(pairs, key=lambda x: x[1]):
            yield ID

    
class Messenger(object):
    """
    My module-level C{msg} instance writes messages to STDOUT or
    another writable object in an extremely flexible
    manner.

    My public API is mostly in L{__call__}, although L{writeLine},
    L{writeChar}, and L{lineWritten} are public and can be useful on
    their own.
    """
    N_dashes = 120
    
    def __init__(self):
        self.fh = None
        self.newlineNeeded = False
        self._lineWasWritten = False

    def __nonzero__(self):
        return self.fh is not None

    def _dashes(self, breakBefore=False, breakAfter=False):
        text = "\n" if breakBefore else ""
        text += "-" * self.N_dashes
        if breakAfter:
            text += "\n"
        return text
    
    def writeLine(self, line):
        """
        Writes the supplied I{line} of text to my current writable object I{fh},
        with a trailing newline.

        If there's been a call to L{writeChar} since the last time
        this method was called, a newline is written before I{line}.

        If I{fh} is currently C{None}, nothing is written.
        """
        if not self.fh:
            return
        self._lineWritten = True
        try:
            if self.newlineNeeded:
                self.fh.write("\n")
                self.newlineNeeded = False
            self.fh.write(line + "\n")
            self.fh.flush()
        except: self._fhSet(None)

    def writeChar(self, x):
        """
        Writes the single character I{x} to my current ratable object
        I{fh}, with no newline.

        Sets my I{newlineNeeded} flag to indicate that a newline is
        needed before the next line of text gets written via
        L{writeLine}.
        
        If I{fh} is currently C{None}, nothing is written.
        """
        if not self.fh:
            return
        try:
            self.fh.write(x)
            self.fh.flush()
            self.newlineNeeded = True
        except: self._fhSet(None)
        
    def _fhSet(self, arg):
        def fhSet(fh):
            if fh is self.fh:
                return fh
            fhPrev = self.fh
            if self.fh and self.fh is not sys.stdout:
                self.fh.close()
            self.fh = fh
            return fhPrev 
        
        if hasattr(arg, 'write'):
            return fhSet(arg)
        if isinstance(arg, bool):
            return fhSet(sys.stdout if arg else None)
        if isinstance(arg, int):
            self.writeLine("\n"*arg)
            return
        return fhSet(None)

    def lineWritten(self):
        """
        Returns C{True} if a line was written since the last time this was
        called.

        The module-level call to make is C{msg.lineWritten()}.
        """
        yes = self._lineWasWritten
        self._lineWasWritten = False
        return yes
    
    def __call__(self, *args, **kw):
        """
        A very convenient and flexible messaging method. Call at the
        module level with C{msg}.

        The module-level call to make is C{msg(...)}.
        
        Call with C{True} as the first or only argument to log to
        STDOUT, or with an open file handle to log to that. Call with
        C{False} or C{None} as the first or only argument to stop
        logging. With a single argument, returns the previous file
        handle.

        Call it with a string formatting prototype and the appropriate
        number of formatting arguments to log a line of text. (Of
        course, you can just supply the text with no formatting codes
        instead.)
    
        You can precede the text or text proto/args with C{0} or C{-1}
        to insert a blank line before the text. Similarly, you can
        follow it with C{0} or C{-1} to append a blank line after the
        text.
    
        You can precede or follow the with a single hyphen character
        ("-") to precede or follow the text with a row of hyphens as a
        separator. (You can also combine this with a numerical
        argument that comes immediately before or after.)

        Call with no arguments just to get the present file
        handle. That will always be returned unless a single argument
        was supplied, in which case the previous file handle is what
        gets returned.
        """
        args = list(args)
        prefix = ""; suffix = ""
        fh = self.fh
        if args:
            if hasattr(args[0], 'write') or args[0] in (None, True, False):
                fh = self._fhSet(args.pop(0))
        if not args:
            return fh
        if len(args) == 1:
            arg = args[0]
            if not arg:
                self.writeChar('\n')
            else:
                text = self._dashes(True) if arg == '-' else arg
                if len(text) == 1:
                    self.writeChar(text)
                else: self.writeLine(text)
            return self.fh
        for repeat in range(2):
            if isinstance(args[0], int):
                if args[0] > 0:
                    prefix += ' ' * args[0]
                else:
                    prefix += '\n'
            elif args[0] == '-':
                prefix += self._dashes(True, True)
            else: break
            args.pop(0)
        N_braces = len(re.findall(r'\{[^\{]*\}', args[0]))
        while len(args) > N_braces + 1:
            if args[-1] == '-':
                suffix += self._dashes(True)
            elif args[-1] in (-1, 0):
                suffix += "\n"
            else: break;
            args.pop()
        self.writeLine(prefix + sub(*args) + suffix)
        return fh
msg = Messenger()
