#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# pingspice:
# Object-oriented circuit construction and efficient asynchronous
# simulation with Ngspice and twisted.
#
# Copyright (C) 2017 by Edwin A. Suominen,
# http://edsuom.com/pingspice
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
Utility stuff for the B{ade} package by Edwin A. Suominen:
Asynchronous Differential Evolution.
"""

import re, sys

from twisted.python import failure

def sub(proto, *args):
    """
    This really should be a built-in function.
    """
    return proto.format(*args)

def oops(failureObj, log=None, keepGoing=False):
    def text():
        if isinstance(failureObj, failure.Failure):
            info = failureObj.getTraceback()
        else: info = str(failureObj)
        return sub("Failure:\n{}\n{}", '-'*40, info)
    if log is None:
        print(text())
    elif isinstance(failureObj, failure.Failure):
        log.failure(sub("Failure:\n{}\n", '-'*40), failureObj)
    else:
        log.failure(text())
    if not keepGoing:
        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        pdb.post_mortem(tb)
        from twisted.internet import reactor
        reactor.stop()


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


class Messenger(object):
    N_dashes = 100
    
    def __init__(self):
        self.fh = None
        self.newlineNeeded = False

    def __nonzero__(self):
        return self.fh is not None

    def dashes(self, breakBefore=False, breakAfter=False):
        text = "\n" if breakBefore else ""
        text += "-" * self.N_dashes
        if breakAfter:
            text += "\n"
        return text
    
    def writeLine(self, line):
        if self.fh is None:
            return
        if self.newlineNeeded:
            self.fh.write("\n")
            self.newlineNeeded = False
        self.fh.write(line + "\n")
        self.fh.flush()

    def writeChar(self, x):
        if self.fh is None:
            return
        self.fh.write(x)
        self.fh.flush()
        self.newlineNeeded = True
        
    def fhSet(self, arg):
        def _fhSet(fh):
            fhPrev = self.fh
            if self.fh and self.fh is not sys.stdout:
                self.fh.close()
            self.fh = fh
            return fhPrev 
        
        if hasattr(arg, 'write'):
            return _fhSet(arg)
        if isinstance(arg, bool):
            return _fhSet(sys.stdout if arg else None)
        if isinstance(arg, int):
            self.writeLine("\n"*arg)
            return
        return _fhSet(None)
    
    def __call__(self, *args, **kw):
        args = list(args)
        prefix = ""; suffix = ""
        fhPrev = self.fh
        if args:
            if hasattr(args[0], 'write') or args[0] in (None, True, False):
                fhPrev = self.fhSet(args.pop(0))
        if not args:
            if fhPrev is sys.stdout:
                return True
            return fhPrev
        if len(args) == 1:
            arg = args[0]
            text = self.dashes(True) if arg == '-' else arg
            if len(text) == 1:
                self.writeChar(text)
            else:
                self.writeLine(text)
            return self
        for repeat in range(2):
            if isinstance(args[0], int):
                if args[0] > 0:
                    prefix += ' ' * args[0]
                else:
                    prefix += '\n'
            elif args[0] == '-':
                prefix += self.dashes(True, True)
            else: break
            args.pop(0)
        N_braces = len(re.findall(r'\{[^\{]*\}', args[0]))
        while len(args) > N_braces + 1:
            if args[-1] == '-':
                suffix += self.dashes(True)
            elif args[-1] in (-1, 0):
                suffix += "\n"
            else: break;
            args.pop()
        self.writeLine(prefix + sub(*args) + suffix)
        return self
msg = Messenger()


class _Debug(object):
    debugMode = False
    @classmethod
    def _msg(cls, *args):
        if not args:
            return cls.debugMode
        if len(args) == 1:
            if isinstance(args[0], bool):
                cls.debugMode = args[0]
                return
        if cls.debugMode:
            msg(*args)
MSG = _Debug()._msg
