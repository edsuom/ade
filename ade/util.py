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
Utility stuff used by most modules of L{ade}.
"""

import os, sys, re

from twisted.python import failure


def sub(proto, *args):
    """
    This really should be a built-in function.
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


class Messenger(object):
    """
    My module-level C{msg} instance writes messages to STDOUT or
    another writable object in an extremely flexible
    manner.

    My public API is mostly in L{__call__}, although L{writeLine},
    L{writeChar}, and L{lineWritten} are public and can be useful on
    their own.
    """
    N_dashes = 100
    
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
        separator. (You can also do this with a numerical argument
        immediately before or after.)
    
        Unless a single argument was supplied, returns the present
        file handle.
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


class Args(object):
    """
    Convenience class by Edwin A. Suominen for compact and sensible
    commandline argument parsing. The code of this class, separate
    from the rest of this module and package, is dedicated to the
    public domain.

    Usage: Construct an instance with a text description of your
    application. Then call the instance for each option you want to
    add, with a short letter (just a single letter) preceded by a
    single hyphen ("-"), a long option preceded by a pair of hyphens
    ("--"), a default value if the option isn't just C{store_true},
    and a text description of the option. There will be a total of 3-4
    arguments.

    You will access the option value using the short letter value,
    which gives you 26 possibilities for options (52 if you use both
    upper and lowercase). If you need more than that, you may be
    overcomplicating your command line.

    Call the instance with just one argument, a text description, to
    allow for positional arguments. The arguments will be accessed
    from the instance as sequence items.

    The instance will look exactly like an L{argparse.ArgumentParser}
    object, all set up and ready to have its attributes accessed.
    """
    def __init__(self, text):
        self.args = None
        import argparse
        lines = text.strip().split('\n')
        kw = {'description': lines[0]}
        if len(lines) > 1:
            kw['epilog'] = " ".join(lines[1:])
        self.parser = argparse.ArgumentParser(**kw)

    def __nonzero__(self):
        return any([
            bool(getattr(self.args, x))
            for x in dir(self.args) if not x.startswith('_')])

    def addDefault(self, text, default, dest=None):
        if dest and '{}' in text: text = text.format(dest)
        if "default" not in text.lower():
            text += " [{}]".format(default)
        return text
    
    def __call__(self, *args):
        if len(args) == 4:
            shortArg, longArg, default, helpText = args
            dest = shortArg[1:]
            helpText = self.addDefault(helpText, default, dest)
            self.parser.add_argument(
                shortArg, longArg, dest=dest, default=default,
                action='store', type=type(default), help=helpText)
            return
        if len(args) == 3:
            shortArg, longArg, helpText = args
            self.parser.add_argument(
                shortArg, longArg, dest=shortArg[1:],
                action='store_true', help=helpText)
            return
        if len(args) == 1:
            helpText = args[0]
            self.parser.add_argument(
                '_args_', default=None, nargs='*', help=helpText)
            return

    def __iter__(self):
        for x in getattr(self, '_args_', []):
            yield x

    def __len__(self):
        return len(getattr(self, '_args_', []))

    def __getitem__(self, k):
        return getattr(self, '_args_', [])[k]
    
    def __getattr__(self, name):
        if self.args is None:
            self.args = self.parser.parse_args()
        return getattr(self.args, name, None)
