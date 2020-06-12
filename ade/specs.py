#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018-20 by Edwin A. Suominen,
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
For defining specifications of an individual, including but not
limited to parameter ranges.
"""

import os.path, re

from ade.util import *


class DictStack(object):
    """
    Dictionaries within a dictionary, for L{Specs}.
    """
    def __init__(self, s):
        self.s = s
        self.names = []

    def __nonzero__(self):
        """
        I am C{True} if a dict is in progress, i.e., I've been started
        since construction or a call to L{done}.
        """
        return bool(self.names)
        
    def add(self, name):
        """
        Call to add a new dictionary or sub-dictionary with I{name}.
        
        The first time this is called after construction or a call to
        L{done}, starts a new dictionary to hold sub-dictionaries and
        records I{name} as its name. Thereafter, before L{done} is
        called, adds a new sub-dictionary to the dictionary, keyed by
        I{name}.

        I will hold onto the dictionary name, or sub-dictionary key,
        as my I{currentName}.
        """
        if name.isdigit(): name = int(name)
        self.currentName = name
        if name in self.names:
            return
        if self.names:
            self.dct[name] = {}
        else: self.dct = {}
        self.names.append(name)

    def entry(self, key, value):
        """
        Sets a dictionary or sub-dictionary entry.
        
        If L{add} has only been called once since construction or the
        last L{done}, sets an entry under I{key} of my dictionary to
        I{value}. Otherwise, sets an entry to the sub-dictionary that
        I{currentName} points to, under I{key} with I{value}.
        """
        if key.isdigit(): key = int(key)
        if len(self.names) > 1:
            self.dct[self.currentName][key] = value
            return
        self.dct[key] = value
        
    def done(self):
        """
        Sets my dictionary as an attribute of my parent L{Specs} object
        I{s}, using the dictionary's name as the attribute name.

        Clears things out for another call or series of calls to
        L{add} and L{entry}.
        """
        if not self.dct:
            # TODO: Allow for empty dicts with nothing but perhaps a
            # comment between two lines of hyphens
            return
        setattr(self.s, self.names[0], self.dct)
        self.names = []
        del self.dct


class Specs(object):
    """
    I define the specifications of an individual, including but not
    limited to parameter ranges.

    A single attribute value is defined with the construct C{name =
    value} or C{name value}.

    A dict of values is defined with just the dict C{name} on a line
    by itself followed by a line of hyphens, e.g., C{-----}. Then,
    after the hyphens, one or more C{key value} lines. (Zero lines
    should make an empty dict, but is not supported.) Then another
    line of hyphens.

    But dicts can be stacked! If a I{key} has two parts separated by a
    colon, the first part is an entry in the top-level dict that is
    actually another dict, and the second part is an entry in I{that}
    dict. For example,::

        params
        -------------------------------------------
        351:Rg        990
        351:ttrig     2E-4
        351:Vbatt     62.7
        
        361:Rg        82000
        361:ttrig     2E-4
        361:Vbatt     63.343
        
        371:ttrig     2E-3
        371:Vbatt     63.42
        -------------------------------------------

    There are three sub-dicts inside I{params}, accessible with the
    keys I{351}, I{361}, and I{371}.
    """
    def __init__(self):
        self.ds = DictStack(self)

    @property
    def dict_underway(self):
        """
        Property: C{True} if a dict is under construction.
        """
        return self.ds
        
    def dict_start(self, name):
        """
        Public API for L{DictStack.add}.
        """
        self.ds.add(name)
        
    def add(self, name, subkey, value):
        """
        Adds something to my instance. There are three possibilities.
        
            1. If I{subkey} is C{None} and I have no L{DictStack} in
               progress, adds an attribute I{name} with I{value} to my
               L{Specs} object.

            2. If I{subkey} is C{None} and I have a L{DictStack} in
               progress, adds an entry to its top level, referenced by
               I{name} with the I{value}.

            3. With a I{subkey}, adds a new sub-dict to my
               L{DictStack} (one must be in progress) and an entry to
               that, referenced by I{key} with the I{value}.
        """
        if subkey:
            # Possibility #3
            self.ds.add(name)
            self.ds.entry(subkey, value)
            return
        if self.ds:
            # Possibility #2
            self.ds.entry(name, value)
            return
        # Possibility #1
        setattr(self, name, value)

    def dict_next(self):
        """
        Call whenever the dashed lines indictating a dict definition are
        encountered.

        Calling while there's nothing but a dict name established will
        do nothing. Calling with a dict under construction will
        finalize it.
        """
        if self.dict_underway: self.ds.done()
        
    def get(self, *names, **kw):
        """
        With a single argument, returns the value of the name
        attribute. With multiple arguments, returns the named entry of
        the named entry ... of the named dict.

        Supply the the dict (and possibly sub-dict) name(s) in
        top-first order.

        Returns a 0 entry if it doesn't exist, or the value of
        I{default} if that keyword is set to something.

        An empty top-level dict is returned if only that is requested
        (no entry keys specified), even if it doesn't exist.
        """
        names = list(names)
        first = names.pop(0)
        if 'dct' in kw:
            dct = kw['dct']
            if first in dct:
                obj = dct[first]
                if names and isinstance(obj, dict):
                    kw['dct'] = obj
                    return self.get(*names, **kw)
                return obj
            return kw.get('default', 0)
        obj = getattr(self, first, {})
        if names and isinstance(obj, dict):
            kw['dct'] = obj
            return self.get(*names, **kw)
        return obj
    
        
class SpecsLoader(object):
    """
    I load and parse a text file that defines the specifications of an
    individual, including but not limited to parameter ranges. Call my
    instance to get a fully populated instance of L{Specs}.

    Construct me with an instance of L{Setups} and call my instance
    to obtain a L{Specs} object with the specifications as its
    attributes.
    
    Comments (lines beginning with '#', and trailing parts of a line
    after a '#') and blank lines are ignored.
    """
    reNum = re.compile(r'[\+\-]?[0-9]+(\.[0-9]+)?([e][\+\-]?[0-9]+)?$')
    
    def __init__(self, filePath):
        self.filePath = filePath

    def parseName(self, tokens):
        """
        I call this when I encounter a line defining a new entry, with the
        first and possibly only token being the entry's sub-dict key
        ("subkey").

        Returns the name and subkey, or the name and C{None} if the
        entry is not for a subdict and thus no subkey is defined.
        """
        first = tokens.pop(0).replace("'", "")
        parts = [x.strip() for x in first.split(':')]
        if len(parts) == 1:
            return parts[0], None
        if len(parts) > 2:
            raise ValueError(
                "A name can only be by itself or with one key")
        return parts
    
    def parse(self, value):
        """
        I call this when I encounter a line defining just one or more
        space-delimited value, once per value.
        """
        value = value.lower()
        for string, assignedValue in (
                ('none', None), ('true', True), ('false', False)):
            if value == string:
                return assignedValue
        match = self.reNum.match(value)
        if match:
            return float(value)
        return value

    def read(self, fh, s):
        """
        Reads and parses specs from the supplied open file handle I{fh},
        populating the supplied L{Specs} object I{s}.

        Modifies I{s} in place. Returns the number of lines read.
        """
        for k, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('--'):
                # Start or end of a dict definition
                s.dict_next()
                continue
            tokens = line.split()
            name, subkey = self.parseName(tokens)
            if subkey and not s.dict_underway:
                raise ValueError(
                    "The name:key format is only for dicts inside a dict")
            if not tokens:
                # A standalone name is the start of a dict definition
                s.dict_start(name)
                continue
            if tokens[0] == "=":
                # An equals sign is semantic fluff
                tokens.pop(0)
            seq = []
            for token in tokens:
                if token.startswith('#'):
                    # Trailing comment
                    break
                seq.append(self.parse(token.rstrip(',')))
            if len(seq) == 1:
                # A single token is its own attribute, not part of a
                # sequence
                seq = seq[0]
            # Set the named attribute the value (or sequence of values) as the 
            s.add(name, subkey, seq)
        return k+1
        
    def __call__(self):
        """
        Loads and parses specs from the specified I{filePath},
        constructing, populating, and returning a L{Specs} object.
        """
        s = Specs()
        if os.path.exists(self.filePath):
            with open(self.filePath) as fh:
                N = self.read(fh, s)
                msg("Read {:d} lines from {}", N, self.filePath)
        else: raise OSError(sub("File '{}' not found", self.filePath))
        return s
            
