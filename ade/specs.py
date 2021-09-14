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


class DictStacker(object):
    """
    Stacks entries and possibly sub-dictionaries within a dictionary,
    for L{Specs}.
    """
    def __init__(self, name):
        self.name = name
        self.dct = {}
    
    def add(self, value, keys):
        """
        Adds the supplied I{value} as an entry (or sub-entry) of my
        dict. Called by L{}
        """
        dct = self.dct
        while True:
            key = keys.pop(0)
            if keys:
                dct = dct.setdefault(key, {})
            else:
                dct[key] = value
                break
        
    def done(self):
        """
        Returns the name and contents of the dictionary I built.
        """
        return self.name, self.dct


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
    def dict_start(self, name):
        """
        Called by L{SpecsLoader.read} when the first hyphens-line of a
        pair is encountered.
        """
        self.ds = DictStacker(name)

    def dict_add(self, value, *keys):
        """
        Adds the supplied I{value} as an entry (or sub-entry) of a started
        dict with at least one key, more if it is a sub-entry.
        """
        self.ds.add(value, list(keys))

    def dict_done(self):
        """
        Called by L{SpecsLoader.read} when the second and last
        hyphens-line of a pair is encountered.
        """
        name, dct = self.ds.done()
        self.add(name, dct)
        del self.ds

    def add(self, name, value):
        """
        Sets my attribute I{name} to I{value}.
        """
        setattr(self, name, value)
        
    def get(self, *names, **kw):
        """
        With a single argument, returns the value of the name
        attribute. With multiple arguments, returns the named entry of
        the named entry ... of the named dict.

        Supply the the dict (and possibly sub-dict) name(s) in
        top-first order.

        Returns an empty dict if the attribute or entry doesn't exist.
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
            return {}
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

        Returns the parts of the name, or the name and C{None} if the
        entry is not for a subdict and thus no subkey is defined.
        """
        parts = []
        first = tokens.pop(0).replace("'", "")
        for part in first.split(':'):
            part = part.strip()
            if part.isdigit():
                # Dict and sub-dict keys can be integers
                part = int(part)
            parts.append(part)
        return parts
    
    def parseValue(self, value):
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
        dictDef = False
        dictName = None
        for k, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('--'):
                # Start or end of a dict definition
                if dictDef:
                    # This is the end, as we are already in
                    # dict-definition mode
                    s.dict_done()
                    dictDef = False
                    dictName = None
                elif dictName:
                    # This must be the start of a dict definition
                    s.dict_start(dictName)
                    dictDef = True
                else:
                    # Whoops, neither, and that won't work
                    raise ValueError(
                        "Dict region must be preceded by a dict name")
                continue
            tokens = line.split()
            nameParts = self.parseName(tokens)
            if not dictDef:
                if len(nameParts) > 1:
                    raise ValueError(
                        "The name:key format is only for dicts inside a dict")
                if not tokens:
                    # A standalone name is the start of a dict definition
                    dictName = nameParts[0]
                    continue
            if tokens[0] == "=":
                # An equals sign is semantic fluff
                tokens.pop(0)
            seq = []
            for token in tokens:
                if token.startswith('#'):
                    # Trailing comment
                    break
                seq.append(self.parseValue(token.rstrip(',')))
            if len(seq) == 1:
                # A single value token is its own attribute, not part
                # of a sequence
                seq = seq[0]
            if dictDef:
                # Set the entry (or sub-entry) of an ongoing dict
                s.dict_add(seq, *nameParts)
                continue
            # Set the named attribute to the value (or sequence of
            # values). We know at this point that there is but a
            # single item in nameParts
            s.add(nameParts[0], seq)
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
            
