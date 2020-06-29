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
Unit tests for L{specs}.
"""

from ade import specs
from ade.test import testbase as tb


VERBOSE = True


class MockSpecs(object):
    def __init__(self):
        self.calls = []
    
    def add(self, name, subkey, value):
        self.calls.append(['add', name, subkey, value])


class Test_DictStacker(tb.TestCase):
    def setUp(self):
        self.ds = specs.DictStacker('foo')
    
    def test_add_one(self):
        self.ds.add(1, ['first'])
        name, dct = self.ds.done()
        self.assertEqual(name, 'foo')
        self.assertEqual(dct, {'first':1})

    def test_add_multiple(self):
        self.ds.add(1, ['first'])
        self.ds.add(2, ['second'])
        name, dct = self.ds.done()
        self.assertEqual(name, 'foo')
        self.assertEqual(dct, {'first':1, 'second':2})

    def test_add_nested(self):
        self.ds.add(0, ['zero'])
        self.ds.add(1, ['x1', 1])
        self.ds.add(2, ['x1', 2])
        self.ds.add(10, ['x10', 1])
        self.ds.add(20, ['x10', 2])
        self.ds.add(4, ['x2', 2])
        name, dct = self.ds.done()
        self.assertEqual(name, 'foo')
        self.assertEqual(dct, {
            'zero':     0,
            'x1':       {1:1, 2:2},
            'x2':       {2:4},
            'x10':      {1:10, 2:20},
            }
        )


class Test_Specs(tb.TestCase):
    def setUp(self):
        self.s = specs.Specs()
    
    def test_add(self):
        self.s.add('foo', 1)
        self.assertEqual(self.s.foo, 1)

    def test_add_dict_basic(self):
        self.s.dict_start('foo')
        self.s.dict_add(1, 'alpha')
        self.s.dict_add(2, 'bravo')
        self.s.dict_done()
        self.assertEqual(self.s.foo, {'alpha':1, 'bravo':2})

    def test_add_dict_nested(self):
        self.s.dict_start('foo')
        self.s.dict_add(1.1, 'alpha', 'first')
        self.s.dict_add(1.2, 'alpha', 'second')
        self.s.dict_add(2.1, 'bravo', 'first')
        self.s.dict_add(2.2, 'bravo', 'second')
        self.s.dict_done()
        self.assertEqual(self.s.foo, {
            'alpha': {'first':1.1, 'second':1.2},
            'bravo': {'first':2.1, 'second':2.2}})
        
    def test_get_attr(self):
        self.s.foo = 1
        self.assertEqual(self.s.get('foo'), 1)
        self.assertEqual(self.s.get('bar'), {})
                
    def test_get_dict(self):
        stuff = {'alpha':1, 'bravo':2}
        self.s.stuff = stuff
        self.assertEqual(self.s.get('stuff'), stuff)
        self.assertEqual(self.s.get('stuff', 'alpha'), 1)
        self.assertEqual(self.s.get('stuff', 'bravo'), 2)
        self.assertEqual(self.s.get('stuff', 'charlie'), {})

    def test_get_subdict(self):
        stuff = {'alpha':1, 'bravo':{'second':2, 'third':3}}
        self.s.stuff = stuff
        self.assertEqual(self.s.get('stuff'), stuff)
        self.assertEqual(self.s.get('stuff', 'alpha'), 1)
        self.assertEqual(self.s.get('stuff', 'bravo', 'second'), 2)
        self.assertEqual(self.s.get('stuff', 'bravo', 'third'), 3)


class Test_SpecsLoader(tb.TestCase):
    def setUp(self):
        filePath = tb.fileInModuleDir("test.specs")
        self.sl = specs.SpecsLoader(filePath)

    def test_get_attribute(self):
        s = self.sl()
        self.assertEqual(s.get('C19_US', 'k0'), 42)
        
    def test_parseName_dictName(self):
        self.assertEqual(self.sl.parseName(['foo']), ['foo'])
        
    def test_parseName_dictKey(self):
        tokens = ['foo:bar']
        self.assertEqual(self.sl.parseName(tokens), ['foo', 'bar'])

    def test_parseValue(self):
        self.assertIs(self.sl.parseValue("None"), None)
        self.assertTrue(self.sl.parseValue("True"))
        self.assertFalse(self.sl.parseValue("False"))
        self.assertEqual(self.sl.parseValue("3.14159"), 3.14159)

    def check(self, value, *args):
        self.assertEqual(self.s.get(*args), value)
        
    def test_call(self):
        self.s = self.sl()
        self.check([0, 1], 'first', 'alpha')
        self.check([-1.5, +1.5], 'first', 'bravo')
        self.check(3.14159, 'first', 'charlie')
        self.check([0, 10], 'second', 'alpha')
        self.check([-15, +15], 'second', 'bravo')

    def test_call_complicated(self):
        self.s = self.sl()
        self.check(['t0', -29.486, +75.451, 3.0], 'C19_US', 'relations', 'r')
