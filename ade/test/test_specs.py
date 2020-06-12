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


class Test_DictStack(tb.TestCase):
    def setUp(self):
        self.s = MockSpecs()
        self.ds = specs.DictStack(self.s)
    
    def test_add_one(self):
        self.assertFalse(self.ds)
        self.ds.add('first')
        self.assertTrue(self.ds)
        self.ds.entry('alpha', 1)
        self.ds.entry('bravo', [-1, +1])
        self.ds.done()
        self.assertEqual(self.s.first, {'alpha':1, 'bravo':[-1, +1]})
        self.assertFalse(self.ds)

    def test_add_multiple(self):
        self.ds.add('first')
        self.ds.entry('alpha', 1)
        self.ds.entry('bravo', [-1, +1])
        self.ds.done()
        self.ds.add('second')
        self.ds.entry('alpha', 10)
        self.ds.done()
        self.assertEqual(self.s.first, {'alpha':1, 'bravo':[-1, +1]})
        self.assertEqual(self.s.second, {'alpha':10,})

    def test_add_nested(self):
        self.ds.add('whitman')
        self.ds.entry('multitudes', 1e6)
        self.ds.add('first')
        self.ds.entry('alpha', 1)
        self.ds.entry('bravo', 2)
        self.ds.add('second')
        self.ds.entry('charlie', 3)
        self.ds.done()
        self.assertEqual(self.s.whitman, {
            'multitudes': 1e6,
            'first': {'alpha': 1, 'bravo': 2}, 'second': {'charlie':3}})


class Test_Specs(tb.TestCase):
    def setUp(self):
        self.s = specs.Specs()
    
    def test_add_attr(self):
        self.s.add('foo', None, 1)
        self.assertEqual(self.s.foo, 1)

    def test_add_dict(self):
        self.assertFalse(self.s.dict_underway)
        self.s.dict_start('bar')
        self.assertTrue(self.s.dict_underway)
        self.s.dict_next() # Should have no effect
        self.assertTrue(self.s.dict_underway)
        self.s.add('alpha', None, 1)
        self.s.add('bravo', None, 2)
        self.assertTrue(self.s.dict_underway)
        self.s.dict_next()
        self.assertFalse(self.s.dict_underway)
        self.assertEqual(self.s.bar, {'alpha':1, 'bravo':2})

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
        self.assertEqual(self.s.get('stuff', 'charlie'), 0)

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

    def test_parseName_dictName(self):
        self.assertEqual(self.sl.parseName(['foo']), ('foo', None))
        
    def test_parseName_dictKey(self):
        tokens = ['foo:bar']
        self.assertEqual(self.sl.parseName(tokens), ['foo', 'bar'])

    def test_parse(self):
        self.assertIs(self.sl.parse("None"), None)
        self.assertTrue(self.sl.parse("True"))
        self.assertFalse(self.sl.parse("False"))
        self.assertEqual(self.sl.parse("3.14159"), 3.14159)

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
        self.check(['t0', -29.486, +75.451, 3.0], 'C19_', 'relations', 'r')
