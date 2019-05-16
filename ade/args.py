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
#
# Unlike all other code this project, which is licensed as above, the
# code of this module args.py has been dedicated by the author into
# the public domain, still on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND.

"""
A very handy convenience class for making argparse even
easier. See the L{examples} scripts of I{ade} to see how it is used.
"""


class Args(object):
    """
    Convenience class by Edwin A. Suominen for compact and sensible
    commandline argument parsing.

    The code of this class, separate from the rest of the ade package
    and project, is dedicated to the public domain.

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

    Call the instance with a text description as a single argument to
    allow for positional arguments. The arguments will be accessed
    from the instance as sequence items.

    Call the instance with a callable to run it if the global module
    name is '__main__' (i.e., it's been called as a script) and the
    'h' option (for help) is not set.

    The instance will look exactly like an C{argparse.ArgumentParser}
    object, all set up and ready to have its attributes accessed.
    """
    def __init__(self, text):
        self.args = None
        import argparse, textwrap
        lines = text.strip().split('\n')
        paras = []; paraLines = []
        kw = {'formatter_class': argparse.RawDescriptionHelpFormatter,}
        while lines:
            line = lines.pop(0).strip()
            if line: paraLines.append(line)
            if not line or not lines:
                paras.append(textwrap.fill(" ".join(paraLines)))
                del paraLines[:]
        if paras: kw['description'] = paras.pop(0)
        if paras: kw['epilog'] = "\n\n".join(paras)
        self.parser = argparse.ArgumentParser(**kw)

    def __nonzero__(self):
        return len(self) > 0

    def addDefault(self, text, default, dest=None):
        if dest and '{}' in text: text = text.format(dest)
        if "default" not in text.lower():
            text += " [{}]".format(default)
        return text

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
            arg = args[0]
            if callable(arg):
                name = arg.__module__
                if name == '__main__' and not self.h: return arg()
                return
            self.parser.add_argument(
                '_args_', default=None, nargs='*', help=arg)
