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
B{A}synchronous B{D}ifferential B{E}volution.

You really just need to be concerned with two things: Constructing and
setting up a L{Population} object, and constructing and calling a
L{DifferentialEvolution} object with it.
"""

def extract_examples():
    """
    Call via the I{ade-examples} entry point to extract example files
    to a subdirectory I{ade-examples} of your home directory, creating
    the subdirectory if necessary.

    It will not overwrite existing files, so feel free to modify the
    examples. Delete a modified example file (or the whole
    subdirectory) and run this again to restore the default file.
    """
    import re
    pkg_dir = ('ade', 'examples')
    from ade.util import msg; msg(True)
    import os, os.path, shutil, pkg_resources
    sDir = pkg_resources.resource_filename(*pkg_dir)
    eDir = os.path.expanduser(
        os.path.join("~", "-".join(pkg_dir)))
    msg("Extracting {} to\n{}\n{}", " ".join(pkg_dir), eDir, "-"*79)
    if os.path.exists(eDir):
        msg("Subdirectory already exists")
    else:
        os.mkdir(eDir)
        msg("Subdirectory created")
    reFile = re.compile(r'[a-z].+\.(py|c|txt)$')
    for fileName in pkg_resources.resource_listdir(*pkg_dir):
        if not reFile.match(fileName):
            continue
        ePath = os.path.join(eDir, fileName)
        if os.path.exists(ePath):
            msg("{} already exists", ePath)
        else:
            sPath = os.path.join(sDir, fileName)
            if os.path.isfile(sPath):
                shutil.copy(sPath, ePath)
                msg("{} created", ePath)

