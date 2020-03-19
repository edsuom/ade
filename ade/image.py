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
Visualizing the progress of L{de}.
"""

import os, os.path

from twisted.python.filepath import FilePath
from twisted.internet import defer, protocol, reactor, inotify

from ade.util import msg


class ImageViewer(object):
    """
    I spawn an image viewing process that shows the plot file whose
    path is specified in my constructor.

    Leaves a file with a '.pid' extension after the image filename in
    the image directory. If a previous run of this code spawned an
    image viewing process and that process PID is still running,
    does not spawn a new one.

    Complains to the messenger if no suitable image viewer is found or
    there is an error, unless the I{noComplain} constructor keyword is
    set C{True}.

    B{TODO}: Make this work on inferior operating systems. There is a
    version of I{pqiv} compiled for Windows.
    """
    notifier = None
    transport = None
    exList = [
        ["/usr/bin/pqiv", "--disable-scaling"],
        ["/usr/bin/qiv", "-Te"],
    ]
    
    class IV_Protocol(protocol.ProcessProtocol):
        def __init__(self):
            self.d = defer.Deferred()
            self.stderr = []
        def childDataReceived(self, childFD, data):
            if childFD == 2:
                self.stderr.append(data)
        def processEnded(self, reason):
            self.d.callback(self.stderr)

    @property
    def pidFilePath(self):
        return self.filePath.path + '.pid'
            
    def __init__(self, filePath, noComplain=False):
        """
        C{ImageViewer(filePath, noComplain=False)}
        """
        self.transport = None
        if not os.path.exists(filePath):
            with open(filePath, 'wb'):
                pass
        self.filePath = FilePath(filePath)
        self.noComplain = noComplain
        self._startNotifier()
        self.d = None
        reactor.addSystemEventTrigger('before', 'shutdown', self._stopNotifier)
    
    def _startNotifier(self):
        def notify(ignored, fp, mask):
            if self.transport is None:
                if fp.isfile():
                    if not self.processRunning():
                        self.spawnViewer(fp)
                        self._stopNotifier()
            elif not fp.isfile():
                self.termViewer()
                self._startNotifier()

        if self.notifier is None:
            self.notifier = inotify.INotify()
            self.notifier.startReading()
            self.notifier.watch(self.filePath, callbacks=[notify])
    
    def _stopNotifier(self):
        if self.notifier:
            self.notifier.loseConnection()
            self.notifier = None

    def processRunning(self):
        """
        Returns C{True} if there is a process running with the PID record
        in a file at my I{pidFilePath}.
        """
        if not os.path.exists(self.pidFilePath):
            return False
        with open(self.pidFilePath, 'r') as fh:
            text = fh.read().strip()
            if text:
                try: PID = int(text)
                except: return False
        # TODO: There is surely a more elegant and robust way to
        # determine whether an image viewing process is still running,
        # but this will do for now
        try: pgid = os.getsid(PID)
        except: return False
        return True
            
    def savePID(self, PID):
        """
        Saves the supplied integer I{PID} to a file that is at my
        I{pidFilePath}.
        """
        with open(self.pidFilePath, 'w') as fh:
            fh.write(str(PID))
            
    @defer.inlineCallbacks
    def spawnViewer(self, fp):
        """
        Spawns the first available image viewer specified in my I{exList}.

        "Returns" a C{Deferred} that fires with the PID of the spawned
        image viewer process.
        """
        for ex in self.exList:
            exPath = ex[0]
            if os.path.exists(exPath):
                self.ivp = self.IV_Protocol()
                args = [os.path.basename(exPath)]
                args.extend(ex[1:])
                args.append(fp.path)
                self.transport = reactor.spawnProcess(
                    self.ivp, exPath, args, env=None)
                self.savePID(self.transport.pid)
                stderr = yield self.ivp.d
                if stderr:
                    if not self.noComplain:
                        msg(0, "Error launching '{}':\n{}", args[0], stderr)
                else: break
        else:
            if not self.noComplain:
                msg(0, "No image viewer found")

    def termViewer(self):
        if self.transport is None: return
        if self.transport.pid:
            self.transport.signalProcess('TERM')
        self.transport = None
                
            
        
            
            
