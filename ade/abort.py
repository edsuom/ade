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
Takes care of aborting L{Population} and L{DifferentialEvolution}
on console keystroke.

When an instance of either object is run on the command line in a
console, pressing the Enter key will cause it to quit running.

Since SIGINT (^C) doesn't work properly, it set to SIG_IGN so it's
ignored.
"""

import signal

from twisted.internet import reactor, stdio, protocol

# Since ^C doesn't work properly, let's just ignore it entirely
signal.signal(signal.SIGINT, signal.SIG_IGN)


class AbortError(Exception):
    """
    L{DifferentialEvolution.shutdown} sends one of these to the
    errback of a C{DeferredList} to ensure that it aborts.
    """


class Keyboard(protocol.Protocol):
    """
    Receives STDIN. Pressing the Enter key which will cause I{ade} to
    quit.
    """
    def __init__(self):
        """
        C{Keyboard()}
        """
        self.callbacks = []
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)
        stdio.StandardIO(self)

    def addCallback(self, func):
        """
        Adds I{func} to my list of shutdown callbacks.
        """
        if not callable(func):
            raise TypeError("Callback not callable")
        self.callbacks.append(func)
        
    def shutdown(self):
        """
        Cleanly shuts me down.
        """
        if self.triggerID is None: return
        reactor.removeSystemEventTrigger(self.triggerID)
        self.triggerID = None
        self.transport.loseConnection()
    
    def dataReceived(self, data):
        """
        Runs down the shutdown callback the first time data is received,
        e.g., the Enter key is pressed.
        """
        while self.callbacks:
            self.callbacks.pop(0)()
        self.shutdown()


KB = Keyboard()
def callOnAbort(func):
    """
    Adds C{func} to the list of callbacks to call if the Enter key is
    pressed.
    """
    KB.addCallback(func)

def shutdown():
    """
    Shuts down the L{Keyboard} protocol.
    """
    KB.shutdown()



