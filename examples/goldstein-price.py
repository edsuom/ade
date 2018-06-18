#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ade:
# Asynchronous Differential Evolution.
#
# Copyright (C) 2018 by Edwin A. Suominen,
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

import os

from twisted.internet import protocol, reactor, defer


class ExecProtocol(protocol.ProcessProtocol):
    def __init__(self):
        self.running = None
        self.lock = defer.DeferredLock()
        self.lock.acquire()
        self.response = None

    def shutdown(self):
        """
        Shuts down the process.
        """
        if self.running:
            self.running = False
            try: self.transport.signalProcess('KILL')
            except: pass
            self.transport.loseConnection()
            if self.lock.locked:
                self.response = None
                self.lock.release()

    def connectionMade(self):
        """
        Upon connection to the executable process via stdio, releases the
        lock.
        """
        if self.lock.locked:
            self.lock.release()
                
    @defer.inlineCallbacks
    def cmd(self, *args):
        """
        Immediately writes a single-line command as a sequence of
        space-separated tokens to the executable via stdin, returning
        a C{Deferred} that fires with the result received by stdout.
        """
        if not args:
            raise ValueError("You must supply at least a single-word command!")
        yield self.lock.acquire()
        self.lock.release()
        self.transport.write(" ".join(args))
        yield self.lock.acquire()
        return self.response
        
    def outReceived(self, data):
        """
        Processes a line of data from the executable, likely a response.
        """
        data = data.replace('\r', '').strip('\n')
        if not data or not self.lock.locked:
            return
        self.response = data
        self.lock.release()


class Runner(object):
    """
    """
    def __init__(self, execPath):
        def startup():
            reactor.spawnProcess(
                self.ep,
                execPath, [execPath],
                env=os.environ, usePTY=False)

        self.ep = ExecProtocol()
        reactor.callWhenRunning(startup)
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)

    def shutdown(self, kill=False):
        if self.triggerID is None:
            return
        triggerID = self.triggerID
        self.triggerID = None
        self.ep.shutdown()
        reactor.removeSystemEventTrigger(triggerID)

    def eval(self, *args):
        return float(self.ep.cmd(*[float(x) for x in args]))


def main():
    r = ExecRunner('./goldstein-price')
    reactor.callWhenRunning(darwin.run)
    reactor.run()

if __name__ == '__main__' and not args.h:
    main()

