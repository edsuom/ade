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

"""
Example script for the I{ade} package: goldstein-price.py

Runs as many subordinate python processes as there are CPU cores to
solve the Goldstein-Price test function using asynchronous
differential evolution.

This will happen very fast on a modern multicore CPU!

You need to compile the C code first by running the following command:

C{gcc -Wall -o goldstein-price goldstein-price.c}

"""

import os

import numpy as np

from twisted.python import failure
from twisted.internet import protocol, reactor, defer

from ade.util import oops, msg
from ade.population import Population
from ade.de import DifferentialEvolution


class ExecProtocol(protocol.ProcessProtocol):
    """
    Process protocol for communicating with a process running the
    goldstein-price executable you compiled, via STDIO.
    """
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
        initial lock.
        """
        if self.lock.locked:
            self.lock.release()
                
    @defer.inlineCallbacks
    def cmd(self, *args):
        """
        Sends function arguments to the executable via stdin, returning
        a C{Deferred} that fires with the result received via
        stdout, with numeric conversions.
        """
        yield self.lock.acquire()
        self.transport.write(" ".join([str(x) for x in args]) + '\n')
        yield self.lock.acquire()
        self.lock.release()
        defer.returnValue(float(self.response))
        
    def outReceived(self, data):
        """
        Processes a line of response data from the executable.
        """
        data = data.replace('\r', '').strip('\n')
        if not data or not self.lock.locked:
            return
        self.response = data
        self.lock.release()


class Runner(object):
    """
    I manage one executable.
    """
    def __init__(self, execPath):
        def startup():
            reactor.spawnProcess(
                self.ep,
                'stdbuf', ['stdbuf', '-oL', execPath],
                env=os.environ, usePTY=False)
        
        self.ep = ExecProtocol()
        reactor.callWhenRunning(startup)
        self.triggerID = reactor.addSystemEventTrigger(
            'before', 'shutdown', self.shutdown)

    def shutdown(self):
        if self.triggerID is None:
            return
        triggerID = self.triggerID
        self.triggerID = None
        self.ep.shutdown()
        reactor.removeSystemEventTrigger(triggerID)

    def __call__(self, *args):
        return self.ep.cmd(*args).addErrback(oops)


class MultiRunner(object):
    """
    I manage a pool of Runners.
    """
    def __init__(self, execPath, N=None):
        if N is None:
            import multiprocessing as mp
            N = mp.cpu_count()
        self.dq = defer.DeferredQueue(N)
        for k in range(N):
            r = Runner(execPath)
            self.dq.put(r)

    def shutdown(self):
        dList = []
        for k in range(self.dq.size):
            d = self.dq.get().addCallback(lambda r: r.shutdown())
            dList.append(d)
        return defer.DeferredList(dList)
            
    def __call__(self, values, SSE=None):
        def done(r):
            result = r(*values)
            self.dq.put(r)
            return result
        return self.dq.get().addCallbacks(done, oops)


class Solver(object):
    def __init__(self, N=None):
        self.mr = MultiRunner(self.execPath, N)
        self.p = Population(self.mr, self.names, self.bounds)
        self.p.reporter.minDiff = 0.0001

    def report(self, values, counter):
        def gotSSE(SSE):
            msg(0, self.p.pm.prettyValues(values, "SSE={:.5f} with", SSE), 0)
        return self.mr(values).addCallbacks(gotSSE, oops)
        
    @defer.inlineCallbacks
    def __call__(self):
        yield self.p.setup()
        self.p.addCallback(self.report)
        de = DifferentialEvolution(self.p)
        yield de()
        yield self.mr.shutdown()
        msg(0, "Final population:\n{}", self.p)
        reactor.stop()


class GoldSteinPrice_Solver(Solver):
    execPath = "./goldstein-price"
    names = ["x", "y"]
    bounds = [(-2, +2), (-2, +2)]
    
    
def main():
    msg(True)
    r = GoldSteinPrice_Solver()
    reactor.callWhenRunning(r)
    reactor.run()

if __name__ == '__main__':
    main()

