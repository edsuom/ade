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
Example script I{goldstein-price.py}: The Goldstein-Price
test function.

Runs as many subordinate python processes as there are CPU cores to
solve the Goldstein-Price test function using asynchronous
differential evolution. This will happen very fast on a modern
multicore CPU!

You need to compile the U{C code<http://edsuom.com/ade/ade.examples.goldstein-price.c>} first by
running the following command:

C{gcc -Wall -o goldstein-price goldstein-price.c}

It's worth looking through the API docs for the objects listed below
(and of course the
U{Python<http://edsuom.com/ade/ade.examples.goldstein-price.py>} and
U{C<http://edsuom.com/ade/ade.examples.goldstein-price.c>} source) to
familiarize yourself with a clear and simple usage of I{ade}.
"""

import os, time

import numpy as np

from twisted.python import failure
from twisted.internet import protocol, reactor, defer

from ade.util import oops, msg
from ade.population import Population
from ade.de import DifferentialEvolution


class ExecProtocol(protocol.ProcessProtocol):
    """
    I am a U{Twisted<http://twistedmatrix.com>} process protocol for
    communicating with a process running an executable you've
    compiled, via STDIO.
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

        If the line isn't blank and someone is waiting for it by
        having acquired my C{DeferredLock}, writes the data to my
        I{response} attribute and releases the lock.
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
        """
        Shuts down my executable.
        """
        if self.triggerID is None:
            return
        triggerID = self.triggerID
        self.triggerID = None
        self.ep.shutdown()
        reactor.removeSystemEventTrigger(triggerID)

    def __call__(self, *args):
        """
        Sends the args to my executable via STDIN, returning a C{Deferred}
        that fires with the result when it arrives via STDOUT.
        """
        return self.ep.cmd(*args).addErrback(oops)


class MultiRunner(object):
    """
    I manage a pool of L{Runner} instances.
    """
    def __init__(self, execPath, N=None):
        """
        C{MultiRunner}(execPath, N=None)
        """
        if N is None:
            import multiprocessing as mp
            N = mp.cpu_count()
        self.dq = defer.DeferredQueue(N)
        for k in range(N):
            r = Runner(execPath)
            self.dq.put(r)

    def shutdown(self):
        """
        Shuts down each L{Runner}, returning a C{Deferred} that fires when
        they've all stopped.
        """
        dList = []
        for k in range(self.dq.size):
            d = self.dq.get().addCallback(lambda r: r.shutdown())
            dList.append(d)
        return defer.DeferredList(dList)
            
    def __call__(self, values):
        """
        Call my instance with a 2-sequence of I{values} and I will call
        the next free L{Runner} in my pool with it. Returns a
        C{Deferred} that fires with the result when it arrives.
        """
        def done(r):
            result = r(*values)
            self.dq.put(r)
            return result
        return self.dq.get().addCallbacks(done, oops)


class Solver(object):
    """
    I solve a compiled test equation using asynchronous differential
    evolution.

    @cvar N: The number of parallel processes to run. Leave it at
        C{None}, the default, to have the number set to however many
        CPU cores you have.
    """
    N = None
    
    def __init__(self):
        """
        C{Solver}()
        """
        self.mr = MultiRunner(self.execPath, self.N)
        self.p = Population(self.mr, self.names, self.bounds, popsize=20)
        self.p.reporter.minDiff = 0.0001
    
    def report(self, values, counter, SSE):
        """
        Prints a one-line message about a new best candidate's values and
        SSE.
        """
        msg(0, self.p.pm.prettyValues(values, "SSE={:.5f} with", SSE), 0)
        
    @defer.inlineCallbacks
    def __call__(self):
        """
        Call my instance to set up a L{Population} full of L{Individual}s,
        establish a reporting callback, and run asynchronous
        L{DifferentialEvolution}.
        """
        t0 = time.time()
        yield self.p.setup()
        self.p.addCallback(self.report)
        de = DifferentialEvolution(self.p)
        yield de()
        yield self.mr.shutdown()
        msg(0, "Final population:\n{}", self.p)
        msg(0, "Elapsed time: {:.2f} seconds", time.time()-t0, 0)
        reactor.stop()


class GoldSteinPrice_Solver(Solver):
    """
    I solve the Goldstein-Price test equation.

    You need to compile the U{C
    code<http://edsuom.com/ade/ade.examples.goldstein-price.c>} first
    by running the following command:

    C{gcc -Wall -o goldstein-price goldstein-price.c}
    """
    execPath = "./goldstein-price"
    names = ["x", "y"]
    bounds = [(-2, +2), (-2, +2)]
    
    
def main():
    """
    Called when this module is run as a script.
    """
    msg(True)
    r = GoldSteinPrice_Solver()
    reactor.callWhenRunning(r)
    reactor.run()


if __name__ == '__main__':
    main()

