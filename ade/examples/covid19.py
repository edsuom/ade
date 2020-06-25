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
Example script covid19.py: Identifying coefficients for a logistic
growth model (with multiple growth regimes) of the number of reported
cases of Covid-19 in the U.S. vs time in days since the first case.

This example reads a 6-field BZIP2-compressed CSV file
"ade-covid19.csv.bz2" whose original you can obtain from The New York
Times as "us-counties.csv." Each line contains:

    1. The date, YYYY-MM-DD format,

    2. U.S. county,

    3. U.S. state,

    4. FIPS identifier (5-digit integer),

    5. Number of reported cases,

    6. Number of reported deaths.

Use this example script covid19.py to see a graphical representation
of the number of cases in your county or state, or the entire U.S.,
along with a projection using a six-parameter "logistic-growth with
growth-regime transition" model. Here's how:

1. Clone the "ade" repo, and update it to make sure you have the
   latest commit,

2. Do "pip install -e ." from within the top-level 'ade' directory
   of the repo,

3. Run the command "ade-examples" to create a directory "~/ade-examples".

4. Do the command line "./covid19.py <state> <county>" or
   "./covid19.py <state>" inside the ~/ade-examples directory. You'll
   want to have the "pqiv" program on your computer.  It should be as
   simple as doing "apt install pqiv". If you want to include
   everything in the U.S., just do "./covid19.py" with no
   arguments. In any case, there are a number of command-line options
   you can use.

This works on Linux; you'll probably know how to do the equivalent in
lesser operating systems. The code might even run.

The latest copy of the "ade-covid19.csv.bz2" file available on the
server is downloaded for this example per the generous terms of the
New York Times for its public database. Here's the LICENSE file
included with the original "us-counties.csv" file (unaltered except
for BZIP2 compression):

    "Copyright 2020 by The New York Times Company"
    
    "In light of the current public health emergency, The New York
    Times Company is providing this database under the following
    free-of-cost, perpetual, non-exclusive license. Anyone may copy,
    distribute, and display the database, or any part thereof, and
    make derivative works based on it, provided (a) any such use is
    for non-commercial purposes only and (b) credit is given to The
    New York Times in any public display of the database, in any
    publication derived in part or in full from the database, and in
    any other public use of the data contained in or derived from the
    database."
      
    "By accessing or copying any part of the database, the user
    accepts the terms of this license. Anyone seeking to use the
    database for other purposes is required to contact The New York
    Times Company at covid-data@nytimes.com to obtain permission."
    
    "The New York Times has made every effort to ensure the accuracy
    of the information. However, the database may contain typographic
    errors or inaccuracies and may not be complete or current at any
    given time. Licensees further agree to assume all liability for
    any claims that may arise from or relate in any way to their use
    of the database and to hold The New York Times Company harmless
    from any such claims."

If you'd like to clone the New York Times database for yourself, the
URL is https://github.com/nytimes/covid-19-data.git. Otherwise, just
delete the copy of ade-covid19.csv.bz2 in your ~/ade-examples
directory and the latest version on the author's server will be
downloaded again.

Let me say that I have been a subscriber to the Times for at least
five years and will be for years to come. It's a fairly pricey
subscription, but the outstanding journalism they do (including the
work to assemble this database) is worth it. I was using the Johns
Hopkins University database but got annoyed at how they continually
kept adding more and more restrictive license terms while also
imposing a two-day delay in the results in their public github
repo. Fine, JHU, I get it ... you want your Coronavirus dashboard to
be the place people go to look, not other people's work based on the
data you compile. At least that's the impression I got.

A few people may come across this source file out of their own
interest in and concern about the COVID-19 coronavirus. I hope this
example of my open-source evolutionary optimization software of mine
gives them some insights about the situation.

BUT PLEASE NOTE THIS CRITICAL DISCLAIMER: First, I disclaim everything
that the New York Times does. I'm pretty sure their lawyers had good
reason for putting that stuff in there, so I'm going to repeat
it. Except think "Ed Suominen" when you are reading "The New York
Times":

    The New York Times has made every effort to ensure the accuracy of
    the information. However, the database may contain typographic
    errors or inaccuracies and may not be complete or current at any
    given time. Licensees further agree to assume all liability for
    any claims that may arise from or relate in any way to their use
    of the database and to hold The New York Times Company harmless
    from any such claims.

Second, I know very little about biology, beyond a layman's
fascination with it and the way everything evolved. (Including this
virus!) I do have some experience with modeling, including using this
ADE package to develop some really cool power semiconductor simulation
software that I'll be releasing in a month or so from when I'm doing
the GitHub commit with this COVID-19 example. The software (also to be
free and open-source!) has a sophisticated subcircuit model for power
MOSFETs that evolves 40+ parameters (an unfathomably huge search
space). And it does so with this very package whose example you are
now reading.

The model I'm using for the number of reported cases of COVID-19
follows the logistic growth model, with a small (and not terribly
significant) linear term added. It has just 4 parameters, and finding
the best combination of those parameters is no problem at all for
ADE.

So, YES, THIS IS STILL A DISCLAIMER, I am not an expert in any of the
actual realms of medicine, biology, etc. that we rely on for telling
us what's going on with this virus. I just know how to fit models to
data, in this case a model that is well understood to apply to
biological populations.

Don't even THINK of relying on this analysis or the results of it for
any substantive action. If you really find it that important, then
investigate for yourself the math, programming, and the theory behind
my use of the math for this situation. Run the code, play with it,
critique it, consider how well the model does or does not
apply. Consider whether the limiting part of the curve might occur
more drastically or sooner, thus making this not as big a deal. Listen
to experts and the very real reasoning they may have for their own
projections about how bad this could get.

It's on you. I neither can nor will take any responsibility for what
you do. OK, ENOUGH DISCLAIMER.

This example uses asynchronous differential evolution to efficiently
find a population of best-fit combinations of parameters for a
first-order differential equation modeling the number of new Covid-19
cases per day.

The model is compared to data for daily numbers of new reported
COVID-19 cases, where I{xd} is the number of new cases expected to be
reported each day, I{t} is the time since the first observation, in
days, and I{x} is the number of total reported cases at time I{t}.

With the exception of the residual subplot in the middle, the plot
shows cumulative numbers of total cases C{x(t)}, not the differential
model function C{xd(t, x)} that is fitted to the number of daily
B{new} cases. To produce these plots and the predictions that we are
all interested in, the plots are anchored to the most recently known
value of the number of reported cases. Then I{xd(t, x)} is integrated
in both directions from the date for that most recent known value.

The integration goes backwards to show fit between the model and
historical numbers of total cases, which can be expected to get worse
in percentage terms as the known point gets farther away but probably
not so bad in absolute terms as both modeled and historical numbers
shrink rapidly.

For extrapolations from the most recent known date, the integration is
done forwards from that point. Again, precision will get worse as the
model ventures farther into the future. Extrapolation is dangerous
even for models with a solid theoretical basis (and this one was
produced by a biology+medicine layman), so see the disclaimer.
"""

import re, random
from datetime import date, timedelta

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from twisted.python import failure
from twisted.internet import reactor, defer, task

from asynqueue import ThreadQueue, ProcessQueue, DeferredTracker
from yampex.plot import Plotter

from ade.constraints import RelationsChecker
from ade.population import Population
from ade.de import DifferentialEvolution
from ade.image import ImageViewer
from ade.abort import abortNow
from ade.specs import SpecsLoader
from ade.util import *


from data import Data


DISCLAIMER = """
Dedicated to the public domain by Edwin A. Suominen. My
only relevant expertise is in fitting nonlinear models
to data, not biology or medicine. See detailed
disclaimer in source file covid19.py.
"""

def oops(failureObj):
    """
    A handy universal errback.

    Prints the failure's error message to STDOUT and then stops
    everything so you can figure out what went wrong.
    """
    if isinstance(failureObj, failure.Failure):
        info = failureObj.getTraceback()
    else: info = str(failureObj)
    print(sub("Failure:\n{}\n{}\n", '-'*40, info))
    abortNow()
    os._exit(1)

def sanitized(*args):
    """
    Returns a sanitized string uniquely determined by the supplied
    args, any of which can be a string or C{None}.

    C{None} args are ignored. Any spaces within an arg are
    removed. The text of multiple args is joined with an
    underscore. For example, supplying "Santa Fe" and "New Mexico" as
    args results in "SantaFe_NewMexico".
    """
    parts = []
    for arg in args:
        if arg is None:
            continue
        parts.append(arg.replace(" ", ""))
    return "_".join(parts)
    

class Covid19Data(Data):
    """
    Run L{setup} on my instance to load the ade-covid19.csv.bz2 file
    from the EAS server, a compressed version of the original
    us-counties.csv provided by the New York Times.

    @cvar county: Set this to a U.S. county name if only a particular
        county is of interest. (You'll also want to set I{state} to
        avoid adding up numbers from the counties with the same name
        in multiple states.)

    @cvar state: Set this to a U.S. state name if only a particular
        state is of interest. Leaving both this and the county name
        blank selects everything, i.e., the whole U.S. is of interest.

    @cvar relations: Set this to a dict of dicts that defines a linear
        relationship between two parameters and the maximum deviation
        from such a relationship that will be accepted in
        parameters. See L{ade.constraints.RelationsChecker}.

    @ivar t:
    
    @see: The L{Data} base class.
    """
    basename = "covid19"
    reDate = re.compile(r'(20[0-9]{2})-([0-9]{2})-([0-9]{2})')

    county = None
    state = None

    def __len__(self):
        return len(self.dates)

    def includeRow(self, row):
        """
        Returns C{True} if the supplies I{row} of the CSV file 
        matches my subclass's I{county} and I{state}.
        """
        if self.county and row[1] != self.county:
            return
        if self.state and row[2] != self.state:
            return
        return True
        
    def build_t(self, result):
        """
        Builds a list of the unique I{dates} from the first field in each
        row from the I{result} list of rows from the CSV file.

        Then builds an array I{t} of times in days (float) from the
        first date.

        Returns a dict of sets I{xref} keyed by date. Each set
        contains the row indices of I{result} that contain cases
        numbers for that date.
        """
        def g2i(k):
            return int(m.group(k))

        def hours(dateObj):
            seconds = (dateObj - firstDate).total_seconds()
            return seconds / 3600

        xref = {}
        firstDate = None
        self.dates = set()
        for k, row in enumerate(result):
            if k == 0: continue
            dateText = row[0]
            m = self.reDate.match(dateText)
            if not m:
                raise ValueError(sub(
                    "Couldn't parse '{}' as a date!", dateText))
            thisDate = date(g2i(1), g2i(2), g2i(3))
            if firstDate is None:
                firstDate = thisDate
            self.dates.add(thisDate)
            if thisDate not in xref:
                xref[thisDate] = {k}
            else: xref[thisDate].add(k)
        # Convert set to sorted list
        self.dates = sorted(list(self.dates))
        # Build array t
        self.t = np.array([hours(x)/24 for x in self.dates], dtype=float)
        return xref
                
    def build_X(self, xref, result):
        """
        Builds an array of the total cases on each day represented by the
        corresponding items in I{dates} and I{t}.

        This can take a while if you haven't set me up to filter out
        the vast majority of the rows with a I{state} and perhaps also
        a I{county} attribute.
        """
        self.X = np.zeros_like(self.t)
        for kt, date in enumerate(self.dates):
            for kr in xref[date]:
                row = result[kr]
                self.X[kt] += int(row[4])
        
    def parseValues(self, result, daysAgo):
        """
        Parses the date and reported case numbers from the lists of
        text-values in the I{result}, limiting the latest data to the
        specified number of I{daysAgo}.

        To not limit latest data, set I{daysAgo} to zero.
        """
        xref = self.build_t(result)
        self.build_X(xref, result)
        if daysAgo:
            for name in ('t', 'dates', 'X'):
                setattr(self, name, getattr(self, name)[:-daysAgo])
                   
    def setup(self, daysAgo=0):
        """
        Calling this gets you a C{Deferred} that fires when setup is done
        and my I{t} and I{X} ivars are ready.

        @keyword daysAgo: Limit latest data to this many days ago
            (integer) rather than up to today (0 days ago).
        """
        d = self.load()
        d.addCallback(self.parseValues, daysAgo)
        d.addErrback(oops)
        return d


class Results(object):
    """
    I am a simple container to hold the results of modeling with one
    set of parameters.
    """
    __slots__ = ['t', 'XD', 'X', 'R', 'SSE']

    def __init__(self, t0, t1):
        self.t = np.arange(t0, t1, np.sign(t1-t0))

    def flip(self):
        for name in self.__slots__:
            Z = getattr(self, name, None)
            if Z is None: continue
            setattr(self, name, np.flip(Z))


class Model(object):
    """
    I define a model for the number of new daily reported cases for
    each day after the date of the region's first reported case. The
    model is a logistic growth model modified to have multiple growth
    regimes and a smooth transition between them.

    The conventional logistic growth model is the logistic (Verhulst)
    model: "A biological population with plenty of food, space to
    grow, and no threat from predators, tends to grow at a rate that
    is proportional to the population....Of course, most populations
    are constrained by limitations on resources--even in the short
    run--and none is unconstrained forever."
    U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html}

    It requires the integration of a first-order differential
    equation. Here is the expression in the original logistic growth
    model for the number of new cases per day:::

        xd(t,x) = x*r(t)*(1-x/L)

    The author of this code (EAS) proposes a major modification to
    that model: providing for multiple growth regimes with a smooth
    transition between them. That is reflected by I{r(t)} being a
    function of I{t} rather than a constant.

    @ivar N: The number of parameters in the model.
    
    """
    # Logistic growth with multiple growth regimes
    # Wide-open default parameter bounds
    names = [
        # --- Regime #1 ---------------------------------------------------
        # Upper limit to number of total cases (<= population)
        'L',
        # Constant number of new cases reported each day since beginning
        'b',
        # r1: Initial daily growth rate (0.1 = 10% per day)
        'r1',
        # --- Regime #2 ---------------------------------------------------
        # t1: Midpoint between regime #1, #2 (days after first case)
        't1',
        # s1: Span of regime #1 (days between regime halfway points)
        's1',
        # r2: Second daily growth rate
        'r2',
        # --- Regime #3 ---------------------------------------------------
        # t2: Midpoint between regime #2, #3 (days after first case)
        't2',
        # s2: Span of regime #2 (days between regime halfway points)
        's2',
        # r3: Third daily growth rate
        'r3',
        # -----------------------------------------------------------------
    ]

    #--- Logistic Growth Model with Multiple Growth Regimes -------------------

    def __init__(self, names):
        OK = False
        names = set(names)
        if {'L', 'b', 'r1'} <= names:
            N = 3
            OK = True
        if {'t1', 's1', 'r2'} <= names:
            if N == 3:
                N = 6
            else: OK = False
        if {'t2', 's2', 'r3'} <= names:
            if N == 6:
                N = 9
            else: OK = False
        if len(names) > N: OK = False
        if not OK:
            raise ValueError(sub(
                "Parameters {} not a valid combination!", ', '.join(names)))
        self.r = self.r_3 if N==3 else self.r_6 if N==6 else self.r_9
        self.N = N

    @property
    def f_text(self):
        parts = [sub(
            "xd(t,x) = {}*x*(1-x/L) + b", "r(t)" if self.N > 3 else "r")]
        if self.N > 3:
            parts.extend([
                "r(t) = r1 + c12*(1 + tanh(1.1*(t-t1)/s1))",
                "c12 = 0.5*(r2-r1)",
            ])
        if self.N > 6:
            parts.insert(-1, "     + r2 + c23*(1 + tanh(1.1*(t-t2)/s2))")
            parts.append("c23 = 0.5*(r3-r2)")
        return "\n".join(parts)

    def r_3(self, t, r1):
        """
        Implements C{r(t)} for a 3-parameter model having a single growth
        regime (conventional logistic growth model).
        """
        return r1

    def r_6(self, t, r1, t1, s1, r2):
        """
        Implements C{r(t)} for a 6-parameter model having a two growth
        regimes:::

            r(t) = r1 + c12*(1 + tanh(1.1*(t-t1)/s1))
            c12 = 0.5*(r2-r1)
        """
        c12 = 0.5*(r2-r1)
        return r1 + c12*(1 + np.tanh(1.1*(t-t1)/s1))

    def r_9(self, t, r1, t1, s1, r2, t2, s2, r3):
        """
        Implements C{r(t)} for a 9-parameter model having a three growth
        regimes:::

            r(t) = r1
                 + c12*(1 + tanh(1.1*(t-t1)/s1))
                 + c23*(1 + tanh(1.1*(t-t1-t2)/s2))
            c12 = 0.5*(r2-r1)
            c23 = 0.5*(r3-r2)
        """
        r12 = self.r_6(t, r1, t1, s1, r2)
        c23 = 0.5*(r3-r2)
        return r12 + c23*(1 + np.tanh(1.1*(t-t1-t2)/s2))

    def xd(self, t, x, *values):
        """
        Implements C{xd(t,x)} for a 3-9 parameter model having 1-3 growth
        regimes. With a single growth regime (3 parameters), this is a
        conventional logistic growth (Verhulst) model:

        U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html},

        Given a scalar time (in days) followed by arguments defining
        curve parameters, returns the number of B{new} Covid-19
        cases expected to be reported on that day.::

            xd(t,x) = x*r(t)*(1-x/L) + b

        This requires integration of a first-order differential
        equation, which is performed in L{curve}.
        
        With 6 or 9 parameters, there are 2 or 3 growth regimes (my
        own non-expert addition, caveat emptor). Smooth transitions
        between regimes is effected by the hyperbolic tangent. This
        function has proved very useful for use in the nonlinear
        models I've developed for MOSFET circuit simulation.

        It provides a smooth transition from one simulation regime to
        another. In the case of COVID-19 infections, it seemed to me
        that there was such a smooth transition happening as the
        effect of social distancing measures have become felt in
        various countries/regions, and now as things have started
        opening up again mostly in red states.

        There are three main parameters to a hyperbolic tangent
        component of a model: The maximum magnitude (never quite
        achieved, only as a limit), the rate of transition, and the
        time at which transition is happening most rapidly, i.e., the
        middle. In this modified logistic growth model, those three
        terms appear as:

            - I{c12}: Change from growth regime #1 to
              growth regime #2),

            - I{s1}: Span of transition, days required to go about 1/2
              from first growth rate to the second one.
        
            - I{t1}: Days after the region's first reported case for
              the middle, when transition is fastest.

        Let's take a look at each of those parameters in turn.

        In a conventional logistic growth model, the cumulative number
        of population members I{x} starts to level off as C{x/L}
        starts to approach 1.0. The conventional logistic-growth
        function C{f(t,x)} never goes negative because its value
        tapers off to zero as I{x} approaches I{L}. The value of I{x}
        never can exceed I{L}. In the author's modification, the value
        of I{r} is not a constant but the result of a function of
        time:::

            r(t,r1,t1,s1,r2) = r1 + c12*(1 + tanh(1.1*(t-t1)/s1))
            c12 = 0.5*(r2-r1)
        
        Once I{t} passes I{t1}, the growth rate will have gone half of
        the way from its initial value I{r1} to its second value
        I{r2}.
        
        The Jacobian with respect to I{x} is provided by
        L{xd_jac}. Because no version of I{r(t)} is dependent on I{x},
        there are no complications doing the chain rule for
        differentiation.
        
        The additional parameters I{c23}, I{s2}, and I{t2} define the
        transition to a third growth regime if that is being modeled
        as well.
        """
        X = np.array(x)
        L, b = values[:2]
        return self.r(t, *values[2:])*X*(1 - X/L) + b
    
    def xd_jac(self, t, x, *values):
        """
        Jacobian for L{xd}, with respect to I{x}::

            xd(t, x) = r(t)*x*(1 - x/L) + b
            xd(t, x) = r(t)*x - r(t)/L*x^2

            x2d(t, x) = r(t) - 2*r(t)*x/L
            x2d(t, x) = r(t)*(1 - 2*x/L)
        """
        X = np.array(x)
        L = values[0]
        return self.r(t, *values[2:])*(1 - 2*X/L)

    #--------------------------------------------------------------------------
    
    def __call__(self, values, t0, t1, X0):
        """
        Call my instance with a list of parameter I{values}, starting
        I{t0} and ending times I{t1}, both in days from first report,
        and the fixed (known) model value I{X0} at I{t0}.

        Unless there was a problem with the ODE solver, returns an
        instance of L{Results} with the following attributes:

            - I{t}, a 1-D array containing the number of days since
              first report for each modeled day, in ascending order

            - I{XD}, the modeled number of new reported cases per
              day for each day in I{t}, and

            - I{X}, the modeled total number of reported cases for
              each day in I{t}.

        If there was a problem with the ODE solver, returns C{None}
        instead.

        Uses the logistic growth model, the power law model, or both,
        with or without a linear (constant xd) term.

        Obtains modeled numbers of new cases and total cases by
        integrating the first-order ODE C{xd(t, x)} starting at
        C{t=t0} and ending at C{t=t1}. If I{t1} is less than I{t0},
        the integration will go backwards using a sign-inverted output
        of C{xd(t, x)}. All of that is transparent to the user, and
        I{t} is in ascending order either way.
        """
        def f(t, x):
            return self.xd(t, x, *values)

        def jac(t, x):
            return [self.xd_jac(t, x, *values)]
        
        # Solve the initial value problem to get X, which in turns
        # lets us obtain values for XD, dependent as it probably is
        # (due to logistic component likely included) on X
        r = Results(t0, t1)
        sol = solve_ivp(
            f, [t0, t1], [X0],
            method='Radau', t_eval=r.t, vectorized=True, jac=jac)
        if not sol.success:
            return
        r.X = sol.y[0]
        if t1 < t0: r.flip()
        r.XD = f(r.t, r.X)
        return r
    
            
class Evaluator(Picklable):
    """
    I evaluate fitness of the model defined by my L{Model} instance
    I{model} against the number of new COVID-19 cases reported each
    day.

    The data in my 1-D array I{XD} contains daily numbers of B{new}
    reported COVID-19 cases. The modeled value of the functions xd{x}
    is the number of new cases expected to be reported and I{t} is the
    time since the first observation (in days).
    
    Construct an instance of me, run the L{setup} method, and wait (in
    non-blocking Twisted-friendly fashion) for the C{Deferred} it
    returns to fire. Then call the instance a bunch of times with
    parameter values for a L{curve} to get a (deferred)
    sum-of-squared-error fitness of the curve to the actual data.

    @ivar XD: The actual number of new cases per day, each day, for
        the selected country or region since the first reported case
        in the NYT dataset on January 21, 2020, B{after} my
        L{transform} has been applied.
    """
    scale_SSE = 1e-2

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __len__(self):
        return len(self.data.t)
    
    def kt(self, t):
        """
        Returns an index to my I{t} and I{X} vectors for the specified
        number of days I{t} after 1/21/20.

        If I{t} is an array, an array of integer indices will be
        returned.
        """
        K = np.searchsorted(self.data.t, t)
        N = len(self)
        return np.clip(K, 0, N-1)
    
    def Xt(self, t):
        """
        Returns the value of I{X} at the specified number of days I{t}
        after 1/21/20. If the days are beyond the limits of the data
        on hand, the latest data value will be used.
        """
        return self.data.X[self.kt(t)]

    def relations(self):
        """
        Returns an instance of L{RelationsChecker} constructed with my
        data's I{relations} dict-of-dicts, or C{None} if no relations
        defined.
        """
        if self.data.relations is None:
            return
        return RelationsChecker(self.data.relations)

    def dayText(self, k=None):
        """
        Returns text indicating the date I{k} days after the first
        reported case.

        To get text for the last (most recent) reported case, leave
        I{k} C{None}.
        """
        if k is None:
            day = self.dates[-1]
        else:
            firstDay = self.dates[0]
            day = firstDay + timedelta(days=k)
        return day.strftime("%m/%d")

    def setup(self, state, county, daysAgo=0):
        """
        Call with a state and county.

        Computes a differential vector I{XD} that contains the
        differences between a particular day's cumulative cases and
        the previous day's. The length this vector is the same as
        I{X}, with a zero value for the first day.
        
        Returns a C{Deferred} that fires with two equal-length
        sequences, the names and bounds of all parameters to be
        determined.

        Also creates a dict of I{indices} in those sequences, keyed by
        parameter name.

        @keyword daysAgo: Set to a positive number of days ago
            to limit the latter end of the John Hopkins data. Useful
            for back-testing or when the current day's data dump is
            believed to be not yet complete.
        @type daysAgo: int
        """
        def done(null):
            # Calculate differential and show data on console
            self.XD = np.zeros_like(self.X)
            msg("Cumulative and new cases reported thus far for {}",
                self.location, '-')
            xPrev = None
            for k, x in enumerate(self.X):
                xd = 0 if xPrev is None else x-xPrev
                xPrev = x
                msg("{:03d}\t{}\t{:d}\t{:d}",
                    k, self.dayText(k), int(x), int(xd))
                self.XD[k] = self.transform(xd)
            # Done, return names and bounds to caller
            return names, bounds

        # Set region of interest
        if state and county:
            self.location = sub("{} County, {}", county, state)
        elif state:
            self.location = state
        else: self.location = "entire U.S."
        data = self.data = Covid19Data()
        data.state = state; data.county = county
        # Load specs, possibly including narrowed parameter bounds for
        # this region
        s = SpecsLoader('covid19.specs')()
        self.suffix = sanitized(state, county)
        dictName = sub("C19_{}", self.suffix)
        data.k0 = int(s.get(dictName, 'k0'))
        self.position = s.get('pos_defaults')
        self.position.update(s.get(dictName, 'pos'))
        data.relations = {}
        relations = s.get(dictName, 'relations')
        for var_12 in relations:
            m, b, limit = relations[var_12]
            var_1, var_2 = var_12.split('_')
            data.relations.setdefault(var_1, {})[var_2] = [m, b, limit]
        names = []; bounds = []
        pb = s.get(dictName, 'pb')
        for name in Model.names:
            if name in pb:
                names.append(name)
                bounds.append(pb[name])
        # An instance of Model for all fitting and plotting
        self.model = Model(names)
        return data.setup(daysAgo).addCallbacks(done, oops)

    def transform(self, XD=None, inverse=False):
        """
        Applies a transform to the numbers of new cases per day each day
        I{XD}, real or modeled. Set I{inverse} C{True} to apply the
        inverse of the transform.

        To use your own (presumably modeled) I{XD}, supply it via that
        keyword. Otherwise, I will do an inverse transform on my own
        I{XD} attribute. In that case, the I{inverse} keyword is
        ignored because it is assumed to be C{True}.

        The crude transform currently used is just a square root of
        the absolute magnitude, with sign preserved.  The seems like a
        reasonable initial compromise between a log transform (useful
        for a purely exponential model), and not transforming at
        all. Will investigate Cox-Box as an option.
        """
        if XD is None:
            XD = self.XD
            inverse = True
        if inverse:
            return np.sign(XD) * XD**2
        return np.sign(XD) * np.sqrt(np.abs(XD))
    
    def residuals(self, values):
        """
        Computes a 1-D array of transformed residuals to I{XD} from my
        L{curve}, fixed to the last known value with other derivatives
        computed backwards through my time vector I{t}.

        The residual array is left-trimmed to start with my first
        valid day I{k0}.

        Sets I{XD} and I{R} attributes to the instance of L{Results}
        that is obtained from calling L{curve} and returns the
        instance, unless a C{None} object was obtained because of an
        ODE problem, in which case C{None} is returned.

        Obtains a L{Results} instance with a residuals vector I{R}
        computed between the modeled and actual values, and sets its
        I{SSE} to the sum of squared error of those residuals.

        Returns the instance with I{R} and I{SSE} set, as well as
        I{t}, I{XD}, and I{X}. If computation of the modeled values
        failed, returns C{None}.
        """
        t0 = self.data.t[-1]
        t1 = self.data.t[self.k0]
        r = self.model(values, t0, t1, self.Xt(t0))
        if r is None: return
        r.XD = self.transform(r.XD)
        K = [self.kt(x) for x in r.t]
        # During setup, self.XD got its transform done for all time
        r.R = r.XD - self.XD[K]
        if r is not None:
            r.SSE = self.scale_SSE * np.sum(np.square(r.R))
        return r
    
    def __call__(self, values):
        """
        Evaluation function for the parameter I{values}.

        If the ODE is unsuccessful and calling L{residuals} results in
        a C{None} object, returns a crazy bad SSE.
        """
        r = self.residuals(values)
        return getattr(r, 'SSE', 1e9)


class Reporter(object):
    """
    An instance of me is called each time a combination of parameters
    is found that's better than any of the others thus far.

    Prints the sum-of-squared error and parameter values to the
    console and updates a plot image (PNG) at L{plotFilePath}.

    @cvar minShown: Minimum number of reported cases to show at the
        bottom of the log plot.

    @cvar N: Number of random points in extrapolated scatter plot.
    """
    minShown = 10
    daysForward = 30
    
    def __init__(self, runner, ratio=False, daily=False):
        """
        C{Reporter(evaluator, population, ratio=False, daily=False)}
        """
        for name in ('names', 'ev', 'p'):
            setattr(self, name, getattr(runner, name))
        self.includeRatio = ratio
        self.includeDaily = daily
        self.prettyValues = self.p.pm.prettyValues
        N = 3; height = 17
        for option in ratio, daily:
            if option:
                N += 1
                height += 2
        self.pt = Plotter(
            1, N,
            filePath=self.plotFilePath, width=12, height=height, h2=[0, 2])
        self.pt.use_grid()
        self.pt.set_fontsize('textbox', 12)
        ImageViewer(self.plotFilePath)

    @property
    def kToday(self):
        """
        Returns the current (as of the last known date) number of days
        since the date of first reported case.

        Calling this "kToday" is a holdover from when there was no
        option to right-trim the dataset with the '-d' command-line
        option for backtesting, when the most recent known date was
        always assumed to be the current date.
        """
        firstDay = self.ev.dates[0]
        # Unused code from the previous stupid version where current
        # date was used, not last known, left in as a comment for
        # historical reference
        # ---------------------------------------------------------------------
        # seconds_in = (date.today() - firstDay).total_seconds()
        # ---------------------------------------------------------------------
        seconds_in = (self.ev.dates[-1] - firstDay).total_seconds()
        return int(seconds_in / 3600 / 24)

    @property
    def kForToday(self):
        """
        Property: A 2-tuple with the last known date's index (the current
        number of days since first reported case) and k_offset.
        
        The value of k_offset will be zero if latest data includes
        today, higher for each day the data is stale.

        Calling this "kForToday" is a holdover from when there was no
        option to right-trim the dataset with the '-d' command-line
        option for backtesting, when the most recent known date was
        always assumed to be the current date.
        """
        kToday = self.kToday
        return kToday, kToday - len(self.ev) + 1

    @property
    def plotFilePath(self):
        """
        Property: A file path (in the current
        directory) of a PNG file to write an update with a Matplotlib
        plot image of the modeling result.

        The suffix for the region is included, along with a I{.png}
        extension.
        """
        suffix = self.ev.suffix
        if not suffix: suffix = "US"
        return sub("covid19-{}.png", suffix)
                       
    def pos(self, name):
        """
        Returns a text box position identified by I{name}, or the default
        position specified in the "pos_defaults" dict if undefined.
        """
        return self.ev.position[name]
    
    def clipLower(self, X):
        return np.clip(X, self.minShown, None)

    def curvePoints(self, values, k0, k1):
        """
        Obtains modeled numbers of new daily cases and cumulative reported
        cases from I{k0} to I{k1} days after first reported case.
        
        Given a sequence of parameter I{values} and a first I{k0} and
        second I{k1} number of days from first reported case, returns
        (1) a 1-D array I{t} with days from first reported case, in
        ascending order between I{k0} and I{k1}, inclusive, and (2) a
        1-D array I{X} with the expected number of B{cumulative} cases
        reported on each of those days.

        The value of I{X} corresponding to time I{k0} is anchored to
        the known number of reported cases then. Integration for
        extrapolation will have I{k1} greater than I{k0}. For
        backwards integration (to do historical comparisons), set
        I{k1} smaller than I{k0}. In any event, the I{t} and I{X} 1-D
        arrays in the returned I{r} object will be in ascending order
        of time.

        Returns an instance I{r} of I{Results} with the modeled arrays.
        """
        t0, t1 = [float(x) for x in [k0, k1]]
        r = self.ev.model(values, t0, t1, self.ev.Xt(t0))
        if r is None: return
        return r

    def add_model(self, ax, t, X, semilog=False):
        f = getattr(ax, 'semilogy' if semilog else 'plot')
        f(t, X, color='red', marker='o', linewidth=2, markersize=3)

    def add_scatter(self, ax, k0, k1, k_offset=0):
        """
        Adds points, using randomly selected parameter combos in present
        population, at times that are plotted with a slight random
        dither around each day in the future.
        """
        Nt = k1 - k0
        Ni = len(self.p)
        tc = np.empty((Nt, Ni))
        Xc = np.empty_like(tc)
        X0 = self.ev.Xt(k0)
        for ki in range(Ni):
            r = self.curvePoints(list(self.p[ki]), k0, k1)
            tc[:,ki], Xc[:,ki] = r.t+0.1*stats.norm.rvs(size=Nt), r.X
        tc = tc.flatten() - k_offset
        Xc = self.clipLower(Xc.flatten())
        ax.semilogy(tc, Xc, color='black', marker=',', linestyle='')

    @staticmethod
    def cases(X, k):
        return int(round(X[k]))
        
    def annotate_past(self, sp, X, k=None):
        """
        Given a 1-D array I{X} of case numbers that will be displayed as
        the first line in the supplied subplot I{sp}, adds an
        annotation for the actual number I{k} days after first report.

        The last item in array I{X} must be the most recent actual
        number of cases.

        To just annotate the last item in I{X}, leave I{k} C{None}.
        """
        if k is None:
            kX = -1
        else:
            # Length of X, just actual data to be plotted up to most
            # recent
            N = len(X)
            # Convert index from all data to just data to be plotted
            kX = k - (len(self.ev) - N)
            if kX >= N: return
            if kX < 0: return
        sp.add_annotation(
            kX, "{}: {:,.0f}", self.ev.dayText(k), self.cases(X, kX))
    
    def model_past(self, sp, values):
        """
        Plots the past data against the best-fit model in subplot I{sp},
        given the supplied parameter I{values}, with the model curve
        anchored at the right to today's actual reported cases.

        Returns a 4-tuple of equal-length 1-D arrays with (1) the
        actual time I{t}, (2) the actual cumulative numbers of cases
        I{X_data}, (3) the actual numbers of new daily cases, and (4)
        the modeled number of new daily cases.
        """
        def annotate_error(k):
            kk = k - k0 - 1
            xc = X_curve[kk]
            x = X_data[kk]
            xe = (xc - x)/x
            sp.add_annotation(
                kk, "{}: {:+.1f}%",
                self.ev.dayText(k), int(round(100*xe)), kVector=1)
            
        k0 = self.ev.k0
        kToday, k_offset = self.kForToday
        r = self.curvePoints(values, kToday, k0)
        t, X_curve = r.t, r.X
        t -= k_offset
        K = self.ev.kt(t)
        X_data = self.clipLower(self.ev.X[K])
        XD = self.ev.transform()[K]
        # Date of first reported case number plotted
        self.annotate_past(sp, X_data, k0+1)
        # Error in expected vs actual reported cases, going back
        # several days starting with today
        kList = range(kToday, k0, -1)
        Nfd = len(kList) / 15
        N_data = len(X_data)
        for kk, k in enumerate(kList):
            k_data = N_data - kk - 1
            if not k_data % 7:
                sp.add_axvline(k_data)
            if kk > 10:
                if Nfd and kk % Nfd: continue
            annotate_error(k)
        ax = sp.semilogy(t, X_data)
        # Add the best-fit model values for comparison
        self.add_model(ax, t, X_curve, semilog=True)
        return t, X_data, XD, r.X, r.XD
    
    def model_future(self, sp, values):
        """
        Plots the past data against the best-fit model in subplot I{sp},
        given the supplied parameter I{values}.

        The model curve is anchored at the most recent data point in
        the actual reported cases, and then extends beyond for
        extrapolation.

        If the most recent data point is not from today, the
        annotation will actually start in the past. The number of days
        between today (when this is run) and the date of the last data
        point is subtracted from the "days after 1/21/20" index before
        being applied to the method L{dayText}.

        Returns the modeled time I{t}, cumulative cases I{X_curve},
        and new daily cases I{XD}.
        """
        def annotate_past(k):
            self.annotate_past(sp, X_data, k)

        def annotate_future(daysInFuture):
            """
            Add annotation for I{daysInFuture}, returning C{True} if it
            actually was added.
            """
            k_curve = daysInFuture + k_offset
            if k_curve < 0 or k_curve >= len(X_curve):
                return
            sp.add_annotation(
                k_curve, "{}: {:,.0f}",
                self.ev.dayText(k0+daysInFuture),
                self.cases(X_curve, k_curve), kVector=1)
            return True
            
        N_back = 4
        k0, k_offset = self.kForToday
        k1 = k0 + self.daysForward
        t_data, X_data = [
            getattr(self.ev, name)[-N_back:] for name in ('t', 'X')]
        r = self.curvePoints(values, k0, k1)
        t = r.t - k_offset
        X_curve = r.X
        # Vertical line "today"
        t0 = self.ev.t[-1]
        sp.add_axvline(t0)
        # "Today" + previous few days
        for k in range(k0-N_back, k0+1):
            annotate_past(k)
        # This next week
        for k in range(1, 7):
            annotate_future(k)
        # Days after next week, in one-week intervals
        for weeks in range(1, 10):
            if not annotate_future(7*weeks):
                break
            sp.add_axvline(t0 + 7*weeks)
        # Start with a few of the most recent actual data points
        sp.add_axvline(self.ev.t[-1])
        ax = sp.semilogy(t_data, X_data)
        # Add the best-fit model extrapolation
        self.add_model(ax, t, X_curve, semilog=True)
        # Add scatterplot sorta-probalistic predictions
        self.add_scatter(ax, k0, k1, k_offset)
        return t, X_curve, r.XD
    
    def AICc(self, r):
        """
        Computes the Akaike Information Criterion, corrected for small
        sample size (AICc) for the residuals I{R} in the supplied
        instance I{r} of L{Results}.

        The residuals were previou between transformed model and actual
        values. The AICc is defined as follows:::

            AIC = 2*k + N*ln(SSE/N)
        
            AICc = AIC + 2*(k^2 + k)/(N-k-1)

        where SSE is the sum-of-squared error (i.e., the "residual
        sums-of-squares," as Brandmaier puts it at
        U{https://www.researchgate.net/post/What_is_the_AIC_formula},
        already computed and stored in I{r} as I{SSE}, I{N} is the
        number of samples (i.e., residual vector length), and I{k} is
        the number of model parameters, not to be confused with the
        commonly used index variable of the same name.

        Although it provides no absolute indication of fitness, the
        lower the AICc, the better the model is considered to be. Use
        it for deciding what components of the model should be
        included for a particular country/region.
        
        Returns a 3-tuple with AICc, N, k
        """
        N = len(r.R)
        k = self.p.Nd
        AIC = 2*k + N*np.log(r.SSE/N)
        AICc = AIC + 2*(k**2 + k)/(N-k-1)
        return AICc, N, k
    
    def fit_info(self, sp, r):
        """
        Adds info to subplot I{sp} about the goodness of fit for residuals
        I{R} in the supplied instance I{r} of L{Results}.
        """
        def tb(*args):
            sp.add_textBox(self.pos('stats'), *args)
            msg(*args)

        # Significance of non-normality
        tb("Non-normality: p = {:.4f}", stats.normaltest(r.R)[1])
        # AICc
        AICc, N, k = self.AICc(r)
        tb("AICc is {:+.2f} with SSE={:.5g}, N={:d}, k={:d}",
           AICc, r.SSE, N, k)
        
    def subplot_upper(self, sp, values):
        """
        Does the upper subplot with model fit vs data.

        Returns the I{t} I{X} vectors for the actual data being
        plotted, the I{XD} vector for the actual data, the I{X} vector
        for the modeled data, and the I{XD} vector for the modeled
        data.
        """
        sp.add_line('-', 2)
        sp.set_tickSpacing('x', 7.0, 1.0)
        sp.add_textBox(
            self.pos('model'), self.ev.model.f_text)
        for k, name in enumerate(self.names):
            sp.add_textBox(
                self.pos('params'), "{}: {:.5g}", name, values[k])
        # Data vs best-fit model
        return self.model_past(sp, values)

    def subplot_middle(self, sp, values, t):
        """
        Does a middle subplot with residuals between the data I{X_data}
        and modeled data I{X_curve}, given model parameter I{values}
        and evaluation times (days after 1/21/20) I{t}.
        """
        r = self.ev.residuals(values)
        if r is None: return
        sp.set_ylabel("Residual")
        sp.set_xlabel("Modeled (fitted) new cases/day (square-root transform)")
        sp.set_zeroLine(color="red", linestyle='-', linewidth=3)
        sp.add_line('')
        sp.add_marker('o', 4)
        sp.set_colors("purple")
        sp.add_textBox(
            'NW', "Residuals: Modeled vs actual new cases/day (transformed)")
        self.fit_info(sp, r)
        sp(r.XD, r.R, zorder=3)
        
    def subplot_lower(self, sp, values):
        """
        Does the lower subplot with extrapolation, starting at my I{k0}
        days from first report.

        Returns the I{t} and I{X} vectors for the future modeled data
        being plotted, plus the modeled I{XD} vector.
        """
        sp.add_line('-', 2)
        sp.set_tickSpacing('x', 7.0, 1.0)
        return self.model_future(sp, values)

    def subplot_ratio(self, sp, ta, Ra, tm, Rm):
        """
        Draws an optional subplot below the main ones showing the modeled
        ratio between new and total cases, both past (actual) and
        future (modeled).

        Call with actual-data time and ratio vectors I{ta}, I{Ra} and
        modeled time and ratio vectors I{tm}, I{Rm}.
        """
        Ra = 100 * Ra
        K = np.flatnonzero(np.logical_or(Rm < 0, Rm > 1.0))
        Rm[K] = 1.0
        Rm = 100 * Rm
        sp.add_axvline(-1)
        sp.add_annotation(
            ta[-1], "{} had {:.1f}% of all\ncase reports thus far",
            self.ev.dayText(), Ra[-1])
        sp.add_textBox(
            self.pos('daily_pct'),
            "New cases each day as a percentage of total cases thus far")
        sp.set_ylabel("% New")
        ax = sp(ta, Ra)
        self.add_model(ax, tm, Rm)

    def subplot_daily(self, sp, ta, XDa, tm, XDm):
        """
        Draws an optional subplot below the main ones showing the number
        of new daily cases, both past (actual) and future (modeled).

        Call with actual-data time and daily-case vectors I{ta},
        I{XDa} and modeled time and daily-case vectors I{tm}, I{XDm}.
        """
        sp.add_axvline(-1)
        sp.add_annotation(
            ta[-1], "{} had {:.0f} new case reports",
            self.ev.dayText(), XDa[-1])
        sp.add_textBox(
            self.pos('daily_new'),
            "New cases reported each day")
        sp.set_ylabel("Newly Reported")
        ax = sp(ta, XDa)
        self.add_model(ax, tm, XDm)
        
    def __call__(self, values, counter, SSE):
        """
        Prints out a new best parameter combination and its curve vs
        observations, with lots of extrapolation to the right.
        """
        def tb(*args):
            if len(args[0]) < 3:
                pos = args[0]
                args = args[1:]
            else: pos = self.pos('summary')
            sp.add_textBox(pos, *args)
        
        # Make a frozen local copy of the values list to work with, so
        # outdated stuff doesn't get plotted
        values = list(values)
        msg(0, self.prettyValues(
            values, "SSE={:.5g} on eval {:d}:", SSE, counter), '-')
        self.pt.set_title(
            "Modeled (red) vs Actual (blue) Reported Cases of COVID-19: {}",
            self.ev.location)
        self.pt.set_ylabel("Reported Cases")
        self.pt.set_xlabel("Days after January 22, 2020")
        self.pt.use_minorTicks('x', 1.0)
        with self.pt as sp:
            tb('S', "Reported cases in {} vs days after first case.",
               self.ev.location)
            tb('S', "Annotations show residuals between model and data.")
            ta, Xa, XDa, Xam, XDam = self.subplot_upper(sp, values)
            self.subplot_middle(sp, values, ta)
            tb("Expected cases reported in {} vs days after",
               self.ev.location)
            tb("first case. Dots show daily model predictions for each")
            tb("of a final population of {:d} evolved parameter", len(self.p))
            tb("combinations. Annotations show actual values in")
            tb("past, best-fit projected values in future.")
            for line in DISCLAIMER.split('\n'):
                sp.add_textBox("SE", line)
            tm, Xm, XDm = self.subplot_lower(sp, values)
            if self.includeRatio or self.includeDaily:
                tm = np.concatenate([ta, tm])
                Xm = np.concatenate([Xam, Xm])
                XDm = np.concatenate([XDam, XDm])
                if self.includeRatio:
                    self.subplot_ratio(sp, ta, XDa/Xa, tm, XDm/Xm)
                if self.includeDaily:
                    self.subplot_daily(sp, ta, XDa, tm, XDm)
        self.pt.show()
        msg("")


class Runner(object):
    """
    I run everything to fit a curve to thermistor data using
    asynchronous differential evolution.

    Construct an instance of me with an instance of L{Args} that has
    parsed command-line options, then have the Twisted reactor call
    the instance when it starts. Then start the reactor and watch the
    fun.
    """
    def __init__(self, args):
        """
        C{Runner(args)}
        """
        if args.b and args.r:
            raise ValueError("Options -b and -r are mutually exclusive!")
        self.args = args
        self.ev = Evaluator()
        if args.t:
            self.N_cores = 1
            self.q = ThreadQueue(returnFailure=True)
        else:
            self.N_cores = args.N if args.N else ProcessQueue.cores()-1
            self.q = ProcessQueue(self.N_cores, returnFailure=True)
        self.fh = open("covid19.log", 'w') if args.l else True
        msg(self.fh)

    @defer.inlineCallbacks
    def shutdown(self):
        """
        Call this to shut me down when I'm done. Shuts down my
        C{ProcessQueue}, which can take a moment.

        Repeated calls have no effect.
        """
        yield self.p.reporter.waitForCallbacks()
        if self.q is not None:
            msg("Shutting down...")
            yield self.q.shutdown()
            msg("Task Queue is shut down")
            self.q = None
            msg("Goodbye")

    def tryGetClass(self, suffix):
        """
        Tries to return a reference to a subclass of L{Covid19Data} with
        the prefix I{C19} followed by an underscore and then the
        supplied I{suffix}.

        If I{suffix} is an empty string, the L{C19_} full-US subclass
        is used.
        """
        name = "C19_"
        if suffix: name += suffix
        try: klass = globals()[name]
        except ImportError:
            klass = None
        if klass is not None and issubclass(klass, Covid19Data):
            msg("Using data class '{}'", name)
            return klass
        raise ImportError(sub("No data subclass '{}' found!", name))
            
    def evaluate(self, values):
        """
        The function that gets called with each combination of parameters
        to be evaluated for fitness.
        """
        if values is None:
            return self.shutdown()
        values = list(values)
        if self.q: return self.q.call(self.ev, values)

    @defer.inlineCallbacks
    def reEvaluate(self):
        """
        Re-evaluates loaded individuals, keeping the previous history if
        not too much has changed SSE-wise.
        """
        def done(i, k, SSE_prev):
            SSE = float(i.SSE)
            if not SSE or np.isinf(SSE_prev) or np.isinf(SSE):
                info = ": !!!"
            else:
                rd = (SSE-SSE_prev) / SSE_prev
                if abs(rd) > 1E-3:
                    info = sub(": {:+.1f}%", 100*rd)
                    iChanged.add(i)
                else: info = ""
            msg(2, "{:03d}  {:>8.1f} --> {:>8.1f}  {}", k, SSE_prev, SSE, info)
        
        msg(-1, "Re-evaluating {:d} loaded Individuals...", len(self.p), '-')
        iChanged = set()
        dt = DeferredTracker(interval=0.05)
        yield self.p.history.purgePop()
        for k, i in enumerate(self.p):
            SSE_prev = float(i.SSE)
            d = i.evaluate().addCallback(done, k, SSE_prev)
            dt.put(d)
            yield dt.deferUntilFewer(self.N_cores)
        yield dt.deferToAll()
        msg("")

    @defer.inlineCallbacks
    def __call__(self):
        """
        Where the action happens.
        """
        def picklePath():
            """
            Returns a consistent path for a pickle file in which to dump/load
            the final population for a given state and county.

            If the state or county is not specified, supply C{None} for
            that arg.

            The file always goes in the current working directory,
            although that could be easily changed.
            """
            suffix = sanitized(state, county)
            possibleDash = "-" if suffix else ""
            return sub("covid19{}{}.dat", possibleDash, suffix)
        
        args = self.args
        startTime = time.time()
        if len(args) > 2:
            raise RuntimeError("You must supply 0-2 args")
        if len(args) == 2:
            state = args[0]
            county = args[1]
        elif len(args) == 1:
            state = args[0]
            county = None
        else:
            state = None
            county = None
        self.names, self.bounds = yield self.ev.setup(
            state, county, args.d).addErrback(oops)
        if args.L:
            loadPicklePath = picklePath()
            self.p = Population.load(
                loadPicklePath, func=self.evaluate, bounds=self.bounds)
            msg("Resuming from population saved in {}", loadPicklePath)
        else:
            self.p = Population(
                self.evaluate, self.names, self.bounds, popsize=args.p)
        rc = self.ev.relations()
        if rc: self.p.setConstraints(rc)
        reporter = Reporter(self, ratio=not args.R, daily=not args.D)
        self.p.addCallback(reporter)
        if len(self.p):
            yield self.reEvaluate()
        elif self.p.running is None:
            OK = yield self.p.setup()
            if not OK:
                yield self.shutdown()
        self.p.initialize()
        F = [float(x) for x in args.F.split(',')]
        de = DifferentialEvolution(
            self.p,
            CR=args.C, F=F, maxiter=args.m,
            uniform=args.u, dwellByGrave=1,
            randomBase=0.0 if args.b else 1.0 if args.r else 0.5,
            adaptive=not args.n, bitterEnd=args.e, logHandle=self.fh)
        yield de()
        msg(0, "Final population:\n{}", self.p)
        msg(0, "Elapsed time: {:.2f} seconds", time.time()-startTime, 0)
        savePicklePath = picklePath()
        self.p.save(savePicklePath)
        msg("Saved final population of best parameter combinations "+\
            "to {}", savePicklePath)
        yield self.shutdown()
        reactor.stop()
        msg(None)

    @staticmethod
    def profile(f, *args, **kw):
        def done(result):
            pr.disable()
            import pstats
            with open('covid19-profile.out', 'wb') as fh:
                ps = pstats.Stats(pr, stream=fh).sort_stats('cumulative')
                ps.print_stats()
            return result
        
        def substituteFunction(*args, **kw):
            d = f(*args, **kw)
            d.addCallback(done)
            return d
    
        from cProfile import Profile
        pr = Profile(); pr.enable()
        substituteFunction.func_name = f.func_name
        return substituteFunction()

    def run(self):
        return self().addErrback(oops)
    
    
def main():
    """
    Called when this module is run as a script.
    """
    r = Runner(args)
    if args.i:
        reactor.callWhenRunning(r.profile, r.run)
    else: reactor.callWhenRunning(r.run)
    reactor.run()


args = Args(
    """
    Parameter finder for model of Covid-19 number of US reported cases
    vs time using Differential Evolution.

    Downloads a compressed CSV file of Covid-19 data points from
    edsuom.com to the current directory (if it's not already
    present). The data points are plotted against the current best-fit
    curve, along with a statistical extrapolation plot, in the PNG
    file (also in the current directory) covid19.png.

    Dumps a finalized ade.Population object to a pickle file
    covid19-<state>_<county>-N[369].dat. You can resume from such a
    file with the C{-L} option but processing is slower for some
    reason.
    
    On Linux, a plot window should be opened automatically. If not,
    you can obtain the free "qiv" image viewer and see the plots,
    automatically updated, with the command "qiv -Te
    covid19.png". (Possibly the other OS's may have something that
    works, too.)

    Press the Enter key to quit early.
    """
)
args('-d', '--days-ago', 0,
     "Limit latest data to N days ago rather than up to today")
args('-m', '--maxiter', 100, "Maximum number of DE generations to run")
args('-e', '--bitter-end', "Keep working to the end even with little progress")
args('-p', '--popsize', 40, "Population: # individuals per unknown parameter")
args('-C', '--CR', 0.7, "DE Crossover rate CR")
args('-F', '--F', "0.5,1.0", "DE mutation scaling F: two values for range")
args('-r', '--random', "Use DE/rand/1/bin instead of DE/rand-0.50/1/bin")
args('-b', '--best', "Use DE/best/1/bin instead of DE/rand-0.50/1/bin")
args('-n', '--not-adaptive', "Don't use automatic F adaptation")
args('-u', '--uniform', "Initialize population uniformly instead of with LHS")
args('-N', '--N-cores', 0, "Limit the number of CPU cores")
args('-t', '--threads',
     "Use a single worker thread instead of processes (for debugging)")
args('-l', '--logfile',
     "Write results to logfile 'covid19.log' instead of STDOUT")
args('-L', '--pickle-load',
     "Resume previous ade.Population object stored for this state/county")
args('-R', '--exclude-ratio',
     "Exclude subplot with ratio of new vs cumulative cases")
args('-D', '--exclude-daily',
     "Exclude subplot with new daily cases")
args('-i', '--profile', "Run with the Python profiler (slower)")
args("[<State Name> [<County Name>]]")
args(main)
