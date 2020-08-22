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

import re, random, textwrap
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
Dedicated to the public domain by Edwin
A. Suominen. My only relevant expertise is in
fitting nonlinear models to data, not biology or
medicine. See disclaimer in source file covid19.py.
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
    if not parts: parts.append("US")
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

    There is one epidemiological assumption being made: Parameter L is
    set to an assumed fraction of the state or county's
    population. That fraction is set by global specs parameter I{Lp}.
    
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
        # --- Weekly variation --------------------------------------------
        # aw: Amplitude of weekly variation (fractional, <1.0)
        'aw',
        # tw: Time when at mean (non-varied) value (days after first case)
        'tw',
        # -----------------------------------------------------------------
    ]

    #--- Logistic Growth Model with Multiple Growth Regimes -------------------

    def __init__(self, names, L):
        self.L = L
        OK = False
        names = set(names)
        self.two_pi_7 = None
        if {'b', 'r1'} <= names:
            N = 2
            OK = True
        if {'aw', 'tw'} <= names:
            N = 4
            self.two_pi_7 = 2*np.pi/7
        if {'t1', 's1', 'r2'} <= names:
            if N >= 2:
                N += 3
            else: OK = False
        if {'t2', 's2', 'r3'} <= names:
            if N >= 5:
                N += 3
            else: OK = False
        if len(names) > N: OK = False
        if not OK:
            raise ValueError(sub(
                "Parameters {} not a valid combination!", ', '.join(names)))
        self.r = self.r_2 if N in (2,4) else self.r_5 if N in (5,7) else self.r_8
        self.N = N

    @property
    def f_text(self):
        terms = [sub("{}*x*(1-x/L)", "r(t)" if self.N > 4 else "r")]
        if self.two_pi_7: terms.insert(0, "w(t)")
        parts = [sub("xd(t,x) = {} + b", "*".join(terms))]
        if self.two_pi_7: parts.append("w(t) = 1+aw*sin(2*pi/7*(t-tw))")
        if self.N > 2:
            parts.extend([
                "r(t) = r1 + c12*(1 + tanh(1.1*(t-t1)/s1))",
                "c12 = 0.5*(r2-r1)",
            ])
        if self.N > 5:
            parts.insert(-1, "     + r2 + c23*(1 + tanh(1.1*(t-t2)/s2))")
            parts.append("c23 = 0.5*(r3-r2)")
        return "\n".join(parts)

    def r_2(self, t, r1):
        """
        Implements C{r(t)} for a 2-parameter model having a single growth
        regime (conventional logistic growth model). (The value of
        I{L} is a single rough estimate based on population.)
        """
        return r1

    def r_5(self, t, r1, t1, s1, r2):
        """
        Implements C{r(t)} for a 5-parameter model having a two growth
        regimes:::

            r(t) = r1 + c12*(1 + tanh(1.1*(t-t1)/s1))
            c12 = 0.5*(r2-r1)
        """
        c12 = 0.5*(r2-r1)
        return r1 + c12*(1 + np.tanh(1.1*(t-t1)/s1))

    def r_8(self, t, r1, t1, s1, r2, t2, s2, r3):
        """
        Implements C{r(t)} for a 9-parameter model having a three growth
        regimes:::

            r(t) = r1
                 + c12*(1 + tanh(1.1*(t-t1)/s1))
                 + c23*(1 + tanh(1.1*(t-t1-t2)/s2))
            c12 = 0.5*(r2-r1)
            c23 = 0.5*(r3-r2)
        """
        r12 = self.r_5(t, r1, t1, s1, r2)
        c23 = 0.5*(r3-r2)
        return r12 + c23*(1 + np.tanh(1.1*(t-t1-t2)/s2))

    def w(self, t, aw, tw):
        """
        Implements C{w(t)} to impose a weekly variation on the growth
        rate, observed due to testing limitations on weekends:::

            w(t) = 1+aw*sin(2*pi/7*(t+tw))
        """
        return 1 + aw*np.sin(self.two_pi_7*(t + tw))

    def r(self, t, values):
        """
        Implements the term C{w(t)*r(t)}, returning the starting point for
        I{XD}.
        """
    
    def xd(self, t, x, *values):
        """
        Implements C{xd(t,x)} for a 3-9 parameter model having 1-3 growth
        regimes. With a single growth regime (3 parameters), this is a
        conventional logistic growth (Verhulst) model:

        U{https://services.math.duke.edu/education/ccp/materials/diffeq/logistic/logi1.html},

        Given a scalar time (in days) followed by arguments defining
        curve parameters, returns the number of B{new} Covid-19
        cases expected to be reported on that day.::

            xd(t,x) = r(t)*x*(1-x/L) + b

        If the I{aw} and I{tw} parameter bounds are defined, there is
        also a I{w(t)} term:::
        
            xd(t,x) = w(t)*r(t)*x*(1-x/L) + b
        
        This requires integration of a first-order differential
        equation, which is performed in L{curve}.

        The value of I{L} is a single rough estimate based on
        population. Set the maximum fraction of the population that
        could be infected with specs global parameter I{Lp}.
        
        With 5 or 8 parameters, there are 2 or 3 growth regimes (my
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
        values = list(values)
        b = values.pop(0)
        if self.two_pi_7:
            tw = values.pop()
            aw = values.pop()
        XD = self.r(t, *values)*X*(1 - X/self.L)
        if self.two_pi_7:
            XD *= self.w(t, aw, tw)
        return XD + b
    
    def xd_jac(self, t, x, *values):
        """
        Jacobian for L{xd}, with respect to I{x}::

            xd(t, x) = r(t)*x*(1 - x/L) + b
            xd(t, x) = r(t)*x - r(t)/L*x^2

            x2d(t, x) = r(t) - 2*r(t)*x/L
            x2d(t, x) = r(t)*(1 - 2*x/L)
        """
        X = np.array(x)
        N = len(values)
        values = values[1:N-2] if self.two_pi_7 else values[1:]
        return self.r(t, *values)*(1 - 2*X/self.L)

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

    @ivar pop: The population of my region of interest.
    """
    scale_SSE = 1e-5
    pes_default = 1
    no_transform = True

    @property
    def lastDay(self):
        """
        Property: An 8-character string with no spaces indicating the last
        (most recent) date in the dataset. For example, "20200801" for
        August 1, 2020.
        """
        return self.dates[-1].strftime("%y%m%d")
    
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
        else: self.location = "Entire U.S."
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
        self.pes = s.get(dictName, 'pes')
        if not self.pes: self.pes = self.pes_default
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
        # The logistic growth model's upper bound parameter L is
        # inferred from population, herd immunity threshold estimate,
        # and a rough guess about how many more actual cases than
        # reported there are. Including L as an unknown to be solved
        # for rarely adds much because the estimated values almost
        # always wind up at a substantial fraction of the population
        # except when they get evolved to very low, obviously
        # contrived values that only attempt to cover up the
        # limitations in the last growth regime.
        self.pop = s.get(dictName, 'pop')
        self.L = s.get('Lp')*self.pop
        # An instance of Model for all fitting and plotting
        self.model = Model(names, self.L)
        return data.setup(daysAgo).addCallbacks(done, oops)

    def transform(self, XD=None, inverse=False):
        """
        Applies a transform to the numbers of new cases per day each day
        I{XD}, real or modeled.  Set I{inverse} C{True} to apply the
        inverse of the transform.

        If my I{no_transform} attribute is set C{True}, however, the
        output will be the same as the input.

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
        if self.no_transform:
            return XD
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


class ErrorAnnotator(object):
    """
    I accumulate requests for annotations of the errors between
    modeled vs actual cases and then display a sensible subset of them
    without crowding things too much.

    @ivar wwe: B{W}orst B{W}eekly B{E}rrors, one entry per week prior
        to "today." Dict keyed by integer number of weeks ago (e.g.,
        key of 0 is for the most recent 7 days). Each entry is a
        2-tuple with C{(k, error)} where k is the number of days since
        the first plotted date and error is the percentage error
        (e.g., 1.5 for 0.015).
    """
    def __init__(self, ev, X_curve, X_data, kToday):
        self.ev = ev
        self.X_curve = X_curve
        self.X_data = X_data
        self.kToday = kToday
        self.k0 = ev.k0
        self.wwe = {}

    def kd(self, k):
        """
        Returns the number of days that I{k}, the supplied number of days
        from the first case, is from my first relevant date I{k0}.
        """
        return k - self.k0 - 1
            
    def error(self, kd):
        """
        Returns the percentage error between the modeled and actual number
        of cases the specified number of days I{k} from my first
        relevant date I{k0}.
        """
        xc = self.X_curve[kd]
        x = self.X_data[kd]
        xe = (xc - x)/x
        return 100*xe
    
    def add(self, k):
        """
        Adds an entry for the specified number of days I{k} from the first
        case. If the error on that date is worse than any other
        already added for that week before "today," it is the record.
        """
        kd = self.kd(k)
        daysBack = self.kToday - k
        weeksBack = int(daysBack/7)
        if weeksBack not in self.wwe:
            error_prev = 0
        else: error_prev = self.wwe[weeksBack][1]
        error = self.error(kd)
        if abs(error) > abs(error_prev):
            self.wwe[weeksBack] = (k, error)
 
    def annotate(self, sp):
        """
        Annotates the subplot referenced by I{sp} with a sensible number
        of requested annotations. Returns a text description of the
        date ranges and cutoffs that were applied.
        """
        N = 0
        for weeksBack in self.wwe:
            k, error = self.wwe[weeksBack]
            sp.add_annotation(
                self.kd(k), "{}: {:+.1f}%",
                self.ev.dayText(k), error, kVector=1)
            if weeksBack > N: N = weeksBack
        N += 1
        return sub(
            "Annotations: Worst model vs data error for each of {:d} "+\
            "preceding week{}", N, "s" if N > 1 else "")


class Reporter(object):
    """
    An instance of me is called each time a combination of parameters
    is found that's better than any of the others thus far.

    Prints the sum-of-squared error and parameter values to the
    console and updates a plot image (PNG) at L{plotFilePath}.

    @cvar minShown: Minimum number of reported cases to show at the
        bottom of the log plot.

    @ivar N: Number of subplots shown.
    """
    minShown = 10
    max_line_length = 60
    plot_width = 11
    plot_base_height = 4

    re_tn = re.compile(r't[1-9]')
    
    def __init__(
            self, runner, daysForward,
            pct=False, past=False, future=False, ratio=False, daily=False):
        """
        C{Reporter(runner, daysForward, **kw)}

        @keyword pct: Set C{True} to show future cases as percentage
            of population and new daily cases per 100,000 people in
            population.

        @keyword past: Set C{True} to include subplots with curve fit
            to past data.
        
        @keyword future: Set C{True} to include subplot with
            extrapolation of fitted curve.

        @keyword ratio: Set C{True} to include subplot with new cases
            each day as a percentage of cumulative on that day.

        @keyword daily: Set C{True} to include subplot with new cases
            each day.
        """
        for name in ('names', 'ev', 'p'):
            setattr(self, name, getattr(runner, name))
        self.daysForward = daysForward
        self.pct = pct
        self.includePast = past
        self.includeFuture = future
        self.includeRatio = ratio and not pct
        self.includeDaily = daily
        self.prettyValues = self.p.pm.prettyValues
        N = 0; height = self.plot_base_height
        h2 = [0]
        if past:
            N += 2
            height += 4
            if future: h2.append(2)
        if future:
            N += 1
            height += 3
        for option in self.includeRatio, self.includeDaily:
            if option:
                N += 1
                height += 3
        self.N = N
        self.pt = Plotter(
            1, N,
            filePath=self.plotFilePath,
            width=self.plot_width, height=height, h2=h2)
        self.pt.use_grid('x', 'major', 'y', 'both')
        self.pt.set_fontsize('textbox', 12 if N > 3 else 11)
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

        The suffix for the region is included, separated by an
        underscore ("_"), then the date separated by a hyphen, then a
        I{.png} extension. For example,
        "Washington_Spokane-20200801.png".
        """
        basename = self.ev.suffix
        if not basename: basename = "US"
        return sub("{}-{}.png", basename, self.ev.lastDay)

    @property
    def rLast(self):
        """
        Property: The last (final) growth rate from the best-fit model
        parameters. C{None} if there is not yet a best L{Individual}
        in my L{Population} I{p}.
        """
        r = None
        i = self.p.best()
        if i is None: return
        for name in sorted(self.names):
            if name.startswith('r'):
                r = i[name]
        return r

    @property
    def tValues(self):
        """
        Property: A sequence of 2-tuples, with names and values of the
        best-fit model's 't' parameters, e.g., I{t1} and I{t2}.

        An empty sequence if there is not yet a best L{Individual} in
        my L{Population} I{p}.
        """
        result = []
        i = self.p.best()
        if i is not None:
            for name in sorted(self.names):
                if self.re_tn.match(name):
                    result.append((name, i[name]))
        return result
    
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
        Xc = Xc.flatten()
        if self.pct:
            f = ax.plot
            Xc = self.make_pct(Xc)
        else:
            f = ax.semilogy
            Xc = self.clipLower(Xc)
        f(tc, Xc, color='black', marker=',', linestyle='')

    @staticmethod
    def cases(X, k):
        return int(round(X[k]))

    def make_pct(self, X):
        """
        Returns a version of I{X} scaled to a percentage of my region's
        population.
        """
        return 100.0*X/self.ev.pop

    def weeklyTicks(self, sp, t0):
        """
        Adds major x-axis tick marks that indicate week boundaries from
        "today".
        """
        sp.set_tickSpacing('x', (7.0, t0), 1.0)
    
    def annotate_past(self, sp, X, k, in_paren=None):
        """
        Given a 1-D array I{X} of case numbers that will be displayed as
        the first line in the supplied subplot I{sp}, adds an
        annotation for the actual number I{k} days after first report.

        The last item in array I{X} must be the most recent actual
        number of cases.
        """
        # Length of X, just actual data to be plotted up to most
        # recent
        N = len(X)
        # Convert index from all data to just data to be plotted
        kX = k - (len(self.ev) - N)
        if kX >= N: return
        if kX < 0: return
        dateText = self.ev.dayText(k)
        if in_paren:
            dateText = sub("{} ({})", dateText, in_paren)
        parts = [sub("{}:", dateText)]
        if self.pct:
            parts.append(sub("{:.2f}%", X[kX]))
        else: parts.append(sub("{:,.0f}", self.cases(X, kX)))
        sp.add_annotation(kX, " ".join(parts))
    
    def model_past(self, sp, values, past=False):
        """
        Plots the past data against the best-fit model in subplot I{sp},
        given the supplied parameter I{values}, with the model curve
        anchored at the right to today's actual reported cases.

        Returns a 4-tuple of equal-length 1-D arrays with (1) the
        actual time I{t}, (2) the actual cumulative numbers of cases
        I{X_data}, (3) the actual numbers of new daily cases, and (4)
        the modeled number of new daily cases.
        """
        def tb(*args):
            sp.add_textBox('S', *args)

        def plot_stuff():
            ea = ErrorAnnotator(self.ev, X_curve, X_data, kToday)
            # Date of first reported case number plotted
            self.annotate_past(sp, X_data, k0+1)
            # Date of most recently reported case number plotted
            self.annotate_past(sp, X_data, kToday)
            # Date of any t values in the plot x-axis range
            tLast = 0
            names = []
            for name, value in self.tValues:
                tLast += value
                if tLast > t[0] and tLast < t[-1]:
                    names.append(name)
                    self.annotate_past(sp, X_data, int(round(tLast)), name)
            # Error in expected vs actual reported cases, going back
            # several days starting with the latest date
            kList = range(kToday, k0, -1)
            N_data = len(X_data)
            sp.add_axvline(-1)
            for kk, k in enumerate(kList):
                k_data = N_data - kk - 1
                ea.add(k)
            text = sub(
                "Reported cases (NY Times data) vs days after first. {}",
                ea.annotate(sp))
            names = ["first", "last"] + names
            text = sub("{}; {} date totals.", text, ", ".join(names))
            for line in textwrap.wrap(text, self.max_line_length):
                tb(line)
            self.weeklyTicks(sp, t[-1])
            sp.use_minorTicks('y', 10)
            ax = sp.semilogy(t, X_data)
            # Add the best-fit model values for comparison
            self.add_model(ax, t, X_curve, semilog=True)
        
        k0 = self.ev.k0
        kToday, k_offset = self.kForToday
        r = self.curvePoints(values, kToday, k0)
        t, X_curve = r.t, r.X
        t -= k_offset
        K = self.ev.kt(t)
        X_data = self.clipLower(self.ev.X[K])
        XD = self.ev.transform()[K]
        if past: plot_stuff()
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
            if self.pct:
                X = X_curve[k_curve]
                proto = "{}: {:.2f}%"
            else:
                X = self.cases(X_curve, k_curve)
                proto = "{}: {:,.0f}"
            sp.add_annotation(
                k_curve, proto,
                self.ev.dayText(k0+daysInFuture), X, kVector=1)
            return True

        def plot():
            sp.add_line('-', 2)
            # Vertical line "today"
            t0 = self.ev.t[-1]
            sp.add_axvline(t0)
            # "Today" + previous few days
            for k in range(k0-N_back, k0+1):
                annotate_past(k)
            # Every day for the next week, if daysForward is small enough
            for k in range(1, 8, 1 if self.daysForward < 22 else 2):
                annotate_future(k)
            # Every other day thereafter
            for k in range(8, self.daysForward-1, 2):
                if not annotate_future(k):
                    break
            self.weeklyTicks(sp, t0)
            sp.use_minorTicks('y', 10)
            # Start with a few of the most recent actual data points
            if self.pct:
                ax = sp(t_data, X_data)
            else: ax = sp.semilogy(t_data, X_data)
            # Add the best-fit model extrapolation
            self.add_model(ax, t, X_curve, semilog=not self.pct)
            # Add scatterplot sorta-probalistic predictions
            self.add_scatter(ax, k0, k1, k_offset)
        
        N_back = 4
        k0, k_offset = self.kForToday
        k1 = k0 + self.daysForward
        t_data, X_data = [
            getattr(self.ev, name)[-N_back:] for name in ('t', 'X')]
        if self.pct:
            if self.includeFuture: sp.set_zeroLine(1.0)
            X_data = self.make_pct(X_data)
        r = self.curvePoints(values, k0, k1)
        t = r.t - k_offset
        X_curve = r.X
        if self.pct: X_curve = self.make_pct(X_curve)
        if self.includeFuture: plot()
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
    
    def residuals(self, values, sp=None):
        """
        Computes residuals between case numbers modeled with the supplied
        parameter I{values} and actual case numbers. Computes and logs
        info about the goodness of fit.

        If a subplot I{sp} is supplied, puts the info on the plot as a
        textbox.

        Returns a L{Results} object from the residual calculation. If
        the calculation results in C{None}, no fit info is logged or
        plotted and C{None} is returned.
        """
        def tb(*args):
            msg(*args)
            if sp: sp.add_textBox(self.pos('stats'), *args)

        # Calculate residuals between modeled and actual case numbers,
        # using supplied parameter values
        r = self.ev.residuals(values)
        if r is None: return
        # Significance of non-normality
        tb("Non-normality: p = {:.4f}", stats.normaltest(r.R)[1])
        # AICc
        AICc, N, k = self.AICc(r)
        tb("AICc is {:+.2f} with SSE={:.5g}, N={:d}, k={:d}",
           AICc, r.SSE, N, k)
        return r
        
    def subplot_upper(self, sp, values, past=False):
        """
        Does the computations for the upper subplot with model fit vs
        data, only drawing the subplot if I{past} is C{True}.

        Returns the I{t} I{X} vectors for the actual data being
        plotted, the I{XD} vector for the actual data, the I{X} vector
        for the modeled data, and the I{XD} vector for the modeled
        data.
        """
        def tb(name, value):
            sp.add_textBox(self.pos('params'), "{}: {:.5g}", name, value)

        if past:
            sp.add_line('-', 2)
            sp.add_textBox(
                self.pos('model'), self.ev.model.f_text)
            tb('L', self.ev.L)
            for k, name in enumerate(self.names):
                tb(name, values[k])
        # Data vs best-fit model
        return self.model_past(sp, values, past)

    def subplot_middle(self, sp, values, t):
        """
        Does a middle subplot with residuals between the data I{X_data}
        and modeled data I{X_curve}, given model parameter I{values}
        and evaluation times (days after 1/21/20) I{t}.
        """
        r = self.residuals(values, sp)
        if r is None: return
        sp.set_tickSpacing('x', True, False)
        sp.set_ylabel("Residual")
        sp.set_xlabel("Modeled (fitted) new cases/day (square-root transform)")
        sp.set_zeroLine(color="red", linestyle='-', linewidth=3)
        sp.add_line('')
        sp.add_marker('o', 4)
        sp.set_colors("purple")
        sp.add_textBox(
            'NW', "Residuals: Modeled vs reported new cases/day (transformed)")
        sp(r.XD, r.R, zorder=3)
        
    def subplot_ratio(self, sp, ta, Ra, tm, Rm, rLast=None):
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
            "New cases each day (%)")
        sp.set_ylabel("% New")
        if rLast: sp.set_zeroLine(100*rLast)
        self.weeklyTicks(sp, ta[-1])
        ax = sp(ta, Ra)
        self.add_model(ax, tm, Rm)

    def subplot_daily(self, sp, ta, XDa, tm, XDm):
        """
        Draws an optional subplot below the main ones showing the number
        of new daily cases, both past (actual) and future (modeled).

        Call with actual-data time and daily-case vectors I{ta},
        I{XDa} and modeled time and daily-case vectors I{tm}, I{XDm}.

        B{TODO}: If my I{pct} is set C{True}, the plot is for the
        number of new cases per 100,000 people in the region's
        population.
        """
        sp.add_axvline(-1)
        sp.add_annotation(
            ta[-1], "{} had {:.0f} new case reports",
            self.ev.dayText(), XDa[-1])
        sp.add_textBox(
            self.pos('daily_new'),
            "New cases each day (N)")
        sp.set_ylabel("Newly Reported")
        self.weeklyTicks(sp, ta[-1])
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
        self.pt.set_xlabel("Days after January 22, 2020")
        self.pt.use_minorTicks('x', 1.0)
        with self.pt as sp:
            ta, Xa, XDa, Xam, XDam = self.subplot_upper(
                sp, values, self.includePast)
            if self.includePast:
                self.subplot_middle(sp, values, ta)
            if self.pct:
                tb("Expected % of population vs days after first case. Dots are")
            else:
                tb("Expected cases reported vs days after first case. Dots are")
            tb("daily model projections for each of a final population of")
            tb("{:d} evolved parameter combinations. Annotations:", len(self.p))
            tb("Reported cases, then best-fit projection.")
            for line in DISCLAIMER.strip().split('\n'):
                sp.add_textBox(self.pos('disclaimer'), line)
            tm, Xm, XDm = self.model_future(sp, values)
            if self.includeRatio or self.includeDaily:
                tm = np.concatenate([ta, tm])
                Xm = np.concatenate([Xam, Xm])
                XDm = np.concatenate([XDam, XDm])
                if self.includeRatio:
                    self.subplot_ratio(
                        sp, ta, XDa/Xa, tm, XDm/Xm, self.rLast)
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
    max_plot_SSE_ratio = 1.2
    
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
        if len(iChanged) > 0.5*len(self.p):
            # The amount of SSE changing indicates that something
            # fundamental has changed, so restart history with
            # re-evaluated population. Unfortunately, all the history
            # under the old SSE-evaluation regime is no longer
            # applicable and will be lost.
            yield self.p.history.clear()
        for i in self.p:
            yield self.p.history.add(i)
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
            basename = sanitized(state, county)
            return sub("{}-N{:d}.dat", basename, len(self.names))
        
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
        reporter = Reporter(
            self, args.f, pct=args.c, past=not args.P,
            future=not args.T, ratio=not args.R, daily=not args.D)
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
        values = self.p.best().values
        reporter.residuals(values)
        savePicklePath = picklePath()
        self.p.save(savePicklePath)
        msg(0, "Saved final population of best parameter combinations "+\
            "to {}", savePicklePath)
        msg("Elapsed time: {:.2f} seconds", time.time()-startTime, 0)
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
args('-f', '--days-forward', 16,
     "Extrapolation limit, days from latest data")
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
args('-P', '--exclude-past',
     "Exclude subplots with past cases and curve fit info")
args('-T', '--exclude-future', "Exclude subplot with extrapolations")
args('-R', '--exclude-ratio',
     "Exclude subplot with ratio of new vs cumulative cases (implied by -c)")
args('-D', '--exclude-daily',
     "Exclude subplot with new daily cases")
args('-c', '--percentage',
     "Show cumulative case numbers as percentage of population (implies -R)")
args('-i', '--profile', "Run with the Python profiler (slower)")
args("[<State Name> [<County Name>]]")
args(main)
