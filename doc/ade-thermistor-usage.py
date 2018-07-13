#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path, atexit
from datetime import datetime

import numpy as np

from yoctolib.yocto_api import *
from yoctolib.yocto_genericsensor import *

import wiringpi as wpi


# Location of the log file
LOG = "~/therm.csv"

# All temperatures in Degrees C
# ----------------------------------------------------------------
# Nominal inside temperature
T0 = 20
# Temperature of inside above outside before fan goes on, at T0
TEMPDIFF_T0 = 3.5
# Hysteresis in temperature difference
TEMPDIFF_H = 1.0
# Minimum inside temp to ever request fan
INSIDE_MIN = 10
# Maximum inside temp to ever not request fan, so long as outside
# is not warmer
INSIDE_MAX = 45

# All times in seconds
# ----------------------------------------------------------------
# Interval between voltage readings, in seconds
INTERVAL = 2.0
# Time constant for single-stage IIR LPF
TC = 90.0
# Minimum fan transition time (to avoid overly rapid cycling)
TRANSITION_TIME = 5*60

# Number of voltage readings per fan decision and log entry
N_MINUTES = 10
N_READINGS = 60*N_MINUTES / INTERVAL

# GPIO for fan control
FAN_GPIO = 27


def sub(proto, *args):
    """
    This really should be a built-in function.
    """
    return proto.format(*args)


class IIR(object):
    """
    First-order IIR LPF section to filter a [V1, V2] sample.
    """
    N_settle = 10000
    
    def __init__(self, ts, tc):
        """
        Call with just desired output of filter to settle to that input
        and output. If setting up for the first time, also define the
        filter time constant I{tc}.
        """
        self.a = np.exp(-ts / tc)
        self.y = None
    
    def __call__(self, x, no_settle=False):
        if self.y is None:
            self.y = np.zeros(2)
            for k in range(self.N_settle):
                self(x, no_settle=True)
        self.y = np.array(x) + self.a*self.y
        return (1.0-self.a) * self.y


class Runner(object):
    """
    I monitor inside and outside temperatures and turn on power to a
    ventilation fan when running it would cool the inside
    meaningfully.
    """
    # See the ade package, examples/thermistor.log, git commit c47ec4
    a0 = np.array([0.370463, 0.370874])
    a1 = np.array([19.9521,  19.1129])
    Rs = np.array([13.8721,  14.1048])
    Vp = 23.463

    dtFormat = "%Y%m%d %H:%M"
    
    def __init__(self):
        logFile = os.path.expanduser(LOG)
        mode = 'a' if os.path.exists(logFile) else 'w'
        self.fh = open(logFile, mode)
        self.iir = IIR(INTERVAL, TC)
        self.readings = []
        errmsg = YRefParam()
        YAPI.RegisterHub("usb", errmsg)
        self.sensors = [YGenericSensor.FindSensor(sub(
            "RX010V01-7E29D.genericSensor{:d}", k+1)) for k in range(2)]
        atexit.register(self.shutdown)
        self.fanState = datetime.datetime.now(), None
        self.gpioSetup()

    def shutdown(self):
        self.fh.close()

    def gpioSetup(self):
        """
        Sets up the GPIO to control the fan.
        """
        wpi.wiringPiSetupGpio()
        wpi.pinMode(FAN_GPIO, 1)
    
    def logLine(self, proto, *args):
        line = proto.format(*args) + '\n'
        self.fh.write(line)
        self.fh.flush()

    def v2t(self, V):
        """
        Given a 1-D vector of voltages from the RX010V01, returns a 1-D
        vector of temperatures (degrees C) for those voltages.

        Each voltage is from a voltage divider connected to the
        following series-connected nodes: (0) ground, (1) one of the
        RX010V01 inputs, and (2) the 23V supply (Vp) of the RX010V01,
        somewhat filtered with an electrolytic capacitor.

        Between (1) and (0) there is the input impedance (11.2K) of
        the RX010V01 input in parallel with a 59K resistor and a 0.1
        uF capacitor. Between (1) and (2) there is an NTC
        thermistor.

        The higher the temperature, the lower the resistance of the
        thermistor and thus the higher the measurement voltage at (1).

        Curve fitting of actual observations found a slightly
        different value of Vp than 23V.
        """
        return self.a1 * np.log(self.Rs*V / (self.a0*(self.Vp-V)))
        
    def getTemps(self):
        """
        Returns a 2-element array of the current (lowpass-filtered)
        temperatures, outside and inside.

        The temperature readings are computed from voltages reported
        by the YoctoPuce RX010V01.

        If either voltage sensor reads C{None}, the readings are
        discarded. Temperature readings are accumulated and C{None} is
        returned until the number of readings per temp output is
        reached. Then the voltage readings are averaged (computed
        temperatures, not voltages) and the average is returned.
        """
        V = [self.sensors[k].get_currentValue() for k in range(2)]
        if None in V:
            return
        V = self.iir(V)
        self.readings.append(self.v2t(V))
        if len(self.readings) < N_READINGS:
            return
        T = np.zeros(2)
        for reading in self.readings:
            T += reading
        T /= len(self.readings)
        self.readings = []
        return T

    def setFan(self, on=False):
        """
        Turns the fan on if C{True} is supplied as an argument, otherwise
        off. The fan state is only altered if doing so wouldn't
        violate the mininimum fan transition time.

        Returns the current I{datetime} object and the actual fan state.
        """
        dt = datetime.datetime.now()
        is_on = self.fanState[1]
        if on == is_on:
            return dt, on
        if is_on is not None:
            td = dt - self.fanState[0]
            if td.total_seconds() < TRANSITION_TIME:
                return dt, is_on
        self.fanState = dt, on
        wpi.digitalWrite(FAN_GPIO, int(on))
        return self.fanState

    def tempDiff(self, inside):
        """
        Returns the temperature difference of inside above outside before
        the fan turns on, given the I{inside} temperature. The
        required difference increases as inside temperature goes down.

        The minimum value returned is zero; the fan will never turn on
        if it's hotter outside than inside.
        """
        if not hasattr(self, '_td_coeffs'):
            m = -float(TEMPDIFF_T0) / (INSIDE_MAX - T0)
            b = TEMPDIFF_T0 - m*T0
            self._td_coeffs = m, b
        m, b = self._td_coeffs
        return max([0, m*inside + b])
    
    def decideFan(self, outside, inside):
        """
        Returns C{True} if the fan should be on, given the I{outside} and
        I{inside} temperatures.
        """
        if self.fanState[1]:
            # Fan is on, so add hysteresis to inside temperature to
            # reduce likelihood of switching off
            inside += TEMPDIFF_H
        if inside < INSIDE_MIN:
            # This is plenty cool enough, no fan.
            return False
        td = self.tempDiff(inside)
        return inside-outside > td

    def run(self):
        """
        Where my processing sits, repeatedly polling the voltage sensors and
        adjusting the fan state accordingly.
        """
        def isOnline():
            for sensor in self.sensors:
                if not sensor.isOnline():
                    self.setFan(False)
                    return False
            return True
        
        self.setFan(False)
        # Enter the event loop, which is exited only by a kill/term signal.
        while True:
            # Wait the interval between iterations, in a try-except
            # block because kill/term signal raises an exception
            try:
                YAPI.Sleep(int(1000*INTERVAL))
            except:
                break
            if not isOnline():
                continue
            temps = self.getTemps()
            if temps is None:
                continue
            fan_on = self.decideFan(temps[0], temps[1])
            dt, is_on = self.setFan(fan_on)
            self.logLine(
                "{} {:04.1f} {:04.1f} {:d}",
                dt.strftime(self.dtFormat), temps[0], temps[1], is_on)


def run():
    import pdb, traceback, sys
    try:
        Runner().run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

            
if __name__ == "__main__":
    run()
