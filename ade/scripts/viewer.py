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
A simple GUI to visualize a population history.
"""

from __future__ import with_statement
from __future__ import division

# Enable this only if you want to infect your code with the GPL or you
# have the commercial version of PyQt. ADE is not GPL licensed and is
# not released with this enabled.
USE_PYQT = False

# Adapted from
# https://github.com/epage/PythonUtils/blob/master/util/qt_compat.py
# -----------------------------------------------------------------------------
if USE_PYQT:
    import sip
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)
    import PyQt4.QtCore as _QtCore
    QtCore = _QtCore
    USES_PYSIDE = False
else:
    import PySide.QtCore as _QtCore
    QtCore = _QtCore
    USES_PYSIDE = True

def _pyside_import_module(moduleName):
    pyside = __import__('PySide', globals(), locals(), [moduleName], -1)
    return getattr(pyside, moduleName)

def _pyqt4_import_module(moduleName):
    pyside = __import__('PyQt4', globals(), locals(), [moduleName], -1)
    return getattr(pyside, moduleName)

if USES_PYSIDE:
    import_module = _pyside_import_module
    Signal = QtCore.Signal
    Slot = QtCore.Slot
    Property = QtCore.Property
    SIGNAL = QtCore.SIGNAL
#else:
#    import_module = _pyqt4_import_module
#    Signal = QtCore.pyqtSignal
#    Slot = QtCore.pyqtSlot
#    Property = QtCore.pyqtProperty
#    SIGNAL = QtCore.SIGNAL
# -----------------------------------------------------------------------------

import numpy as np
from matplotlib.backends.backend_qt4agg \
    import FigureCanvasQTAgg as FigureCanvas
QtGui = import_module('QtGui')
QSizePolicy = QtGui.QSizePolicy

from yampex import plot

from ade.population import Population
from ade.util import *


class GuiPlotter(plot.Plotter, QtGui.QWidget):
    """
    I do the work of C{yampex.plot.Plotter} in a PyQt/PySide window.
    """
    def __init__(self, parent, *args, **kw):
        plot.Plotter.__init__(self, *args, **kw)
        QtGui.QWidget.__init__(self, parent)
        pane = QtGui.QVBoxLayout(self)
        pane.setSpacing(0)
        pane.setContentsMargins(0, 0, 0, 20)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fc = FigureCanvas(self.fig)
        self.fc.setSizePolicy(sp)
        self.fc.updateGeometry()
        pane.addWidget(self.fc)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def __del__(self):
        pass
        
    def sizeHint(self):
        return QtCore.QSize(*[self.DPI*x for x in self.figSize])


class Window(QtGui.QMainWindow):
    """
    I am the main window for the parameter viewer.
    """
    sb = Signal()
    sq = Signal()

    def __init__(self, parent=None, runner=None):
        super(Window, self).__init__(parent)
        self.r = runner
        self.setWindowTitle("ADE Parameter Viewer")
        self.create_main_frame(parent)

    def do_select(self):
        for name in self.rb:
            if self.rb[name].isChecked():
                return self.r.update(name)

    def create_main_frame(self, parent):
        self.main_frame = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        
        # Left pane
        # -------------------------------------------------------
        pane = QtGui.QVBoxLayout()
        # Parameter radio buttons
        self.rb = {}
        for name in self.r.history.names:
            self.rb[name] = rb = QtGui.QRadioButton(name, self)
            self.connect(rb, SIGNAL('toggled()'), self.do_select)
            pane.addWidget(rb)
        # Quit Button
        qb = QtGui.QPushButton("&Quit")
        self.connect(qb, SIGNAL('clicked()'), self.r.shutdown)
        pane.addWidget(qb)

        # Right pane
        # -------------------------------------------------------
        self.pt = GuiPlotter(parent, 1)
        hbox.addWidget(self.pt)
        # -------------------------------------------------------

        self.main_frame.setLayout(hbox)
        self.setCentralWidget(self.main_frame)
        self.sq.connect(self.shutdownNow)
        
    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Q:
                self.sq.emit()
        return super(Window, self).eventFilter(source, event)

    @Slot()
    def shutdownNow(self):
        self.r.shutdown()


class Runner(object):
    """
    """
    def __init__(self, history):
        self.app = QtGui.QApplication([])
        self.history = history
        self.window = Window(parent=None, runner=self)
        self.pt = self.window.pt

    def update(self, name):
        print "UPDATE", name

    def shutdown(self):
        self.app.exit()

    def run(self):
        self.app.installEventFilter(self.window)
        self.window.show()
        self.app.exec_()


        
if __name__ == '__main__':
    p = Population.load("~/pfinder.dat")
    Runner(p.history).run()
