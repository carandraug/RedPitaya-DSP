# Copyright (C) 2015 Tom Parks <thomasparks@outlook.com>
# Copyright (C) 2017-2018 Tiago Susano Pinto <tiagosusanopinto@gmail.com>
#
# RedPitaya-DSP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RedPitaya-DSP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RedPitaya-DSP. If not, see <http://www.gnu.org/licenses/>.

#! /usr/bin/python

import Pyro4
import subprocess
import threading
import os
import time
from mmap import mmap
import struct

import logging
import traceback
import sys

def bin(s):
    ''' Returns the set bits in a positive int as a str.'''
    return str(s) if s<=1 else bin(s>>1) + str(s&1)

def busy_wait(dt):
    '''Wait without sleeping for higher accuracy.'''
    current_time = time.time()
    while (time.time() < current_time+dt):
        pass


from types import FunctionType
from functools import wraps

def printfunc(func):
    @wraps(func)
    def wrapped(*args, **kwrds):
        print(func)
        try:
            return func(*args, **kwrds)
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e
    return wrapped

class PrintMetaClass(type):
    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if type(attribute) == FunctionType:
                # replace it with a wrapped version
                attribute = printfunc(attribute)
            newClassDict[attributeName] = attribute
        return type.__new__(meta, classname, bases, newClassDict)

# RedPitaya board - writes and reads to the board
class Board(object):
    offsetInit = 0x40000000
    offsetSize = 0x4022FFFF - offsetInit
    offsets = {                     # specific memory offsets
        'direct_pinP'  : 0x10,      # Direction for P lines
        'direct_pinN'  : 0x14,      # Direction for N lines
        'out_pinP'     : 0x18,      # Output P
        'out_pinN'     : 0x1C,      # Output N
        'in_pinP'      : 0x20,      # Input P
        'in_pinN'      : 0x24,      # Input N
        'led'          : 0x30,      # LED control
        'asg_channelA' : 0x210000,  # Output channel A (out1)
        'asg_channelB' : 0x220000   # Output channel B (out2)
    }

    def __init__(self):
        with open('/dev/mem', 'r+b') as f:
            self.mem = mmap(f.fileno(), self.offsetSize,
                offset = self.offsetInit)

    def read(self, offset, safe = True):
        if safe and offset not in Board.offsets.values():
            raise NameError('offset {0:#x} not implemented'.format(offset))
        return struct.unpack('<l', self.mem[offset:offset+4])[0]

    def write(self, offset, value, safe = True):
        if safe and offset not in Board.offsets.values():
            raise NameError('offset {0:#x} not implemented'.format(offset))
        self.mem[offset:offset+4] = struct.pack('<l', value)

# Creates the actionTable file, and runs the C-code
class Runner(object):
    def __init__(self):
        self.pid = None
        self.filename = None
        self.execRunnerFile = 'build/bin/dsp'
        self.actionTablesDirectory = 'actionTables/'
        self.writtenActionTable = False
        self.running = None

    def loadDemo(self, deltaTime):
        self.filename = self.actionTablesDirectory + 'actTblTest.txt'
        with open(self.filename, 'w') as f:
            demoDigitalPin = 5
            demoAnalogOutput = -1
            maxOfLines = 100000
            time = 0
            for line in range(maxOfLines):
                print('{} {} {}'.format(time, demoDigitalPin, line%2), file=f)
                print('{} {} {}'.format(time,
                    demoAnalogOutput, line*8000./maxOfLines), file=f)
                # print('{} {} {}\n{} {} {}'.format(time, demoDigitalPin, line%2,
                #     time, demoAnalogOutput, line*8000./maxOfLines), file=f)
                time += deltaTime*1e3
        self.writtenActionTable = True

    def load(self, actiontable, name = None):
        name = name if name else 'actTbl'
        self.filename = (self.actionTablesDirectory + str(name) +
            time.strftime('-%Y%m%d-%H:%M:%S') + '.txt')
        try:
            with open(self.filename, 'w') as f:
                print('actiontable is', actiontable)
                print('action table was that')
                for row in actiontable:
                    actTime, pinNumber, actionValue = row
                    timeNanoSec = int(actTime*1e3) # convert to ns
                    finalRow = (timeNanoSec, pinNumber, actionValue)
                    print('time:{} pin:{} action:{}'.format(*finalRow))
                    print('{} {} {}'.format(*finalRow), file=f)
            self.writtenActionTable = True
        except Exception as caughtException:
            print(caughtException)

    def abort(self):
        if self.pid:
            subprocess.call(['kill', '-9', str(self.pid)])
            self.pid = None

    def start(self):
        if self.writtenActionTable:
            comandLine = [self.execRunnerFile, self.filename]
            print('calling {}'.format(comandLine))
            process = subprocess.Popen(comandLine)
            self.pid = process.pid
            return process
        else:
            raise Exception('Please write the action table!')

    def stop(self):
        self.abort()

# Combines the Runner and the Board
@Pyro4.expose
class Executor(object):
    __metaclass__ = PrintMetaClass
    def __init__(self):
        self.clientConnection = None
        self.runner = Runner()
        self.board = Board()

        # single value output
        # self.board.asga.counter_wrap = self.board.asga.counter_step
        # self.board.asgb.counter_wrap = self.board.asgb.counter_step

        # set all pins to out
        self.board.write(Board.offsets['direct_pinP'], 0xFF)
        self.board.write(Board.offsets['direct_pinN'], 0xFF)

    def Abort(self):
        # kill the server process
        self.runner.abort()
        # The RedPitaya-DSP has a handler for SIGINT that cleans up

    def arcl(self, cameras, lightTimePairs):
        if lightTimePairs:
            # Expose all lights at the start, then drop them out
            # as their exposure times come to an end.
            # Sort so that the longest exposure time comes last.
            lightTimePairs.sort(key = lambda a: a[1])
            curDigital = cameras + sum([p[0] for p in lightTimePairs])
            self.WriteDigital(curDigital)
            print('Start with {}'.format(curDigital))
            totalTime = lightTimePairs[-1][1]
            curTime = 0
            for line, runTime in lightTimePairs:
                # Wait until we need to open this shutter.
                waitTime = runTime - curTime
                if waitTime > 0:
                    busy_wait(waitTime/1000.)
                curDigital -= line
                self.WriteDigital(curDigital)
                curTime += waitTime
                print('At {} set {}'.format(curTime, curDigital))
            # Wait for the final timepoint to close shutters.
            if totalTime - curTime:
                busy_wait( (totalTime - curTime)/1000. )
            print('Finally at {} set {}'.format(totalTime, 0))
            self.WriteDigital(0)
        else:
            self.WriteDigital(cameras) # "expose"
            self.WriteDigital(0)

    # Volts can be between -1 to 1 [-8192 to 8191]
    def convertVoltsToADUs(volts, maxVoltage = 1):
        if volts > 0:
            return int(volts*8191/maxVoltage)
        return int(volts*8192/maxVoltage)

    # expect value in analog-to-digital-units (ADUs)
    def MoveAbsoluteADU(self, channel, value):
        if channel == 0:
            self.board.write(Board.offsets['asg_channelA'], value)
        elif channel == 1:
            self.board.write(Board.offsets['asg_channelB'], value)
        else:
            raise Exception('Unexpected analog channel! (0 or 1)')

    def ReadPosition(self, channel):
        if channel == 0:
            return int(self.board.read(Board.offsets['asg_channelA']))
        elif channel == 1:
            return int(self.board.read(Board.offsets['asg_channelB']))
        else:
            raise Exception('Unexpected analog channel! (0 or 1)')

    def WriteDigital(self, value):
        dP = value & int('11111111', 2)
        dN = (value & int('1111111100000000', 2)) >> 8
        self.board.write(Board.offsets['out_pinP'], dP)
        self.board.write(Board.offsets['out_pinN'], dN)

    def ReadDigital(self, line=0):
        if line == 0:
            return self.board.read(Board.offsets['out_pinP'])
        elif line == 1:
            return self.board.read(Board.offsets['out_pinN'])
        else:
            raise Exception('Unexpected digital pin line! (0 or 1)')

    def writeLed(self, leds):
        self.board.write(Board.offsets['led'], leds)

    def readLed(self, leds):
        self.board.read(Board.offsets['led'])

    def InitProfile(self, numReps):
        # I'm pretty sure this does not zero the prev values.
        # is it for allocating space?
        # self.times, self.digitals, self.analogA, self.analogB = [], [], [], []
        pass

    def profileSet(self, profileStr, digitals, *analogs):
        print("profileset called with")
        print("analog0", analogs[0],
              "times", list(zip(*analogs[0]))[0],
              "vals", list(zip(*analogs[0]))[1])
        # This is downloading the action table
        # digitals is numpy.zeros((len(times), 2), dtype = numpy.uint32),
        # starting at 0 -> [times for digital signal changes, digital lines]
        # analogs is a list of analog lines and the values to put on them at each time
        # digitals = list of lists. sublist is a time, line pair
        # then 4 analog lines. also list of time: value pairs.
        Dtimes, Dvals = zip(*digitals)
        if len(analogs) > 0:
            Atimes, Avals = zip(*analogs[0])
        else:
            Atimes, Avals = [], []
        if len(analogs) > 0:
            Btimes, Bvals = zip(*analogs[1])
        else:
            Btimes, Bvals = [], []

        Dtimes = list(Dtimes)
        Atimes = list(Atimes)
        Btimes = list(Btimes)

        print("dt:{},\n at:{},\n bt:{}".format(Dtimes, Atimes, Btimes))
        print()
        print()
        print("dv:{},\n av:{},\n bv:{}".format(Dvals, Avals, Bvals))

        times, digitals, analogA, analogB = [], [], [], []
        times = sorted(set(Dtimes+Atimes+Btimes))
        for timepoint in times:
            for outline, inval, timesForLine in [(digitals, Dvals, Dtimes),
                                                 (analogA,  Avals, Atimes),
                                                 (analogB,  Bvals, Btimes)]:
                if timepoint in timesForLine:
                    outline.append(inval[timesForLine.index(timepoint)])
                else:
                    prevValue = outline[-1] if outline else 0
                    outline.append(prevValue) # the last value

        # the DSP 'baselines' the analog values the the first value in the analogs
        # when an experiment is started. We don't do this, we just write the values
        # so add the first analog value back on
        analogA = [analogAi+analogA[0] for analogAi in analogA]
        analogB = [analogBi+analogB[0] for analogBi in analogB]

        print("sort")
        self.actiontable = sorted(zip(times, digitals, analogA, analogB))
        print("sorted")

    def DownloadProfile(self, name = None): # This saves the action table
        self.runner.load(self.actiontable, name)

    def trigCollect(self, wait = True):
        process = self.runner.start()
        if wait:
            process.wait()
        if self.clientConnection:
            retVal = (100, [self.ReadPosition(0), self.ReadPosition(1), 0, 0])
            self.clientConnection.receiveData('DSP done', retVal)
        # needs to block on the dsp finishing.

    def demo(self, dt):
        self.runner.loadDemo(dt)
        process = self.runner.start()
        process.wait()

    def receiveClient(self, uri):
        self.clientConnection = Pyro4.Proxy(uri)
        print(uri)

# Exposes Executor, through Pyro4
class Server(object):
    def __init__(self, execHost, execPort, execName = 'redPitaya'):
        print('Started server at {}'.format(
            time.strftime('%A, %B %d, %I:%M %p')))
        executor = Executor()

        Pyro4.config.SERIALIZER = 'pickle'
        Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
        while True:
            try:
                dspDaemon = Pyro4.Daemon(host = execHost, port = execPort)
                break
            except Exception as e:
                print('Socket fail {}'.format(e))
                time.sleep(1)

        print('Providing executor as [{}] at {}'.format(execName,
            dspDaemon.locationStr))
        Pyro4.Daemon.serveSimple({executor: execName},
            daemon = dspDaemon, ns = False, verbose = True)


if __name__ == '__main__':
    pyroHost = '192.168.0.20'
    pyroPort = 8005
    Server(pyroHost, pyroPort)
