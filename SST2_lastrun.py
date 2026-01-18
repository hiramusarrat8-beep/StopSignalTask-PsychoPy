#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on January 18, 2026, at 18:53
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'SST2'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'Age': '',
    'Gender': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 720]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\91826\\Desktop\\SST2\\SST2_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1,1,1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_inst') is None:
        # initialise key_resp_inst
        key_resp_inst = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    Welcome_text = visual.TextStim(win=win, name='Welcome_text',
        text='"WELCOME"',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction" ---
    Inst_text = visual.TextStim(win=win, name='Inst_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_inst = keyboard.Keyboard(deviceName='key_resp_inst')
    
    # --- Initialize components for Routine "fixation" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='images/fix.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Stimulus" ---
    go_arrow = visual.ImageStim(
        win=win,
        name='go_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    stop_arrow = visual.ImageStim(
        win=win,
        name='stop_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from SSD_code
    SSD = 0.20       # starting SSD (200 ms)
    SSD_step = 0.05
    SSD_min = 0.05
    SSD_max = 0.9
    MAXRT = 1.25
    
    
    # Run 'Begin Experiment' code from block_fb_code
    # ---- BLOCK COUNTERS ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    
    stop_trials = 0
    correct_cancels = 0
    
    
    
    # --- Initialize components for Routine "Feedback" ---
    fdback_text = visual.TextStim(win=win, name='fdback_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "block_feedback" ---
    fb_text = visual.TextStim(win=win, name='fb_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "fixation" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='images/fix.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Stimulus" ---
    go_arrow = visual.ImageStim(
        win=win,
        name='go_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    stop_arrow = visual.ImageStim(
        win=win,
        name='stop_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from SSD_code
    SSD = 0.20       # starting SSD (200 ms)
    SSD_step = 0.05
    SSD_min = 0.05
    SSD_max = 0.9
    MAXRT = 1.25
    
    
    # Run 'Begin Experiment' code from block_fb_code
    # ---- BLOCK COUNTERS ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    
    stop_trials = 0
    correct_cancels = 0
    
    
    
    # --- Initialize components for Routine "block_feedback" ---
    fb_text = visual.TextStim(win=win, name='fb_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "fixation" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='images/fix.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Stimulus" ---
    go_arrow = visual.ImageStim(
        win=win,
        name='go_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    stop_arrow = visual.ImageStim(
        win=win,
        name='stop_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from SSD_code
    SSD = 0.20       # starting SSD (200 ms)
    SSD_step = 0.05
    SSD_min = 0.05
    SSD_max = 0.9
    MAXRT = 1.25
    
    
    # Run 'Begin Experiment' code from block_fb_code
    # ---- BLOCK COUNTERS ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    
    stop_trials = 0
    correct_cancels = 0
    
    
    
    # --- Initialize components for Routine "block_feedback" ---
    fb_text = visual.TextStim(win=win, name='fb_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "fixation" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='images/fix.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Stimulus" ---
    go_arrow = visual.ImageStim(
        win=win,
        name='go_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    stop_arrow = visual.ImageStim(
        win=win,
        name='stop_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from SSD_code
    SSD = 0.20       # starting SSD (200 ms)
    SSD_step = 0.05
    SSD_min = 0.05
    SSD_max = 0.9
    MAXRT = 1.25
    
    
    # Run 'Begin Experiment' code from block_fb_code
    # ---- BLOCK COUNTERS ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    
    stop_trials = 0
    correct_cancels = 0
    
    
    
    # --- Initialize components for Routine "block_feedback" ---
    fb_text = visual.TextStim(win=win, name='fb_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "fixation" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='images/fix.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "Stimulus" ---
    go_arrow = visual.ImageStim(
        win=win,
        name='go_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    stop_arrow = visual.ImageStim(
        win=win,
        name='stop_arrow', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from SSD_code
    SSD = 0.20       # starting SSD (200 ms)
    SSD_step = 0.05
    SSD_min = 0.05
    SSD_max = 0.9
    MAXRT = 1.25
    
    
    # Run 'Begin Experiment' code from block_fb_code
    # ---- BLOCK COUNTERS ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    
    stop_trials = 0
    correct_cancels = 0
    
    
    
    # --- Initialize components for Routine "block_feedback" ---
    fb_text = visual.TextStim(win=win, name='fb_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "Thank_you" ---
    thankyou_text = visual.TextStim(win=win, name='thankyou_text',
        text='“Thank you for participating.\n\n  The task is now complete.\n\n   Please inform the experimenter.”',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[Welcome_text],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcome_text* updates
        
        # if Welcome_text is starting this frame...
        if Welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcome_text.frameNStart = frameN  # exact frame index
            Welcome_text.tStart = t  # local t and not account for scr refresh
            Welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Welcome_text.started')
            # update status
            Welcome_text.status = STARTED
            Welcome_text.setAutoDraw(True)
        
        # if Welcome_text is active this frame...
        if Welcome_text.status == STARTED:
            # update params
            pass
        
        # if Welcome_text is stopping this frame...
        if Welcome_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Welcome_text.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                Welcome_text.tStop = t  # not accounting for scr refresh
                Welcome_text.tStopRefresh = tThisFlipGlobal  # on global time
                Welcome_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Welcome_text.stopped')
                # update status
                Welcome_text.status = FINISHED
                Welcome_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Welcome.maxDurationReached:
        routineTimer.addTime(-Welcome.maxDuration)
    elif Welcome.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    Trial_loop = data.TrialHandler2(
        name='Trial_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('instructions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(Trial_loop)  # add the loop to the experiment
    thisTrial_loop = Trial_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
    if thisTrial_loop != None:
        for paramName in thisTrial_loop:
            globals()[paramName] = thisTrial_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_loop in Trial_loop:
        currentLoop = Trial_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop:
                globals()[paramName] = thisTrial_loop[paramName]
        
        # --- Prepare to start Routine "Instruction" ---
        # create an object to store info about Routine Instruction
        Instruction = data.Routine(
            name='Instruction',
            components=[Inst_text, key_resp_inst],
        )
        Instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Inst_text.setText(text)
        # create starting attributes for key_resp_inst
        key_resp_inst.keys = []
        key_resp_inst.rt = []
        _key_resp_inst_allKeys = []
        # store start times for Instruction
        Instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Instruction.tStart = globalClock.getTime(format='float')
        Instruction.status = STARTED
        thisExp.addData('Instruction.started', Instruction.tStart)
        Instruction.maxDuration = None
        # keep track of which components have finished
        InstructionComponents = Instruction.components
        for thisComponent in Instruction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction" ---
        # if trial has changed, end Routine now
        if isinstance(Trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != Trial_loop.thisTrial.thisN:
            continueRoutine = False
        Instruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Inst_text* updates
            
            # if Inst_text is starting this frame...
            if Inst_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Inst_text.frameNStart = frameN  # exact frame index
                Inst_text.tStart = t  # local t and not account for scr refresh
                Inst_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Inst_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Inst_text.started')
                # update status
                Inst_text.status = STARTED
                Inst_text.setAutoDraw(True)
            
            # if Inst_text is active this frame...
            if Inst_text.status == STARTED:
                # update params
                pass
            
            # *key_resp_inst* updates
            waitOnFlip = False
            
            # if key_resp_inst is starting this frame...
            if key_resp_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_inst.frameNStart = frameN  # exact frame index
                key_resp_inst.tStart = t  # local t and not account for scr refresh
                key_resp_inst.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_inst, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_inst.started')
                # update status
                key_resp_inst.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_inst.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_inst.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_inst.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_inst.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_inst_allKeys.extend(theseKeys)
                if len(_key_resp_inst_allKeys):
                    key_resp_inst.keys = _key_resp_inst_allKeys[-1].name  # just the last key pressed
                    key_resp_inst.rt = _key_resp_inst_allKeys[-1].rt
                    key_resp_inst.duration = _key_resp_inst_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Instruction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction" ---
        for thisComponent in Instruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Instruction
        Instruction.tStop = globalClock.getTime(format='float')
        Instruction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Instruction.stopped', Instruction.tStop)
        # check responses
        if key_resp_inst.keys in ['', [], None]:  # No response was made
            key_resp_inst.keys = None
        Trial_loop.addData('key_resp_inst.keys',key_resp_inst.keys)
        if key_resp_inst.keys != None:  # we had a response
            Trial_loop.addData('key_resp_inst.rt', key_resp_inst.rt)
            Trial_loop.addData('key_resp_inst.duration', key_resp_inst.duration)
        # the Routine "Instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Trial_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    practice_trial_loop = data.TrialHandler2(
        name='practice_trial_loop',
        nReps=4.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(practice_trial_loop)  # add the loop to the experiment
    thisPractice_trial_loop = practice_trial_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial_loop.rgb)
    if thisPractice_trial_loop != None:
        for paramName in thisPractice_trial_loop:
            globals()[paramName] = thisPractice_trial_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial_loop in practice_trial_loop:
        currentLoop = practice_trial_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial_loop.rgb)
        if thisPractice_trial_loop != None:
            for paramName in thisPractice_trial_loop:
                globals()[paramName] = thisPractice_trial_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[image],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        # Reset key response each trial
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trial_loop, data.TrialHandler2) and thisPractice_trial_loop.thisN != practice_trial_loop.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.250-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[go_arrow, stop_arrow, key_resp],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        go_arrow.setImage(go_image)
        stop_arrow.setOpacity(is_stop)
        stop_arrow.setImage(stop_image)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from block_fb_code
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for Stimulus
        Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Stimulus.tStart = globalClock.getTime(format='float')
        Stimulus.status = STARTED
        thisExp.addData('Stimulus.started', Stimulus.tStart)
        Stimulus.maxDuration = None
        # keep track of which components have finished
        StimulusComponents = Stimulus.components
        for thisComponent in Stimulus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Stimulus" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trial_loop, data.TrialHandler2) and thisPractice_trial_loop.thisN != practice_trial_loop.thisTrial.thisN:
            continueRoutine = False
        Stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *go_arrow* updates
            
            # if go_arrow is starting this frame...
            if go_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go_arrow.frameNStart = frameN  # exact frame index
                go_arrow.tStart = t  # local t and not account for scr refresh
                go_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go_arrow.started')
                # update status
                go_arrow.status = STARTED
                go_arrow.setAutoDraw(True)
            
            # if go_arrow is active this frame...
            if go_arrow.status == STARTED:
                # update params
                pass
            
            # if go_arrow is stopping this frame...
            if go_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    go_arrow.tStop = t  # not accounting for scr refresh
                    go_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    go_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_arrow.stopped')
                    # update status
                    go_arrow.status = FINISHED
                    go_arrow.setAutoDraw(False)
            
            # *stop_arrow* updates
            
            # if stop_arrow is starting this frame...
            if stop_arrow.status == NOT_STARTED and tThisFlip >= SSD-frameTolerance:
                # keep track of start time/frame for later
                stop_arrow.frameNStart = frameN  # exact frame index
                stop_arrow.tStart = t  # local t and not account for scr refresh
                stop_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_arrow.started')
                # update status
                stop_arrow.status = STARTED
                stop_arrow.setAutoDraw(True)
            
            # if stop_arrow is active this frame...
            if stop_arrow.status == STARTED:
                # update params
                pass
            
            # if stop_arrow is stopping this frame...
            if stop_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_arrow.tStop = t  # not accounting for scr refresh
                    stop_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    stop_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_arrow.stopped')
                    # update status
                    stop_arrow.status = FINISHED
                    stop_arrow.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corr_resp)) or (key_resp.keys == corr_resp):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Stimulus.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Stimulus" ---
        for thisComponent in Stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Stimulus
        Stimulus.tStop = globalClock.getTime(format='float')
        Stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Stimulus.stopped', Stimulus.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for practice_trial_loop (TrialHandler)
        practice_trial_loop.addData('key_resp.keys',key_resp.keys)
        practice_trial_loop.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            practice_trial_loop.addData('key_resp.rt', key_resp.rt)
            practice_trial_loop.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from SSD_code
        # Only update SSD on STOP trials
        if is_stop == 1:
        
            # Correct stop (no response) → harder
            if key_resp.keys is None:
                SSD += SSD_step
            else:
                # Failed stop → easier
                SSD -= SSD_step
        
            # Keep SSD within bounds
            SSD = max(SSD_min, min(SSD, SSD_max))
        
        
        # Run 'End Routine' code from block_fb_code
        # GO trials
        if is_stop == 0:
            go_trials += 1
            if key_resp.keys is None:
                go_omissions += 1
            else:
                go_rts.append(key_resp.rt)
        
        # STOP trials
        if is_stop == 1:
            stop_trials += 1
            if key_resp.keys is None:
                correct_cancels += 1
        
        # Run 'End Routine' code from feedback_code
        # ALWAYS initialize
        feedback = " "
        
        # STOP trials
        if is_stop == 1 and key_resp.keys is None:
            feedback = "Correct"
        
        elif is_stop == 1 and key_resp.keys is not None:
            feedback = "Remember: try to stop"
        
        # GO trials
        elif is_stop == 0 and key_resp.keys is None:
            feedback = "Too slow"
        
        elif is_stop == 0 and key_resp.keys == corr_resp:
            feedback = "Correct"
        
        elif is_stop == 0 and key_resp.keys != corr_resp:
            feedback = "Incorrect response"
            
        # ----- GO RT collection -----
        if is_stop == 0 and key_resp.keys == corr_resp:
            thisExp.addData('go_rt', key_resp.rt)
        else:
            thisExp.addData('go_rt', None)
        
        # ----- STOP outcome -----
        if is_stop == 1 and key_resp.keys is None:
            thisExp.addData('stop_success', 1)
        elif is_stop == 1:
            thisExp.addData('stop_success', 0)
        else:
            thisExp.addData('stop_success', None)
        
        # ----- SSD logging -----
        thisExp.addData('SSD', SSD)
        
        
        
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Feedback" ---
        # create an object to store info about Routine Feedback
        Feedback = data.Routine(
            name='Feedback',
            components=[fdback_text],
        )
        Feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        fdback_text.setText(feedback)
        # store start times for Feedback
        Feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Feedback.tStart = globalClock.getTime(format='float')
        Feedback.status = STARTED
        thisExp.addData('Feedback.started', Feedback.tStart)
        Feedback.maxDuration = None
        # keep track of which components have finished
        FeedbackComponents = Feedback.components
        for thisComponent in Feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Feedback" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trial_loop, data.TrialHandler2) and thisPractice_trial_loop.thisN != practice_trial_loop.thisTrial.thisN:
            continueRoutine = False
        Feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.75:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fdback_text* updates
            
            # if fdback_text is starting this frame...
            if fdback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fdback_text.frameNStart = frameN  # exact frame index
                fdback_text.tStart = t  # local t and not account for scr refresh
                fdback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fdback_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fdback_text.started')
                # update status
                fdback_text.status = STARTED
                fdback_text.setAutoDraw(True)
            
            # if fdback_text is active this frame...
            if fdback_text.status == STARTED:
                # update params
                pass
            
            # if fdback_text is stopping this frame...
            if fdback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fdback_text.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    fdback_text.tStop = t  # not accounting for scr refresh
                    fdback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    fdback_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fdback_text.stopped')
                    # update status
                    fdback_text.status = FINISHED
                    fdback_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Feedback" ---
        for thisComponent in Feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Feedback
        Feedback.tStop = globalClock.getTime(format='float')
        Feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Feedback.stopped', Feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Feedback.maxDurationReached:
            routineTimer.addTime(-Feedback.maxDuration)
        elif Feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.750000)
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'practice_trial_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "block_feedback" ---
    # create an object to store info about Routine block_feedback
    block_feedback = data.Routine(
        name='block_feedback',
        components=[fb_text, key_resp_4],
    )
    block_feedback.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from fb_code
    # ---- Average GO RT ----
    if len(go_rts) > 0:
        avg_go_rt = sum(go_rts) / len(go_rts)
    else:
        avg_go_rt = 0
    
    avg_go_rt_ms = int(avg_go_rt * 1000)
    
    # ---- Proportion missed GO ----
    if go_trials > 0:
        prop_missed_go = go_omissions / go_trials
    else:
        prop_missed_go = 0
    
    # ---- Stop accuracy ----
    if stop_trials > 0:
        percent_correct_cancels = (correct_cancels / stop_trials) * 100
    else:
        percent_correct_cancels = 0
    
    # ---- FINAL FEEDBACK STRING ----
    feedback_msg = (
        "Average response time (GO trials):\n"
        f"{avg_go_rt_ms} ms\n\n"
        "Proportion missed go:\n"
        f"{prop_missed_go:.2f} (should be 0)\n\n"
        "Percentage of correctly cancelled STOP trials:\n"
        f"{percent_correct_cancels:.1f}% (should be close to 50%)"
        " "
        "Press SPACEBAR to begin the experiment"
    )
    
    # ---- Reset for next block ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    stop_trials = 0
    correct_cancels = 0
    
    fb_text.setText(feedback_msg
    
    )
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for block_feedback
    block_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    block_feedback.tStart = globalClock.getTime(format='float')
    block_feedback.status = STARTED
    thisExp.addData('block_feedback.started', block_feedback.tStart)
    block_feedback.maxDuration = None
    # keep track of which components have finished
    block_feedbackComponents = block_feedback.components
    for thisComponent in block_feedback.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "block_feedback" ---
    block_feedback.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fb_text* updates
        
        # if fb_text is starting this frame...
        if fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fb_text.frameNStart = frameN  # exact frame index
            fb_text.tStart = t  # local t and not account for scr refresh
            fb_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fb_text.started')
            # update status
            fb_text.status = STARTED
            fb_text.setAutoDraw(True)
        
        # if fb_text is active this frame...
        if fb_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            block_feedback.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_feedback.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_feedback" ---
    for thisComponent in block_feedback.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for block_feedback
    block_feedback.tStop = globalClock.getTime(format='float')
    block_feedback.tStopRefresh = tThisFlipGlobal
    thisExp.addData('block_feedback.stopped', block_feedback.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "block_feedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    exp_trial_1 = data.TrialHandler2(
        name='exp_trial_1',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(exp_trial_1)  # add the loop to the experiment
    thisExp_trial_1 = exp_trial_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_1.rgb)
    if thisExp_trial_1 != None:
        for paramName in thisExp_trial_1:
            globals()[paramName] = thisExp_trial_1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisExp_trial_1 in exp_trial_1:
        currentLoop = exp_trial_1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_1.rgb)
        if thisExp_trial_1 != None:
            for paramName in thisExp_trial_1:
                globals()[paramName] = thisExp_trial_1[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[image],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        # Reset key response each trial
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trial_1, data.TrialHandler2) and thisExp_trial_1.thisN != exp_trial_1.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.250-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[go_arrow, stop_arrow, key_resp],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        go_arrow.setImage(go_image)
        stop_arrow.setOpacity(is_stop)
        stop_arrow.setImage(stop_image)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from block_fb_code
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for Stimulus
        Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Stimulus.tStart = globalClock.getTime(format='float')
        Stimulus.status = STARTED
        thisExp.addData('Stimulus.started', Stimulus.tStart)
        Stimulus.maxDuration = None
        # keep track of which components have finished
        StimulusComponents = Stimulus.components
        for thisComponent in Stimulus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Stimulus" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trial_1, data.TrialHandler2) and thisExp_trial_1.thisN != exp_trial_1.thisTrial.thisN:
            continueRoutine = False
        Stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *go_arrow* updates
            
            # if go_arrow is starting this frame...
            if go_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go_arrow.frameNStart = frameN  # exact frame index
                go_arrow.tStart = t  # local t and not account for scr refresh
                go_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go_arrow.started')
                # update status
                go_arrow.status = STARTED
                go_arrow.setAutoDraw(True)
            
            # if go_arrow is active this frame...
            if go_arrow.status == STARTED:
                # update params
                pass
            
            # if go_arrow is stopping this frame...
            if go_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    go_arrow.tStop = t  # not accounting for scr refresh
                    go_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    go_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_arrow.stopped')
                    # update status
                    go_arrow.status = FINISHED
                    go_arrow.setAutoDraw(False)
            
            # *stop_arrow* updates
            
            # if stop_arrow is starting this frame...
            if stop_arrow.status == NOT_STARTED and tThisFlip >= SSD-frameTolerance:
                # keep track of start time/frame for later
                stop_arrow.frameNStart = frameN  # exact frame index
                stop_arrow.tStart = t  # local t and not account for scr refresh
                stop_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_arrow.started')
                # update status
                stop_arrow.status = STARTED
                stop_arrow.setAutoDraw(True)
            
            # if stop_arrow is active this frame...
            if stop_arrow.status == STARTED:
                # update params
                pass
            
            # if stop_arrow is stopping this frame...
            if stop_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_arrow.tStop = t  # not accounting for scr refresh
                    stop_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    stop_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_arrow.stopped')
                    # update status
                    stop_arrow.status = FINISHED
                    stop_arrow.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corr_resp)) or (key_resp.keys == corr_resp):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Stimulus.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Stimulus" ---
        for thisComponent in Stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Stimulus
        Stimulus.tStop = globalClock.getTime(format='float')
        Stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Stimulus.stopped', Stimulus.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for exp_trial_1 (TrialHandler)
        exp_trial_1.addData('key_resp.keys',key_resp.keys)
        exp_trial_1.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            exp_trial_1.addData('key_resp.rt', key_resp.rt)
            exp_trial_1.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from SSD_code
        # Only update SSD on STOP trials
        if is_stop == 1:
        
            # Correct stop (no response) → harder
            if key_resp.keys is None:
                SSD += SSD_step
            else:
                # Failed stop → easier
                SSD -= SSD_step
        
            # Keep SSD within bounds
            SSD = max(SSD_min, min(SSD, SSD_max))
        
        
        # Run 'End Routine' code from block_fb_code
        # GO trials
        if is_stop == 0:
            go_trials += 1
            if key_resp.keys is None:
                go_omissions += 1
            else:
                go_rts.append(key_resp.rt)
        
        # STOP trials
        if is_stop == 1:
            stop_trials += 1
            if key_resp.keys is None:
                correct_cancels += 1
        
        # Run 'End Routine' code from feedback_code
        # ALWAYS initialize
        feedback = " "
        
        # STOP trials
        if is_stop == 1 and key_resp.keys is None:
            feedback = "Correct"
        
        elif is_stop == 1 and key_resp.keys is not None:
            feedback = "Remember: try to stop"
        
        # GO trials
        elif is_stop == 0 and key_resp.keys is None:
            feedback = "Too slow"
        
        elif is_stop == 0 and key_resp.keys == corr_resp:
            feedback = "Correct"
        
        elif is_stop == 0 and key_resp.keys != corr_resp:
            feedback = "Incorrect response"
            
        # ----- GO RT collection -----
        if is_stop == 0 and key_resp.keys == corr_resp:
            thisExp.addData('go_rt', key_resp.rt)
        else:
            thisExp.addData('go_rt', None)
        
        # ----- STOP outcome -----
        if is_stop == 1 and key_resp.keys is None:
            thisExp.addData('stop_success', 1)
        elif is_stop == 1:
            thisExp.addData('stop_success', 0)
        else:
            thisExp.addData('stop_success', None)
        
        # ----- SSD logging -----
        thisExp.addData('SSD', SSD)
        
        
        
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'exp_trial_1'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "block_feedback" ---
    # create an object to store info about Routine block_feedback
    block_feedback = data.Routine(
        name='block_feedback',
        components=[fb_text, key_resp_4],
    )
    block_feedback.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from fb_code
    # ---- Average GO RT ----
    if len(go_rts) > 0:
        avg_go_rt = sum(go_rts) / len(go_rts)
    else:
        avg_go_rt = 0
    
    avg_go_rt_ms = int(avg_go_rt * 1000)
    
    # ---- Proportion missed GO ----
    if go_trials > 0:
        prop_missed_go = go_omissions / go_trials
    else:
        prop_missed_go = 0
    
    # ---- Stop accuracy ----
    if stop_trials > 0:
        percent_correct_cancels = (correct_cancels / stop_trials) * 100
    else:
        percent_correct_cancels = 0
    
    # ---- FINAL FEEDBACK STRING ----
    feedback_msg = (
        "Average response time (GO trials):\n"
        f"{avg_go_rt_ms} ms\n\n"
        "Proportion missed go:\n"
        f"{prop_missed_go:.2f} (should be 0)\n\n"
        "Percentage of correctly cancelled STOP trials:\n"
        f"{percent_correct_cancels:.1f}% (should be close to 50%)"
        " "
        "Press SPACEBAR to begin the experiment"
    )
    
    # ---- Reset for next block ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    stop_trials = 0
    correct_cancels = 0
    
    fb_text.setText(feedback_msg
    
    )
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for block_feedback
    block_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    block_feedback.tStart = globalClock.getTime(format='float')
    block_feedback.status = STARTED
    thisExp.addData('block_feedback.started', block_feedback.tStart)
    block_feedback.maxDuration = None
    # keep track of which components have finished
    block_feedbackComponents = block_feedback.components
    for thisComponent in block_feedback.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "block_feedback" ---
    block_feedback.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fb_text* updates
        
        # if fb_text is starting this frame...
        if fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fb_text.frameNStart = frameN  # exact frame index
            fb_text.tStart = t  # local t and not account for scr refresh
            fb_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fb_text.started')
            # update status
            fb_text.status = STARTED
            fb_text.setAutoDraw(True)
        
        # if fb_text is active this frame...
        if fb_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            block_feedback.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_feedback.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_feedback" ---
    for thisComponent in block_feedback.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for block_feedback
    block_feedback.tStop = globalClock.getTime(format='float')
    block_feedback.tStopRefresh = tThisFlipGlobal
    thisExp.addData('block_feedback.stopped', block_feedback.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "block_feedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    exp_trials_2 = data.TrialHandler2(
        name='exp_trials_2',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(exp_trials_2)  # add the loop to the experiment
    thisExp_trial_2 = exp_trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_2.rgb)
    if thisExp_trial_2 != None:
        for paramName in thisExp_trial_2:
            globals()[paramName] = thisExp_trial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisExp_trial_2 in exp_trials_2:
        currentLoop = exp_trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_2.rgb)
        if thisExp_trial_2 != None:
            for paramName in thisExp_trial_2:
                globals()[paramName] = thisExp_trial_2[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[image],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        # Reset key response each trial
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trials_2, data.TrialHandler2) and thisExp_trial_2.thisN != exp_trials_2.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.250-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[go_arrow, stop_arrow, key_resp],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        go_arrow.setImage(go_image)
        stop_arrow.setOpacity(is_stop)
        stop_arrow.setImage(stop_image)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from block_fb_code
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for Stimulus
        Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Stimulus.tStart = globalClock.getTime(format='float')
        Stimulus.status = STARTED
        thisExp.addData('Stimulus.started', Stimulus.tStart)
        Stimulus.maxDuration = None
        # keep track of which components have finished
        StimulusComponents = Stimulus.components
        for thisComponent in Stimulus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Stimulus" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trials_2, data.TrialHandler2) and thisExp_trial_2.thisN != exp_trials_2.thisTrial.thisN:
            continueRoutine = False
        Stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *go_arrow* updates
            
            # if go_arrow is starting this frame...
            if go_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go_arrow.frameNStart = frameN  # exact frame index
                go_arrow.tStart = t  # local t and not account for scr refresh
                go_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go_arrow.started')
                # update status
                go_arrow.status = STARTED
                go_arrow.setAutoDraw(True)
            
            # if go_arrow is active this frame...
            if go_arrow.status == STARTED:
                # update params
                pass
            
            # if go_arrow is stopping this frame...
            if go_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    go_arrow.tStop = t  # not accounting for scr refresh
                    go_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    go_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_arrow.stopped')
                    # update status
                    go_arrow.status = FINISHED
                    go_arrow.setAutoDraw(False)
            
            # *stop_arrow* updates
            
            # if stop_arrow is starting this frame...
            if stop_arrow.status == NOT_STARTED and tThisFlip >= SSD-frameTolerance:
                # keep track of start time/frame for later
                stop_arrow.frameNStart = frameN  # exact frame index
                stop_arrow.tStart = t  # local t and not account for scr refresh
                stop_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_arrow.started')
                # update status
                stop_arrow.status = STARTED
                stop_arrow.setAutoDraw(True)
            
            # if stop_arrow is active this frame...
            if stop_arrow.status == STARTED:
                # update params
                pass
            
            # if stop_arrow is stopping this frame...
            if stop_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_arrow.tStop = t  # not accounting for scr refresh
                    stop_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    stop_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_arrow.stopped')
                    # update status
                    stop_arrow.status = FINISHED
                    stop_arrow.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corr_resp)) or (key_resp.keys == corr_resp):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Stimulus.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Stimulus" ---
        for thisComponent in Stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Stimulus
        Stimulus.tStop = globalClock.getTime(format='float')
        Stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Stimulus.stopped', Stimulus.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for exp_trials_2 (TrialHandler)
        exp_trials_2.addData('key_resp.keys',key_resp.keys)
        exp_trials_2.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            exp_trials_2.addData('key_resp.rt', key_resp.rt)
            exp_trials_2.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from SSD_code
        # Only update SSD on STOP trials
        if is_stop == 1:
        
            # Correct stop (no response) → harder
            if key_resp.keys is None:
                SSD += SSD_step
            else:
                # Failed stop → easier
                SSD -= SSD_step
        
            # Keep SSD within bounds
            SSD = max(SSD_min, min(SSD, SSD_max))
        
        
        # Run 'End Routine' code from block_fb_code
        # GO trials
        if is_stop == 0:
            go_trials += 1
            if key_resp.keys is None:
                go_omissions += 1
            else:
                go_rts.append(key_resp.rt)
        
        # STOP trials
        if is_stop == 1:
            stop_trials += 1
            if key_resp.keys is None:
                correct_cancels += 1
        
        # Run 'End Routine' code from feedback_code
        # ALWAYS initialize
        feedback = " "
        
        # STOP trials
        if is_stop == 1 and key_resp.keys is None:
            feedback = "Correct"
        
        elif is_stop == 1 and key_resp.keys is not None:
            feedback = "Remember: try to stop"
        
        # GO trials
        elif is_stop == 0 and key_resp.keys is None:
            feedback = "Too slow"
        
        elif is_stop == 0 and key_resp.keys == corr_resp:
            feedback = "Correct"
        
        elif is_stop == 0 and key_resp.keys != corr_resp:
            feedback = "Incorrect response"
            
        # ----- GO RT collection -----
        if is_stop == 0 and key_resp.keys == corr_resp:
            thisExp.addData('go_rt', key_resp.rt)
        else:
            thisExp.addData('go_rt', None)
        
        # ----- STOP outcome -----
        if is_stop == 1 and key_resp.keys is None:
            thisExp.addData('stop_success', 1)
        elif is_stop == 1:
            thisExp.addData('stop_success', 0)
        else:
            thisExp.addData('stop_success', None)
        
        # ----- SSD logging -----
        thisExp.addData('SSD', SSD)
        
        
        
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'exp_trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "block_feedback" ---
    # create an object to store info about Routine block_feedback
    block_feedback = data.Routine(
        name='block_feedback',
        components=[fb_text, key_resp_4],
    )
    block_feedback.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from fb_code
    # ---- Average GO RT ----
    if len(go_rts) > 0:
        avg_go_rt = sum(go_rts) / len(go_rts)
    else:
        avg_go_rt = 0
    
    avg_go_rt_ms = int(avg_go_rt * 1000)
    
    # ---- Proportion missed GO ----
    if go_trials > 0:
        prop_missed_go = go_omissions / go_trials
    else:
        prop_missed_go = 0
    
    # ---- Stop accuracy ----
    if stop_trials > 0:
        percent_correct_cancels = (correct_cancels / stop_trials) * 100
    else:
        percent_correct_cancels = 0
    
    # ---- FINAL FEEDBACK STRING ----
    feedback_msg = (
        "Average response time (GO trials):\n"
        f"{avg_go_rt_ms} ms\n\n"
        "Proportion missed go:\n"
        f"{prop_missed_go:.2f} (should be 0)\n\n"
        "Percentage of correctly cancelled STOP trials:\n"
        f"{percent_correct_cancels:.1f}% (should be close to 50%)"
        " "
        "Press SPACEBAR to begin the experiment"
    )
    
    # ---- Reset for next block ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    stop_trials = 0
    correct_cancels = 0
    
    fb_text.setText(feedback_msg
    
    )
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for block_feedback
    block_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    block_feedback.tStart = globalClock.getTime(format='float')
    block_feedback.status = STARTED
    thisExp.addData('block_feedback.started', block_feedback.tStart)
    block_feedback.maxDuration = None
    # keep track of which components have finished
    block_feedbackComponents = block_feedback.components
    for thisComponent in block_feedback.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "block_feedback" ---
    block_feedback.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fb_text* updates
        
        # if fb_text is starting this frame...
        if fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fb_text.frameNStart = frameN  # exact frame index
            fb_text.tStart = t  # local t and not account for scr refresh
            fb_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fb_text.started')
            # update status
            fb_text.status = STARTED
            fb_text.setAutoDraw(True)
        
        # if fb_text is active this frame...
        if fb_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            block_feedback.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_feedback.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_feedback" ---
    for thisComponent in block_feedback.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for block_feedback
    block_feedback.tStop = globalClock.getTime(format='float')
    block_feedback.tStopRefresh = tThisFlipGlobal
    thisExp.addData('block_feedback.stopped', block_feedback.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "block_feedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    exp_trials_3 = data.TrialHandler2(
        name='exp_trials_3',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(exp_trials_3)  # add the loop to the experiment
    thisExp_trial_3 = exp_trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_3.rgb)
    if thisExp_trial_3 != None:
        for paramName in thisExp_trial_3:
            globals()[paramName] = thisExp_trial_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisExp_trial_3 in exp_trials_3:
        currentLoop = exp_trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_3.rgb)
        if thisExp_trial_3 != None:
            for paramName in thisExp_trial_3:
                globals()[paramName] = thisExp_trial_3[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[image],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        # Reset key response each trial
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trials_3, data.TrialHandler2) and thisExp_trial_3.thisN != exp_trials_3.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.250-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[go_arrow, stop_arrow, key_resp],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        go_arrow.setImage(go_image)
        stop_arrow.setOpacity(is_stop)
        stop_arrow.setImage(stop_image)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from block_fb_code
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for Stimulus
        Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Stimulus.tStart = globalClock.getTime(format='float')
        Stimulus.status = STARTED
        thisExp.addData('Stimulus.started', Stimulus.tStart)
        Stimulus.maxDuration = None
        # keep track of which components have finished
        StimulusComponents = Stimulus.components
        for thisComponent in Stimulus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Stimulus" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trials_3, data.TrialHandler2) and thisExp_trial_3.thisN != exp_trials_3.thisTrial.thisN:
            continueRoutine = False
        Stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *go_arrow* updates
            
            # if go_arrow is starting this frame...
            if go_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go_arrow.frameNStart = frameN  # exact frame index
                go_arrow.tStart = t  # local t and not account for scr refresh
                go_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go_arrow.started')
                # update status
                go_arrow.status = STARTED
                go_arrow.setAutoDraw(True)
            
            # if go_arrow is active this frame...
            if go_arrow.status == STARTED:
                # update params
                pass
            
            # if go_arrow is stopping this frame...
            if go_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    go_arrow.tStop = t  # not accounting for scr refresh
                    go_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    go_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_arrow.stopped')
                    # update status
                    go_arrow.status = FINISHED
                    go_arrow.setAutoDraw(False)
            
            # *stop_arrow* updates
            
            # if stop_arrow is starting this frame...
            if stop_arrow.status == NOT_STARTED and tThisFlip >= SSD-frameTolerance:
                # keep track of start time/frame for later
                stop_arrow.frameNStart = frameN  # exact frame index
                stop_arrow.tStart = t  # local t and not account for scr refresh
                stop_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_arrow.started')
                # update status
                stop_arrow.status = STARTED
                stop_arrow.setAutoDraw(True)
            
            # if stop_arrow is active this frame...
            if stop_arrow.status == STARTED:
                # update params
                pass
            
            # if stop_arrow is stopping this frame...
            if stop_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_arrow.tStop = t  # not accounting for scr refresh
                    stop_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    stop_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_arrow.stopped')
                    # update status
                    stop_arrow.status = FINISHED
                    stop_arrow.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corr_resp)) or (key_resp.keys == corr_resp):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Stimulus.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Stimulus" ---
        for thisComponent in Stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Stimulus
        Stimulus.tStop = globalClock.getTime(format='float')
        Stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Stimulus.stopped', Stimulus.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for exp_trials_3 (TrialHandler)
        exp_trials_3.addData('key_resp.keys',key_resp.keys)
        exp_trials_3.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            exp_trials_3.addData('key_resp.rt', key_resp.rt)
            exp_trials_3.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from SSD_code
        # Only update SSD on STOP trials
        if is_stop == 1:
        
            # Correct stop (no response) → harder
            if key_resp.keys is None:
                SSD += SSD_step
            else:
                # Failed stop → easier
                SSD -= SSD_step
        
            # Keep SSD within bounds
            SSD = max(SSD_min, min(SSD, SSD_max))
        
        
        # Run 'End Routine' code from block_fb_code
        # GO trials
        if is_stop == 0:
            go_trials += 1
            if key_resp.keys is None:
                go_omissions += 1
            else:
                go_rts.append(key_resp.rt)
        
        # STOP trials
        if is_stop == 1:
            stop_trials += 1
            if key_resp.keys is None:
                correct_cancels += 1
        
        # Run 'End Routine' code from feedback_code
        # ALWAYS initialize
        feedback = " "
        
        # STOP trials
        if is_stop == 1 and key_resp.keys is None:
            feedback = "Correct"
        
        elif is_stop == 1 and key_resp.keys is not None:
            feedback = "Remember: try to stop"
        
        # GO trials
        elif is_stop == 0 and key_resp.keys is None:
            feedback = "Too slow"
        
        elif is_stop == 0 and key_resp.keys == corr_resp:
            feedback = "Correct"
        
        elif is_stop == 0 and key_resp.keys != corr_resp:
            feedback = "Incorrect response"
            
        # ----- GO RT collection -----
        if is_stop == 0 and key_resp.keys == corr_resp:
            thisExp.addData('go_rt', key_resp.rt)
        else:
            thisExp.addData('go_rt', None)
        
        # ----- STOP outcome -----
        if is_stop == 1 and key_resp.keys is None:
            thisExp.addData('stop_success', 1)
        elif is_stop == 1:
            thisExp.addData('stop_success', 0)
        else:
            thisExp.addData('stop_success', None)
        
        # ----- SSD logging -----
        thisExp.addData('SSD', SSD)
        
        
        
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'exp_trials_3'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "block_feedback" ---
    # create an object to store info about Routine block_feedback
    block_feedback = data.Routine(
        name='block_feedback',
        components=[fb_text, key_resp_4],
    )
    block_feedback.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from fb_code
    # ---- Average GO RT ----
    if len(go_rts) > 0:
        avg_go_rt = sum(go_rts) / len(go_rts)
    else:
        avg_go_rt = 0
    
    avg_go_rt_ms = int(avg_go_rt * 1000)
    
    # ---- Proportion missed GO ----
    if go_trials > 0:
        prop_missed_go = go_omissions / go_trials
    else:
        prop_missed_go = 0
    
    # ---- Stop accuracy ----
    if stop_trials > 0:
        percent_correct_cancels = (correct_cancels / stop_trials) * 100
    else:
        percent_correct_cancels = 0
    
    # ---- FINAL FEEDBACK STRING ----
    feedback_msg = (
        "Average response time (GO trials):\n"
        f"{avg_go_rt_ms} ms\n\n"
        "Proportion missed go:\n"
        f"{prop_missed_go:.2f} (should be 0)\n\n"
        "Percentage of correctly cancelled STOP trials:\n"
        f"{percent_correct_cancels:.1f}% (should be close to 50%)"
        " "
        "Press SPACEBAR to begin the experiment"
    )
    
    # ---- Reset for next block ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    stop_trials = 0
    correct_cancels = 0
    
    fb_text.setText(feedback_msg
    
    )
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for block_feedback
    block_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    block_feedback.tStart = globalClock.getTime(format='float')
    block_feedback.status = STARTED
    thisExp.addData('block_feedback.started', block_feedback.tStart)
    block_feedback.maxDuration = None
    # keep track of which components have finished
    block_feedbackComponents = block_feedback.components
    for thisComponent in block_feedback.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "block_feedback" ---
    block_feedback.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fb_text* updates
        
        # if fb_text is starting this frame...
        if fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fb_text.frameNStart = frameN  # exact frame index
            fb_text.tStart = t  # local t and not account for scr refresh
            fb_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fb_text.started')
            # update status
            fb_text.status = STARTED
            fb_text.setAutoDraw(True)
        
        # if fb_text is active this frame...
        if fb_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            block_feedback.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_feedback.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_feedback" ---
    for thisComponent in block_feedback.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for block_feedback
    block_feedback.tStop = globalClock.getTime(format='float')
    block_feedback.tStopRefresh = tThisFlipGlobal
    thisExp.addData('block_feedback.stopped', block_feedback.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "block_feedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    exp_trials_4 = data.TrialHandler2(
        name='exp_trials_4',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(exp_trials_4)  # add the loop to the experiment
    thisExp_trial_4 = exp_trials_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_4.rgb)
    if thisExp_trial_4 != None:
        for paramName in thisExp_trial_4:
            globals()[paramName] = thisExp_trial_4[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisExp_trial_4 in exp_trials_4:
        currentLoop = exp_trials_4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisExp_trial_4.rgb)
        if thisExp_trial_4 != None:
            for paramName in thisExp_trial_4:
                globals()[paramName] = thisExp_trial_4[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[image],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        # Reset key response each trial
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trials_4, data.TrialHandler2) and thisExp_trial_4.thisN != exp_trials_4.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.25:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.250-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.250000)
        
        # --- Prepare to start Routine "Stimulus" ---
        # create an object to store info about Routine Stimulus
        Stimulus = data.Routine(
            name='Stimulus',
            components=[go_arrow, stop_arrow, key_resp],
        )
        Stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        go_arrow.setImage(go_image)
        stop_arrow.setOpacity(is_stop)
        stop_arrow.setImage(stop_image)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from block_fb_code
        key_resp.keys = None
        key_resp.rt = None
        
        # store start times for Stimulus
        Stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Stimulus.tStart = globalClock.getTime(format='float')
        Stimulus.status = STARTED
        thisExp.addData('Stimulus.started', Stimulus.tStart)
        Stimulus.maxDuration = None
        # keep track of which components have finished
        StimulusComponents = Stimulus.components
        for thisComponent in Stimulus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Stimulus" ---
        # if trial has changed, end Routine now
        if isinstance(exp_trials_4, data.TrialHandler2) and thisExp_trial_4.thisN != exp_trials_4.thisTrial.thisN:
            continueRoutine = False
        Stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *go_arrow* updates
            
            # if go_arrow is starting this frame...
            if go_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go_arrow.frameNStart = frameN  # exact frame index
                go_arrow.tStart = t  # local t and not account for scr refresh
                go_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go_arrow.started')
                # update status
                go_arrow.status = STARTED
                go_arrow.setAutoDraw(True)
            
            # if go_arrow is active this frame...
            if go_arrow.status == STARTED:
                # update params
                pass
            
            # if go_arrow is stopping this frame...
            if go_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    go_arrow.tStop = t  # not accounting for scr refresh
                    go_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    go_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go_arrow.stopped')
                    # update status
                    go_arrow.status = FINISHED
                    go_arrow.setAutoDraw(False)
            
            # *stop_arrow* updates
            
            # if stop_arrow is starting this frame...
            if stop_arrow.status == NOT_STARTED and tThisFlip >= SSD-frameTolerance:
                # keep track of start time/frame for later
                stop_arrow.frameNStart = frameN  # exact frame index
                stop_arrow.tStart = t  # local t and not account for scr refresh
                stop_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_arrow.started')
                # update status
                stop_arrow.status = STARTED
                stop_arrow.setAutoDraw(True)
            
            # if stop_arrow is active this frame...
            if stop_arrow.status == STARTED:
                # update params
                pass
            
            # if stop_arrow is stopping this frame...
            if stop_arrow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_arrow.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_arrow.tStop = t  # not accounting for scr refresh
                    stop_arrow.tStopRefresh = tThisFlipGlobal  # on global time
                    stop_arrow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_arrow.stopped')
                    # update status
                    stop_arrow.status = FINISHED
                    stop_arrow.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 1.25-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corr_resp)) or (key_resp.keys == corr_resp):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Stimulus.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Stimulus" ---
        for thisComponent in Stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Stimulus
        Stimulus.tStop = globalClock.getTime(format='float')
        Stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Stimulus.stopped', Stimulus.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for exp_trials_4 (TrialHandler)
        exp_trials_4.addData('key_resp.keys',key_resp.keys)
        exp_trials_4.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            exp_trials_4.addData('key_resp.rt', key_resp.rt)
            exp_trials_4.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from SSD_code
        # Only update SSD on STOP trials
        if is_stop == 1:
        
            # Correct stop (no response) → harder
            if key_resp.keys is None:
                SSD += SSD_step
            else:
                # Failed stop → easier
                SSD -= SSD_step
        
            # Keep SSD within bounds
            SSD = max(SSD_min, min(SSD, SSD_max))
        
        
        # Run 'End Routine' code from block_fb_code
        # GO trials
        if is_stop == 0:
            go_trials += 1
            if key_resp.keys is None:
                go_omissions += 1
            else:
                go_rts.append(key_resp.rt)
        
        # STOP trials
        if is_stop == 1:
            stop_trials += 1
            if key_resp.keys is None:
                correct_cancels += 1
        
        # Run 'End Routine' code from feedback_code
        # ALWAYS initialize
        feedback = " "
        
        # STOP trials
        if is_stop == 1 and key_resp.keys is None:
            feedback = "Correct"
        
        elif is_stop == 1 and key_resp.keys is not None:
            feedback = "Remember: try to stop"
        
        # GO trials
        elif is_stop == 0 and key_resp.keys is None:
            feedback = "Too slow"
        
        elif is_stop == 0 and key_resp.keys == corr_resp:
            feedback = "Correct"
        
        elif is_stop == 0 and key_resp.keys != corr_resp:
            feedback = "Incorrect response"
            
        # ----- GO RT collection -----
        if is_stop == 0 and key_resp.keys == corr_resp:
            thisExp.addData('go_rt', key_resp.rt)
        else:
            thisExp.addData('go_rt', None)
        
        # ----- STOP outcome -----
        if is_stop == 1 and key_resp.keys is None:
            thisExp.addData('stop_success', 1)
        elif is_stop == 1:
            thisExp.addData('stop_success', 0)
        else:
            thisExp.addData('stop_success', None)
        
        # ----- SSD logging -----
        thisExp.addData('SSD', SSD)
        
        
        
        # the Routine "Stimulus" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'exp_trials_4'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "block_feedback" ---
    # create an object to store info about Routine block_feedback
    block_feedback = data.Routine(
        name='block_feedback',
        components=[fb_text, key_resp_4],
    )
    block_feedback.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from fb_code
    # ---- Average GO RT ----
    if len(go_rts) > 0:
        avg_go_rt = sum(go_rts) / len(go_rts)
    else:
        avg_go_rt = 0
    
    avg_go_rt_ms = int(avg_go_rt * 1000)
    
    # ---- Proportion missed GO ----
    if go_trials > 0:
        prop_missed_go = go_omissions / go_trials
    else:
        prop_missed_go = 0
    
    # ---- Stop accuracy ----
    if stop_trials > 0:
        percent_correct_cancels = (correct_cancels / stop_trials) * 100
    else:
        percent_correct_cancels = 0
    
    # ---- FINAL FEEDBACK STRING ----
    feedback_msg = (
        "Average response time (GO trials):\n"
        f"{avg_go_rt_ms} ms\n\n"
        "Proportion missed go:\n"
        f"{prop_missed_go:.2f} (should be 0)\n\n"
        "Percentage of correctly cancelled STOP trials:\n"
        f"{percent_correct_cancels:.1f}% (should be close to 50%)"
        " "
        "Press SPACEBAR to begin the experiment"
    )
    
    # ---- Reset for next block ----
    go_rts = []
    go_omissions = 0
    go_trials = 0
    stop_trials = 0
    correct_cancels = 0
    
    fb_text.setText(feedback_msg
    
    )
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for block_feedback
    block_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    block_feedback.tStart = globalClock.getTime(format='float')
    block_feedback.status = STARTED
    thisExp.addData('block_feedback.started', block_feedback.tStart)
    block_feedback.maxDuration = None
    # keep track of which components have finished
    block_feedbackComponents = block_feedback.components
    for thisComponent in block_feedback.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "block_feedback" ---
    block_feedback.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fb_text* updates
        
        # if fb_text is starting this frame...
        if fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fb_text.frameNStart = frameN  # exact frame index
            fb_text.tStart = t  # local t and not account for scr refresh
            fb_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fb_text.started')
            # update status
            fb_text.status = STARTED
            fb_text.setAutoDraw(True)
        
        # if fb_text is active this frame...
        if fb_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            block_feedback.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_feedback.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_feedback" ---
    for thisComponent in block_feedback.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for block_feedback
    block_feedback.tStop = globalClock.getTime(format='float')
    block_feedback.tStopRefresh = tThisFlipGlobal
    thisExp.addData('block_feedback.stopped', block_feedback.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "block_feedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Thank_you" ---
    # create an object to store info about Routine Thank_you
    Thank_you = data.Routine(
        name='Thank_you',
        components=[thankyou_text],
    )
    Thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Thank_you
    Thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Thank_you.tStart = globalClock.getTime(format='float')
    Thank_you.status = STARTED
    thisExp.addData('Thank_you.started', Thank_you.tStart)
    Thank_you.maxDuration = None
    # keep track of which components have finished
    Thank_youComponents = Thank_you.components
    for thisComponent in Thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Thank_you" ---
    Thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 20.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thankyou_text* updates
        
        # if thankyou_text is starting this frame...
        if thankyou_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thankyou_text.frameNStart = frameN  # exact frame index
            thankyou_text.tStart = t  # local t and not account for scr refresh
            thankyou_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thankyou_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thankyou_text.started')
            # update status
            thankyou_text.status = STARTED
            thankyou_text.setAutoDraw(True)
        
        # if thankyou_text is active this frame...
        if thankyou_text.status == STARTED:
            # update params
            pass
        
        # if thankyou_text is stopping this frame...
        if thankyou_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thankyou_text.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                thankyou_text.tStop = t  # not accounting for scr refresh
                thankyou_text.tStopRefresh = tThisFlipGlobal  # on global time
                thankyou_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thankyou_text.stopped')
                # update status
                thankyou_text.status = FINISHED
                thankyou_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Thank_you" ---
    for thisComponent in Thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Thank_you
    Thank_you.tStop = globalClock.getTime(format='float')
    Thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Thank_you.stopped', Thank_you.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Thank_you.maxDurationReached:
        routineTimer.addTime(-Thank_you.maxDuration)
    elif Thank_you.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-20.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
