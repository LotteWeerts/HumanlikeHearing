#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# 
#                           U    U   GGG    SSSS  TTTTT
#                           U    U  G       S       T
#                           U    U  G  GG   SSSS    T
#                           U    U  G   G       S   T
#                            UUU     GG     SSS     T
# 
#                    ========================================
#                     ITU-T - USER'S GROUP ON SOFTWARE TOOLS
#                    ========================================
# 
#        =============================================================
#        COPYRIGHT NOTE: This source code, and all of its derivations,
#        is subject to the "ITU-T General Public License". Please have
#        it  read  in    the  distribution  disk,   or  in  the  ITU-T
#        Recommendation G.191 on "SOFTWARE TOOLS FOR SPEECH AND  AUDIO 
#        CODING STANDARDS".
#        =============================================================
# 
# 
# MODULE:         SV-P56.C, FUNCTIONS RELATED TO ACTIVE LEVEL CALCULATIONS
# 
# ORIGINAL BY:    
#    Simao Ferraz de Campos Neto   CPqD/Telebras Brazil
# 
# DATE:           19/May/2005
# 
# RELEASE:        2.00
# 
# PROTOTYPES:     see sv-p56.h.
# 
# FUNCTIONS:
# 
# init_speech_voltmeter ......... initialization of the speech voltmeter state
#                                 variables in a structure of type SVP56_state.
# 
# speech_voltmeter .............. measurement of the active speech level of
#                                 data in a buffer according to P.56. Other 
# 				relevant statistics are also available.
# 
# HISTORY:
# 
#    07.Oct.91 v1.0 Release of 1st version to UGST.
#    28.Feb.92 v2.0 Correction of bug in speech_voltmeter; inclusion of test
#                   for extremes in bin_interp; use of structure to keep  
#                   state variables.   <simao@cpqd.br>
#    18.May.92 v2.1 Creation of init_speech_voltmeter and consequent changes;
#                   speech_voltmeter changed to operate with float data in the 
#                   normalized range. <simao@cpqd.br>
#    01.Sep.95 v2.2 Added very small constant to avoid problems first detected
#                   in a DEC Alpha VMS workstation with log(0) by
#                   <gerhard.schroeder@fz13.fz.dbp.de>; generalized to all
# 		  platforms <simao@ctd.comsat.com>
#    19.May.05 v2.3 Bug correction in bin_interp() routine, based on changes 
# 				  suggested by Mr Kabal. 
# 				  Upper and lower bounds are updated during the interpolation.
# 						<Cyril Guillaume & Stephane Ragot -- stephane.ragot@francetelecom.com>
# 
# =============================================================================

import math

class SVP56_state(object):

    def __init__(self):
        self.f = 0.0        # (float) sampling frequency, in Hz
        self.a = []         # (unsigned long) activity count
        self.c = []         # (double) threshold level; 15 is the no.of thres.
        self.hang = []      # (unsigned long) hangover count
        self.n = 0          # (unsigned long) number of samples read since last reset
        self.s = 0.0        # (double) sum of all samples since last reset
        self.sq = 0.0       # (double) squared sum of samples since last reset
        self.p = 0.0        # (double) intermediate quantities
        self.q = 0.0        # (double) envelope
        self.max = 0.0      # (double) max absolute value found since last reset
        self.refdB = 0.0    # (double) 0 dB reference point, in [dB]
        self.rmsdB = 0.0    # (double) rms value found since last reset
        self.maxP = 0.0     # (double) maximum positive values since last reset
        self.maxN = 0.0     # (double) maximum negative values since last reset
        self.DClevel = 0.0  # (double) average level since last reset
        self.ActivityFactor = 0.0 # (double) Activity factor since last reset

    def __repr__(self):
        return "<SVP56_state '%s' : '%s'>" % (self.f, self.a)

    def SVP56_get_rms_dB(self):
        return self.rmsdB
        
    def SVP56_get_DC_level(self):
        return self.DClevel
        
    def SVP56_get_activity(self):
        return self.ActivityFactor * 100.0
    
    def SVP56_get_pos_max(self):
        return self.maxP
        
    def SVP56_get_neg_max(self):
        return self.maxN
    
    def SVP56_get_abs_max(self):
        return self.max
        
    def SVP56_get_smpno(self):
        return self.n

# const variables
T = 0.03       # in [s]
H = 0.20       # in [s]
M = 15.9       # in [dB]
THRES_NO = 15  # number of thresholds in the speech voltmeter
MIN_LOG_OFFSET=1e-20 # Hooked to eliminate sigularity with log(0.0) (happens w/all-0 data blocks
    
def bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):

    # Consistency check
    if tol < 0.:
        tol = -1.0 * tol
    
    # Check if extreme counts are not already the true active value
    iterno = 1
    diff = math.fabs((upcount - upthr) - Margin)
    if diff < tol:
        return upcount
    diff = math.fabs((lwcount - lwthr) - Margin)
    if diff < tol:
        return lwcount
    
    # Initialize first middle for given (initial) bounds
    midcount = (upcount + lwcount) / 2.0
    midthr = (upthr + lwthr) / 2.0
    
    # Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol)
    diff = (midcount - midthr) - Margin
    while math.fabs(diff) > tol:
        # if tolerance is not met up to 20 iteractions, then relax the 
        # tolerance by 10
        iterno += 1
        if iterno > 20:
            tol *= 1.1
        
        if diff > tol:
            # then new bounds are ... 
            midcount = (upcount + midcount) / 2.0   # upper and middle activities
            midthr = (upthr + midthr) / 2.0	        # ... and thresholds
            lwcount = midcount
            lwthr = midthr
        elif diff < -1.0 * tol:
            # then new bounds are ... 
            midcount = (midcount + lwcount) / 2.0   # middle and lower activities
            midthr = (midthr + lwthr) / 2.0         # ... and thresholds
            upcount = midcount
            upthr = midthr
        
        diff = (midcount - midthr) - Margin

    # Since the tolerance has been satisfied, midcount is selected 
    # as the interpolated value with a tol [dB] tolerance. */
    return midcount
        
def init_speech_voltmeter(state, sampl_freq):
    
    # First initializations
    state.f = sampl_freq
    I = math.floor(H * state.f + 0.5)
    
    # Inicialization of threshold vector
    x = 0.5
    for j in range(1, THRES_NO + 1):
        state.c.insert(0, x)
        x /= 2.0
        
    # Inicialization of activity and hangover count vectors
    for j in range(0, THRES_NO):
        state.a.append(0)
        state.hang.append(I)
        
    # Inicialization for the quantities used in the two P.56's processes
    state.s = state.sq = state.n = state.p = state.q = 0
    
    # Inicialization of other quantities referring to state variables
    state.max = 0
    state.maxP = -32768.0
    state.maxN = 32767.0
    
    # Defining the 0 dB reference level in terms of normalized values
    state.refdB = 0 # dBov
        
def speech_voltmeter(buffer, state):
    
    # Some initializations
    I = math.floor(H * state.f + 0.5)
    g = math.exp(-1.0 / (state.f * T))
    smpno = len(buffer)

    # Calculates statistics for all given data points
    for k in range(smpno):
        x = buffer[k]

        # Compares the sample with the max. already found for the file
        if math.fabs(x) > state.max:
            state.max = math.fabs(x)
        # Check for the max. pos. value
        if x > state.maxP:
            state.maxP = x
        # Check for the max. neg. value
        if x < state.maxN:
            state.maxN = x
            
        # Implements Process 1 of P.56
        state.sq += x * x
        state.s  += x
        state.n  += 1
        
        # Implements Process 2 of P.56
        state.p = g * state.p + (1 - g) * math.fabs(x)
        state.q = g * state.q + (1 - g) * state.p      

        # Applies threshold to the envelope q
        for j in range(THRES_NO):
            if state.q >= state.c[j]:
                state.a[j] += 1
                state.hang[j] = 0
            if (state.q < state.c[j]) and (state.hang[j] < I):
                state.a[j] += 1
                state.hang[j] += 1
   
    # Computes the statistics
    state.DCleven = state.s / state.n
    LongTermLevel = 10 * math.log10(state.sq / state.n + MIN_LOG_OFFSET)
    state.rmsdB = LongTermLevel - state.refdB
    state.ActivityFactor = 0
    ActiveSpeechLevel = -100.0
    
    # Test the lower active counter; if 0, is silence
    if state.a[0] == 0:
        return ActiveSpeechLevel
    else:
       AdB = 10 * math.log10(((state.sq) / state.a[0]) + MIN_LOG_OFFSET)
    
    # Test if the lower act.counter is below the margin: if yes, is silence
    CdB = 20 * math.log10(float(state.c[0]))
    if AdB - CdB < M:
        ActiveSpeechLevel
        
    # Proceed serially for steps 2 and up -- this is the most common case
    Delta = [0.0 for i in range(THRES_NO)]
    for j in range(1, THRES_NO):
        if state.a[j] != 0:
            AdB = 10 * math.log10(((state.sq) / state.a[j]) + MIN_LOG_OFFSET)
            CdB = 20 * math.log10((float(state.c[j])) + MIN_LOG_OFFSET)
            Delta[j] = AdB - CdB
            # then interpolates to find the active
            # level and the activity factor and exits
            if Delta[j] <= M:
                # AmdB is AdB for j-1, CmdB is CdB for j-1
                AmdB = 10 * math.log10(((state.sq) / state.a[j - 1]) + MIN_LOG_OFFSET)
                CmdB = 20 * math.log10(float(state.c[j - 1]) + MIN_LOG_OFFSET)
                ActiveSpeechLevel = bin_interp(AdB, AmdB, CdB, CmdB, M, 0.5 )
                
                state.ActivityFactor = math.pow(10.0, ((LongTermLevel - ActiveSpeechLevel) / 10))
                break
                
    return ActiveSpeechLevel