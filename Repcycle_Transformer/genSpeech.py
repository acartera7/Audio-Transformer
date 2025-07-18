
# ----- Brief Description -----
# 
# Generate speech waveform given inputs:  
# time = waveform duration in seconds,
# sample_rate = sample rate
# key_cycles = list of m intervals [a_i,b_i] i = 0,...,m-1
# key_bcoeffs = m rows of n B-spline coefficients for each key cycle,
# interp_method = 1 for linear
#
# def gliss2Cycles(time, sample_rate, key_cycles)
# returns [all_cycles, keys] 
# def genSpeech1(time, sample_rate, key_cycles, key_bcoeffs, interp_method)
# returns waveform
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# all_cycles = vector of all interval endpoints: x_0,...,x_M
# (so that all the cycles, key and intermediate, are [x_0,x_1], [x_1,x_2], ... [x_{M-1},x_M] )
# and [x_0,x_1] = [a_0,b_0], and [x_{M-1},x_M] = [a_{m-1},b_{m-1}].
# The number intermediate cycles is determined by the quadratic gliss2 method.
# keys = indices of key cycles in list of all cycles, keys[i] = j when a_i = x_j in all_cycles.
#
# ----- ----- ----- ----- -----

import sys
import math
import torch
import torchaudio
import numpy as np
import math
import pathlib

from genCycle import insertCycle
from getBcoeffs import getBcoeffs
from genWavTone import reset

print("Argument List:", str(sys.argv))
args = len(sys.argv)

audio_file = sys.argv[1]
path = "../audio/" + audio_file
print("path: ", path)
index = audio_file.find(".")
param = audio_file[:index]
audio_prefix = audio_file[:index]
print("audio_prefix:", audio_prefix)

waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
time = float(num_frames / sample_rate)
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)
print("with length ", time, " seconds")

# args are: [0] genSpeech.py [1] audio_file [2] rep_cycles_file [3] n 

def import_cycles(file) :
    cycles = []
    cycles_str = []
    with open(file, 'r') as f:
        cycles_str = f.readlines()
        f.close()
    n = len(cycles_str)  # number of lines in file
    for i in range(n) :
        cycle = []
        cycle_pair = cycles_str[i].split(',')
        for val_str in cycle_pair :
            cycle.append(float(val_str))
        cycles.append(cycle)
    return cycles

rep_cycles_file = sys.argv[2]
print("rep_cycles_file:  ", rep_cycles_file)
rep_cycles = import_cycles(rep_cycles_file)
print("rep_cycles:")
print(rep_cycles)

n = int(sys.argv[3])
print("spline dimension: ", n)


# not using this function yet:
def getRepCycles(waveform, sample_rate, segment_size) :

    # do FFT on each segment to guess f_0 using arg_max on frequency bins, then choose a cycle of
    # approximate length 1/f_0 * sample_rate samples, out of a collection of candidate cycles,
    # use penalty function with five possible terms: 
    # 1) closeness to f_0 (use average of cycles to get f_0)
    # 2) power or magnitude (use sum of squares, then 1/x)
    # 3) closeness to middle of segment (use center of cycle)
    # 4) similar shape to previous rep cycle (use dot product of bcoeffs)
    # 5) bias toward larger average (sum of samples or bcoeffs) in first third of cycle.
    # In cases where the best cycle is toward the end of segment, we could allow for this
    # selection but then make a requirement on the distance to the next rep cycle in the next segment,
    # for example so that there are at least two cycle lengths (based on previous cycle) in between. 
    
    rep_cycles = []
    return rep_cycles

# can use getBcoeffs to produce the bcoeffs for the key cycles returned by getRepCycles
# def getBcoeffs(waveform, sample_rate, cycle, n) :
# where waveform can be a segment as in speechCycles


def getRepBcoeffs(waveform, sample_rate, rep_cycles, n) :

    m = len(rep_cycles)
    rep_bcoeffs = torch.zeros(m, n)
    for i in range(m) :
        bcoeffs = getBcoeffs(waveform, sample_rate, rep_cycles[i], n)
        rep_bcoeffs[i] = bcoeffs
    return rep_bcoeffs

key_bcoeffs = getRepBcoeffs(waveform, sample_rate, rep_cycles, n)

def gliss2Cycles(time, sample_rate, key_cycles) :

    # convert list of key cycles [a_i,b_i] inside interval [0,time * sample_rate]
    # into list of all_cycles (described below) with cycle lengths interpolated with gliss2 quadratic method
    # between key cycles but not before first or after last key cycle.  In addition to insertion of 
    # intermediate cycles, those are counted so that we have a list of indices called keys which are the 
    # indices of the key cycles in the list of all cycles.
    # Note: all_cycles is described below and is not of same type as key_cycles, it is just a list of endpoints,
    # which is convenient since the cycles are all consecutive (unlike the key cycles). 

    # For each key_cycle [a_i,b_i] generate sequence of cycles which give interval endpoints up to, but not
    # including the next key cycle [a_{i+1},b_{i+1}].  Enter this subsequence of values into the list all_cycles
    # which looks like {a_i, b_i, u_1, u_2, .... u_h}, where backward differences in this list give
    # the monotonic sequence of interval lengths, and this property extends through the next key cycle:
    # (for example if it is increasing: b_i-a_i < u_1-b_1 < ... < a_{i+1}-u_h < b_{i+1}-a_{i+1}.)
    # Repeat this for each key cycle, but for the last key cycle do not do the interpolation part, 
    # only write endpoints of that key cycle as a_{m-1}, b{m-1}, which ends the list all_cycles. 

    m = len(key_cycles)

    j = 0  # index for all_cycles
    all_cycles = []
    keys = []
    a = 0.0
    b = 0.0

    cycle0 = key_cycles[0]
    a0 = cycle0[0]
    b0 = cycle0[1]
    all_cycles.append(a0)

    # loop on key cycles:
    for i in range(m) :
        
        cycle0 = key_cycles[i]
        a0 = cycle0[0]
        b0 = cycle0[1]
        print("key cycle ", i, ": [a,b] = [", a0, ",", b0, "]") 
        print("j value: ", j)
        # check for consistency:  
        # key_cycles [a_i,b_i] should satisfy 0 < a_0 < b_0 < a_1 ... < b_{m-1} < time * sample_rate 
        if a0 < b :
            print("inconsistent interval data for key cycle: ", i)
            print("a0 = left endpoint of key cycle :  ", a0)
            print("b = right endpoint of previous intermediate cycle:  ", b)
        if i == m-1 :  # last cycle is a key cycle, so append b and continue
            keys.append(j)
            all_cycles.append(b0)
            continue
            # don't advance j since this is last value to write 

        # push index j of this key cycle [a_i,b_i] in all_cycles so that a_i = x_j
        keys.append(j)
        j += 1
        # push right endpoint of this key cycle into all_cycles 
        print("appending b value of key cycle ", i, ": ", b0)
        all_cycles.append(b0)

        # now do intermediate cycles between key cycles i and i+1
        cycle1 = key_cycles[i+1]
        a1 = cycle1[0]
        b1 = cycle1[1]
        L = b1 - a0  # time interval in samples from a_i to b_{i+1}
        c0 = b0 - a0  # starting cycle length in samples
        c1 = b1 - a1  # ending cycle length in samples
        Lp = L - c0 - c1      # = L' = intermediate interval length in samples
        avg = (c0 + c1) / 2   # = average of first and last cycles in samples
        kp = (Lp / avg)       # kp = number of intermediate cycles (float) = k'
        k = int(kp)         # k = number of intermediate cycles
        x = kp - k          # fractional part of kp
        print("k = ", k)
        print("kp = ", kp)
        print("x = ", x)
        print("c0 = ", c0)
        print("c1 = ", c1)
        print("avg = ", avg)
        print("Lp = ", Lp)

        num_cycles = k + 2  # 2 for the ends, c0 and c1, and k in the middle
        numer = L / float(k+2) - avg
        denom = 1 - float(2*k+3) / float(3*k+3)  # this is approx 1/3 for large k
        delta = numer / denom
        print("delta: ", delta) 
    
        print("\n")
   
        # now compute intermediate cycles between key cycles i and i+1
        # and for each new cycle [a,b] append b to all_cycles 
        a = a0
        b = b0
        interm = 0  # sum of intermediate cycle lengths
        for i in range(k) :  # numcycles = k+2
            a = b
            # quadratic version with delta:
            t = i / float(k+1)
            q1 = c0 * (1-t) * (1-t)
            q2 = 2 * (avg + delta) * (1-t) * t
            q3 = c1 * t * t
            cycle_length = q1 + q2 + q3  # cycle length in samples
            print("cycle length ", j, ":  ", cycle_length)
            interm += cycle_length
            b = a + cycle_length
            b = reset(b)
            all_cycles.append(b)  # only push endpoint b, since a is already there
            j += 1
        print("intermediate cycle sum: ", interm)

    return [all_cycles, keys]



def genSpeech1(time, sample_rate, key_cycles, key_bcoeffs, interp_method) :

    # first convert key_cycles to all_cycles with gliss2 quadratic method

    all_cycles, keys = gliss2Cycles(time, sample_rate, key_cycles)
    num_cycles = len(all_cycles) - 1  # count cycles as right end points in this list
    num_keys = len(key_cycles)
    print("all cycles: ")
    print(all_cycles)

    # now all_cycles has list of cycle endpoints x_j so that cycle i is given by 
    # [a_i,b_i] = [x_i,x_{i+1}] i=0,...,M-1, where M = num_cycles
    # and this list starts with first key_cycle and ends with last key cycle.
    # The space before first and after last key cycles will be handled separately. 
    # So the number of cycles is num_cycles and this includes the key cycles. 

    n = key_bcoeffs.size(dim=1) # dimension of splines for each cycle
    
    # Need to interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    all_bcoeffs = torch.zeros(num_cycles,n)  # rows are bcoeffs of each cycle
    all_bcoeffs[0] = key_bcoeffs[0]  # first row
    for i in range(num_keys) :
        key = keys[i]
        all_bcoeffs[key] = key_bcoeffs[i]  # assign entire row 
    # print(all_bcoeffs)
    # now the key bcoeffs are filled into the array all_bcoeffs 

    # keys are an increasing sequence of indices of key cycles which satisfy: 
    # keys[0]=0, keys[num_keys-1]=last_key, and 0 < keys[i] < last_key  
    
    print("num_cycles: ", num_cycles)
    waveform_length = int(sample_rate * time)
    print("waveform length: ", waveform_length)
    waveform = torch.zeros(waveform_length)  # final output samples as 1-dim tensor array

    # Now fill in all_bcoeffs using interpolation between key cycles using tensor operations 
    # on rows of all_bcoeffs, where each row is the set of n bcoeffs for one cycle, 
    # so we are basically doing row ops on a matrix
    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles (rows of bcoeffs) is keys[i+1]-keys[i]-1
        num_interm = keys[i+1]-keys[i]-1
        for j in range(num_interm) :
            start_interm = keys[i]+1  # each intermediate index is now start_interm + j
            # p goes from 1/num_interm to 1=num_interm/num_interm, so p = (j+1)/num_interm 
            p = float(j+1) / num_interm
            all_bcoeffs[start_interm + j] = (1-p) * key_bcoeffs[i] + p * key_bcoeffs[i+1]

    # Now all_bcoeffs should have all intermediate cycles filled 
    print("all_bcoeffs: ", all_bcoeffs)

    # Next, loop through all cycles and write output samples for each cycle
    # into waveform data array. 

    # number of trailing cycles (after last key cycle) is num_cycles - last_key
    # num_trailing_cycles = num_cycles - last_key

    # Need to write one cycle at a time, using endpoints a and b (between samples) 

    a = 0.0
    b = 0.0
    count = 0 # index for output sample data
    # tail = 1
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = all_cycles[i]
        b = all_cycles[i+1]
        cycle = [a, b]
        bcoeffs = all_bcoeffs[i]

        insertCycle(waveform, cycle, bcoeffs)

    return waveform

interp_method = 1

output_waveform = genSpeech1(time, sample_rate, rep_cycles, key_bcoeffs, interp_method)
print("we have wav data")

# write wav_data to file:
size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = output_waveform[i]

path1 = "../audio"
path = path1 + "/genSpeech2" 
path += ".wav"
print("now writing wav file:")
print(path)

torchaudio.save(
    path, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)


