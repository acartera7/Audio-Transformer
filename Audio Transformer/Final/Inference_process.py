import math
import heapq
import numpy as np

CYCLE_EPSILON = 30 #TODO make it percentage wise 25%
FFT_SIZE = 1024
FRONT_VEC_SIZE = 40

AUDIO_SIZE = 16000
SEGMENT_LENGTH = 500
FREQ_WEIGHT = 1
MID_WEIGHT = 2
F0_FREQ_RANGE = (50,200)
NOISE_FACTOR = 0.65

N_SEGMENTS = AUDIO_SIZE//SEGMENT_LENGTH

LOW_F0_INDEX = math.floor(FFT_SIZE/AUDIO_SIZE * F0_FREQ_RANGE[0])
HIGH_F0_INDEX = math.floor(FFT_SIZE/AUDIO_SIZE * F0_FREQ_RANGE[1])


def wav_value(waveform_arr, location:float):
    # Ensure the location is within the bounds of the array
  if location < 0 or location > len(waveform_arr) - 1:
    raise ValueError("Location is out of bounds.")

  # Define the indices for interpolation
  indices = np.arange(len(waveform_arr))

  # Use numpy's interp function to get the interpolated value
  return np.interp(location, indices, waveform_arr)

def vectorize(waveform_arr, n:int):
  indices = np.linspace(0, len(waveform_arr) - 1, n, dtype=int)
  return waveform_arr[indices]

def vectorize_f(waveform_arr, n:int):
  # Calculate the indices of the evenly spaced samples
  sample_points = np.linspace(0, len(waveform_arr) - 1, n)
  
  # Use the wav_value function to get the interpolated values
  return np.array([wav_value(waveform_arr, point) for point in sample_points])
  
def bias_front(waveform_arr):
  front = (len(waveform_arr) - 1) // 3
  #front = int((len(waveform_arr) - 1) // (f0//(F0_FREQ_RANGE[0]//2)))
  #adds up 1st 3rd of the
  return np.sum(waveform_arr[:front])

def bias_front_abs(waveform_arr):
  front = (len(waveform_arr) - 1) // 3
  #front = int((len(waveform_arr) - 1) // (f0//(F0_FREQ_RANGE[0]//2)))
  #adds up 1st 3rd of the
  return np.sum(abs(waveform_arr[:front]))

# gets k largest elements from a list along with its indices 
def klargest_with_indices(arr, k, range=None): 
  heap = []
  for i, val in enumerate(arr):
      if range and not (range[0] <= i <= range[1]):
        continue
      heap.append((i,val))
  return heapq.nlargest(k, heap, key=lambda x: x[1])

def calculate_rms(audio):
  return np.sqrt(np.mean(audio**2))

# returns an array of RMS values from a signal over time specified by the window size 
def rms_over_windows(waveform_arr, window_size=100, silence_threshold = None):
  rms_values = []
  excluded_ranges = []
  
  start = 0
  while start < len(waveform_arr):
    end = start + window_size
    if end > len(waveform_arr):
      end = len(waveform_arr)
    window = waveform_arr[start:end]
    rms_value = calculate_rms(window)
    rms_values.append(rms_value)
    
    if silence_threshold:
      if rms_value < silence_threshold:
        excluded_ranges.append((start, end))

    start = end
  
  return rms_values, excluded_ranges

def fft_max(waveform_arr : np.array):

  # Pad with zeros if necessary 
  if(len(waveform_arr) != FFT_SIZE):
    size_diff = abs(len(waveform_arr)-FFT_SIZE)
    list(waveform_arr).extend([0] * size_diff)

    # perform fft and get magnitude
  fft = np.fft.fft(waveform_arr, FFT_SIZE)
  fft_mag = np.abs(fft)
  
    #turn back to np.array
  fft_mag = fft_mag[:FFT_SIZE // 2]
  max_index = np.argmax(fft_mag)
  return fft_mag, max_index

def find_zerocrossings(waveform_arr:np.array, start_sample, dynamic_threshold=0.01):
  last=None 
  zerocrossings = np.array([], dtype='f')
  
  #Filter out quiet parts of the signal using a RMS with windowing
  _, excluded_ranges = rms_over_windows(waveform_arr, SEGMENT_LENGTH//5, dynamic_threshold)

  # turn the excluded ranges into an array of samples that shouldn't be processed
  excluded_samples = [] 
  for start, end in excluded_ranges:
    excluded_samples.extend(range(start_sample + start, start_sample + end))

  #sweep the whole segment
  for index, value in enumerate(waveform_arr):
    absolute_index = start_sample + index
    if absolute_index in excluded_samples:
      continue 
    if last == None:  #record last sample
      last = value
    #elif value == 0:
    #  if np.sign(last) == -1.0 and np.sign(waveform_arr[index+1]) == 1.0:
    #    np.append(zerocrossings, start_sample+index)
    
    # if last sample is negative and current one is positive a zero crossing occured
    if np.sign(last) == -1.0 and np.sign(value) == 1.0: 
      # x = x1 + (x2-x1)/(y2-y1)*(y-y1)
      inter_x = (index-1) + (-last/(value-last))
      zerocrossings = np.append(zerocrossings,start_sample+inter_x)
    last = value
  return zerocrossings

def find_cycles_f0(f0, start_sample, zero_crossings:list):
  
  cycles = []
  f0_length = 1.0/f0 * AUDIO_SIZE #length of samples of fundamental frequency
  last_sample = start_sample*SEGMENT_LENGTH + SEGMENT_LENGTH 
  for start in zero_crossings:
    cyclef0_end = start+f0_length # where we are expecting the cycle to end 
    
    if cyclef0_end > last_sample+CYCLE_EPSILON: #don't look past end of segment, unless CYCLE_EPSILON allows you to see the end 
      continue

    high = cyclef0_end + CYCLE_EPSILON #acceptabale ranges
    low = cyclef0_end - CYCLE_EPSILON
    
    #get values that fall withing the f0+epsilon range
    try:
      possible_cycles = list(filter(lambda x, high=high, low=low: low <= x <= high ,zero_crossings))
      if not possible_cycles:
        #print(f"Failed to find a cycle for sample {start} with an epsilon of {CYCLE_EPSILON}")
        continue
      if len(possible_cycles) > 1:
        # tie-breaker get the end sample that is closest to the expected
        differences = list(map(lambda x, y=cyclef0_end: abs(y-x), possible_cycles))
        end = possible_cycles[np.argmin(differences)]
      else:
        end = possible_cycles[0]
    except Exception as e:
      print(f"Error: Failed to find a cycle for sample {start} with an epsilon of {CYCLE_EPSILON}\n",e)
    cycles.append((start,end))
  return cycles

def find_repcycle3(segment_wav, start_sample, f0, cycles):
  #score the cycles based an obvious shape at the beginning of the cycle
  
  if len(cycles) < 2:
    return cycles[0]  #if there is only one cycle return it
  
  mid_segment = start_sample+ (SEGMENT_LENGTH/2)
  f0_length = 1.0/f0 * AUDIO_SIZE #length of samples of fundamental frequency

  cycles_data = [] # (index, front, front_abs, f_score, m_score)
  for index, cycle in enumerate(cycles):
    
    i_beginning = math.floor(cycle[0]-start_sample)
    i_end = math.floor(cycle[1]-start_sample)
    cyc_wav = segment_wav[i_beginning:i_end]

    #identify cycles that have an obvious shape at the beginning of the cycle 
    #cyc_vec = vectorize_f(cyc_wav, FRONT_VEC_SIZE)
    front = bias_front_abs(cyc_wav)
 
    #get length of the cycle and comapre with fundamental frequency reciprocal 
    freq = cycle[1] - cycle[0]
    f_score = abs(freq - f0_length)

    #get middle and comare with middle of the whole segment  
    middle = (cycle[0] + cycle[1]) / 2
    m_score = abs(middle - mid_segment)

    cycles_data.append((index, front, f_score, m_score))

  if(len(cycles_data) <= 4): #No need to distinguish outliers if there are only 4 or less cycles
    candidates_list = cycles_data
  else:
    front_mean =  np.mean([tup[1] for tup in cycles_data])
    filtered_list = list(filter(lambda x, front_mean_abs=front_mean: x[1] > front_mean_abs, cycles_data))

    if len(filtered_list) > 1:
      candidates_list = filtered_list
    else:
      candidates_list = cycles_data

  assert candidates_list != [], f"No repcycles found at {start_sample} to {start_sample+SEGMENT_LENGTH}"

    # Normalize scores
  max_f_score = max(candidates_list, key=lambda x: x[2])[2]
  max_m_score = max(candidates_list, key=lambda x: x[3])[3]

  normalized_cycles_data = [
    (index, front, f_score / max_f_score, m_score / max_m_score)
    for index, front,  f_score, m_score in candidates_list
  ]

  # Apply weights and sort
  weighted_cycles_data = [
    (index, front,  f_score / FREQ_WEIGHT, m_score / MID_WEIGHT)
    for index, front,  f_score, m_score in normalized_cycles_data
  ]
  
  sorted_list = sorted(weighted_cycles_data, key=lambda x: x[2]+x[3])
  assert sorted_list != [], f"No repcycles found at {start_sample} to {start_sample+SEGMENT_LENGTH}"
  return cycles[sorted_list[0][0]]

def find_repcycle_quick(start_sample, cycles):
  #look for a cycle that is closest to the middle of the segment
  
  if len(cycles) < 2:
    return cycles[0]  #if there is only one cycle return it
  
  cycle_data = []
  mid_segment = start_sample+(SEGMENT_LENGTH/2)
  
  for cycle in cycles:
    middle = (cycle[0] + cycle[1]) / 2
    cycle_data.append(abs(middle - mid_segment))
  
  return cycles[np.argmin(cycle_data)] #return the cycle closest to the middle of the segment

def process_repcycles(waveform_np, quick:bool=False):

  #identify SNR reciprocal to get a dynamic noise threshold for filtering 
  rms_values, _ = rms_over_windows(waveform_np, SEGMENT_LENGTH) #calculate RMS over time
  
  # Get Noise-to-Signal Ratio
  signal_rms = np.max(rms_values)
  noise_rms = np.mean(rms_values) * NOISE_FACTOR

  # Calculate dynamic silence threshold based on NSR
  silence_threshold = min(noise_rms, signal_rms*.50)

  # Split the waveform into segments
  split_waveform = np.array_split(waveform_np, np.arange(SEGMENT_LENGTH, AUDIO_SIZE, SEGMENT_LENGTH))
  #split_waveform = [waveform_np[i:i+segment_length] for i in range(0, 16000, segment_length)]

  repcycles = []

  for segment_num, segment_wav in enumerate(split_waveform):
    #segment_wav = split_waveform[SEGMENT_NUM]s
    start_sample = segment_num*SEGMENT_LENGTH
    # Get the Zero-Crossings, ignoring noise
    zero_crossings = find_zerocrossings(segment_wav, start_sample, silence_threshold)
    if not len(zero_crossings) > 1:
      repcycles.append([])
    # Get the fundamental frequency using the FFT
      continue
    fft_mag, max_index = fft_max(segment_wav)

    # exclude bins outside the F0_FREQ_RANGE
    # get frequency to bin index

    top_bins = klargest_with_indices(fft_mag, 4, (LOW_F0_INDEX,HIGH_F0_INDEX))
    assert len(top_bins) > 0, f"Failed to find a fundamental frequency in range ({F0_FREQ_RANGE[0]}Hz ,{F0_FREQ_RANGE[1]}Hz]"# for {os.path.join(path_root.split('\\')[-1], filename)})"
    #top_in_range = list(filter(lambda x, h=LOW_F0, l = HIGH_F0: l <= x[0] <= h, top_bins))[0]

    # Fundamental Frequency: Sample Rate / fftsize * fft_bin_index
    f0 = AUDIO_SIZE/FFT_SIZE * top_bins[0][0]

    #print(f"silence_threshold: {silence_threshold}")
    #print(f"f0: {f0}")
    #print(f"F0 magnitude{fft_mag[max_index]}")

    #find cycles within the signal and find a representative one for the segment
    cycles = find_cycles_f0(f0, start_sample, zero_crossings)
    if not len(cycles) > 0:
      repcycles.append([])
      continue
    
    if quick:
      # find a representative cycle by looking for the cycle closest to the middle of the segment
      repcycle = find_repcycle_quick(start_sample, cycles)
    else:
      repcycle = find_repcycle3(segment_wav, start_sample, f0, cycles)
    assert repcycle != None, f"Error: failed to find a representative cycle for segment [{start_sample}, {start_sample+SEGMENT_LENGTH}]" #for {os.path.join(path_root.split('\\')[-1], filename)}"

    repcycles.append(repcycle)

  return repcycles



