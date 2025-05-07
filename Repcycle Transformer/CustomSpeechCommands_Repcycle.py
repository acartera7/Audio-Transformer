# Python Object that loads a dataset of wav files and their representative cycles for use in training a Neural Network 
# Andrei Cartera 
import random
import math
import os, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchaudio 

from torch.utils.data import Dataset
import torch.nn.functional as nnF

from collections import defaultdict

from repcycle_process import process_repcycles


# Install soundfile and ffmpeg-python if not already installed
# pip install soundfile
# pip install ffmpeg-python

#print(torch.__version__)
#print(str(torchaudio.list_audio_backends()))
#
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

# Define the custom order
classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

class CustomSpeechCommandsDataset_Repcycle(Dataset):
  def __init__(self, base_dir: str, subset: str = None, n_segments=32, shuffle: bool = False, vec_size=40, divisor: int = 1, control: bool = False):
    
    self.vec_size=vec_size 
    self.n_segments = n_segments
    self.base_dir = Path(base_dir)
    self.subset = subset
    self.all_audio_paths = list(self.base_dir.glob("*/*.wav"))
    self.validation_files = self._load_list("validation_list.txt")
    self.testing_files = self._load_list("testing_list.txt")
    #if subset == "validation":
    #  self.audio_paths = [p for p in self.all_audio_paths if p in self.validation_files and p not in self.testing_files]
    #elif subset == "testing":
    #  self.audio_paths = [p for p in self.all_audio_paths if p not in self.validation_files and p in self.testing_files]
    #else:
    #  self.audio_paths = [p for p in self.all_audio_paths if p not in self.validation_files and p not in self.testing_files]
    if self.validation_files and self.testing_files:
      if subset == "testing":
        self.audio_paths = [p for p in self.all_audio_paths if p not in self.validation_files and p in self.testing_files]
      else:
        self.audio_paths = [p for p in self.all_audio_paths if p not in self.testing_files]
    else:
      self.audio_paths = self.all_audio_paths
      
    # Sort or shuffle the list
    if(shuffle):
      random.shuffle(self.audio_paths)
    else:
      self.audio_paths = sorted(self.audio_paths, key=lambda x: classes.index(str(x.relative_to(self.base_dir)).split('\\')[0]))

    self.label_dict = {label: idx for idx, label in enumerate(classes)} 
    #print(f"Label Dictionary: {self.label_dict}")

    # Balance the dataset 
    self.divisor = divisor
    self.balance_dataset()
    
  def _load_list(self, filename):
    filepath = self.base_dir / filename
    if filepath.exists():
      with filepath.open() as fileobj:
        return {self.base_dir / line.strip() for line in fileobj}
    return None
    
  #def _load_repc(self, filepath):
  #  with filepath.open() as fileobj:
  #    return {line for line in fileobj}

  def _load_repc(self, filepath):
    repcycles = []
    with filepath.open() as fileobj:
      for line in fileobj:
        line = line.strip()
        if line == '0':
          repcycles.append(0)
        else:
          repcycles.append(tuple(map(float, line.split(','))))
    return repcycles

  def __len__(self):
    return len(self.audio_paths)
  
  def __iter__(self):
    for idx in range(len(self.audio_paths)):
      yield self.__getitem__(idx)

  def __getitem__(self, idx):
    audio_path = self.audio_paths[idx]
    label = audio_path.parent.name  # Get the label as a string
    token = self.label_dict[label]  # Convert the label to an integer token
    waveform, _ = torchaudio.load(audio_path)
    
    repcycles_t = process_repcycles(waveform)
    return repcycles_t, token
    
  
  def getbyname(self, item_name):
    audio_path = self.base_dir / Path(item_name)
    if self.control:
      label = audio_path.parent.name  # Get the label as a string
      token = self.label_dict[label]  # Convert the label to an integer token
      waveform, _ = torchaudio.load(audio_path)
      waveform = self._crop(waveform)
      waveform_np = waveform.squeeze(0).numpy()
      segment_length = waveform.size(dim=1)//self.n_segments
      rms_values, _ = rms_over_windows(waveform_np, segment_length) #calculate RMS over time

            # Get Noise-to-Signal Ratio
      signal_rms = np.max(rms_values)
      noise_rms = np.mean(rms_values)
      nsr = noise_rms/ signal_rms

      # Calculate dynamic silence threshold based on NSR
      silence_threshold = min(nsr * 0.65, signal_rms*.50)

      out = torch.zeros(self.n_segments, self.vec_size)
      for segment_num in range(self.n_segments):
        if rms_values[segment_num] < silence_threshold:
          continue
        start_sample = segment_num*segment_length
        repc_wav = vectorize_f(waveform[0, start_sample+segment_length//2-50:start_sample+segment_length//2+50], self.vec_size)
        out[segment_num] = torch.tensor(repc_wav) 
      return out, token

    repc_path = self.repc_base_dir / audio_path.parent.stem / audio_path.with_suffix(".txt").name
    #"..\test_dataset\five\0a9f9af7_nohash_0.txt

    label = audio_path.parent.name  # Get the label as a string
    token = self.label_dict[label]  # Convert the label to an integer token
    waveform, _ = torchaudio.load(audio_path)
    waveform = self._crop(waveform)
    repcycles = self._load_repc(repc_path)
    repcycles_t = self._get_repcycles_wav(waveform, repcycles)
    return repcycles_t, token

  def plot_item(self, item_name, out_path=None,):
    if out_path is None:
      out_path = "./figures"

    audio_path = self.base_dir / Path(item_name)
    repcycles_t = self.getbyname(item_name)[0]

    out_path = (Path(out_path)) / Path(audio_path.parent.stem + "_" + audio_path.stem + f"_inputtensor{self.vec_size}.png")
    if self.control:
      out_path = out_path.with_stem(out_path.stem + "_control")
    flattened = repcycles_t.flatten().numpy()
    y_max = max(abs(np.amin(flattened)), abs(np.amax(flattened)))
    if y_max < 0.1: 
      y_max = 0.1
      plt.ylim(-y_max, y_max)
    
    # Plot the signal
    plt.figure(figsize=(100,10))
    #plt.vlines(range(0,len(flattened), self.vec_size), -y_max, y_max, colors='red', linestyles="dashed", linewidth=1)
    plt.xticks(range(0,len(flattened), self.vec_size))
    plt.plot(range(len(flattened)), flattened)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Tensor Signal Plot")
    plt.grid()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    if os.path.exists(out_path):
      if os.name == 'nt':  # For Windows
        os.startfile(out_path)

  def _crop(self, waveform, target_length=16000):
    if waveform.shape[1] < target_length:
      padding = target_length - waveform.shape[1]
      waveform = nnF.pad(waveform, (0, padding))
    else:
      waveform = waveform[:, :target_length]
    return waveform
  
  def _get_repcycles_wav(self, waveform, repcycles):

    # [0,0,0,0,0,(2025.2, 2125.3),(),(),0,0,0,0,0,0]
    #2D Array of the repcycles with size of the columns being the max repcycle size
    out = torch.zeros(len(repcycles), self.vec_size)

    for i, cycle in enumerate(repcycles):
      if cycle == 0:
        continue
      repc_wav = vectorize_f(waveform[0, cycle[0].__floor__():cycle[1].__floor__()], self.vec_size)
      out[i] = torch.tensor(repc_wav) 
    
    return out
  
  def balance_dataset(self):
    # Count the number of samples for each label
    label_counts = defaultdict(list)
    for path in self.audio_paths:
      label = path.parent.name
      label_counts[label].append(path)

    # Find the minimum count
    min_count = min(len(paths) for paths in label_counts.values())

    # Limit each label to the nearest multiple of the divisor
    balanced_audio_paths = []
    effective_count = (min_count // self.divisor) * self.divisor
    for paths in label_counts.values():
      balanced_audio_paths.extend(paths[:effective_count])

    self.audio_paths = balanced_audio_paths
    print(f"Balanced dataset to {effective_count} samples per label.")

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

def wav_value(waveform_arr, location:float):
  if location < 0 or location > len(waveform_arr) - 1:
    raise ValueError("Location is out of bounds.")
  frac = location % 1
  y1 = waveform_arr[location.__floor__()]
  y2 = waveform_arr[location.__floor__() ]
  return y1 + frac * (y2 - y1)

#def wav_value(waveform_arr, location:float):
#    # Ensure the location is within the bounds of the array
#  if location < 0 or location > len(waveform_arr) - 1:
#    raise ValueError("Location is out of bounds.")
#  indices = np.arange(len(waveform_arr))
#  return np.interp(location, indices, waveform_arr)

def vectorize_f(waveform_arr, n:int):
  if len(waveform_arr) == n:
    return waveform_arr
  if len(waveform_arr) < n:
    # resample the waveform to fit the desired size
    return resample(waveform_arr, n)
    # pad with zeros
    #return np.pad(waveform_arr.numpy(), (0,n - len(waveform_arr)))
  
  sample_points = np.linspace(0, len(waveform_arr) - 1, n)
  return np.array([wav_value(waveform_arr, point) for point in sample_points])

def resample(waveform_arr, n:int):
  resampled_waveform = np.zeros(n)
  sample_points = np.linspace(0, len(waveform_arr) - 1, n)
  for i, point in enumerate(sample_points):
    resampled_waveform[i] = wav_value(waveform_arr, point)
  return resampled_waveform

#### TESTING ####
if __name__ == "__main__":
  # Create an instance of the dataset
  if not len(sys.argv) > 3:
    print("Usage: python CustomSpeechCommands_R.py <num_segments> <file_name> <vec_size> \n Example: python CustomSpeechCommands.py nine\\0a2b400e_nohash_3.wav 40")

  base_dir = "..\\custom_speech_commands"
  dataset = CustomSpeechCommandsDataset_Repcycle(base_dir=base_dir, n_segments=int(sys.argv[1]), subset="training", shuffle=True, vec_size=int(sys.argv[3]), divisor=32)
  dataset.plot_item(item_name=sys.argv[2], out_path="./figures")

#  repcycles,_ = dataset.getbyname(sys.argv[2])
#  with open(f'{sys.argv[2]}.txt', 'w') as file:
#    for index, value in enumerate(repcycles):
#      if value:
#        file.write("%.3f, %.3f\n" % (value[0], value[1]))
#      else:
#        file.write("0\n")


  
