import io
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as nnF
import torchaudio.transforms as T
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
CHW = (1,200,200)

def save_spectrogram(spectrogram, file_path):
  spectrogram = spectrogram.cpu()  # Move to CPU if on GPU
  spectrogram = spectrogram.log2()
  
  #plt.figure(figsize=(5, 5))  # Make the figure a square
  plt.imshow(spectrogram[0, :, :].numpy(), cmap='inferno', origin='lower')
  #plt.gca().invert_yaxis()  # Invert y-axis
  plt.gca().set_axis_off()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0, 0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
  plt.close()
  print(f"Spectrogram saved to {file_path}")

def plot_spectrogram(spectrogram):
  spectrogram = spectrogram.cpu()  # Move to CPU if on GPU
  spectrogram = spectrogram.log2()
  
  buf = io.BytesIO()
  #plt.figure(figsize=(5, 5))  # Make the figure a square
  plt.imshow(spectrogram[0, :, :].numpy(), cmap='inferno', origin='lower')
  #plt.gca().invert_yaxis()  # Invert y-axis
  plt.axis('off')  # Remove axes
  plt.gca().set_axis_off()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0, 0) 
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
  plt.close() 
  buf.seek(0) 
  return buf.getvalue()

def resize_spectrogram(spectrogram, target_size=(224, 224)):
  # Interpolate to the target size
  resized_spectrogram = nnF.interpolate(spectrogram.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
  return resized_spectrogram.squeeze(0)

def crop(waveform, target_length=16000):
  if waveform.shape[1] < target_length:
    padding = target_length - waveform.shape[1]
    waveform = nnF.pad(waveform, (0, padding))
  else:
    waveform = waveform[:, :target_length]
  return waveform

def process_audio(waveform):
  spectrogram_transform = T.MelSpectrogram(
    n_fft=1024,          # Adjust size of FFT to capture more details
    hop_length=128,     # Set the hop length to control overlap
    normalized=True,
    onesided=True,
    power=1.0
  )
  waveform =  waveform / 32768
  #waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
  waveform = torch.tensor(waveform,dtype=torch.float32).unsqueeze(0)
  waveform = crop(waveform)
  spectrogram = spectrogram_transform(waveform)
  spectrogram = resize_spectrogram(spectrogram, target_size=CHW[1:])
  return spectrogram