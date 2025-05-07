import io
import sys
import math
import threading

from PySide6.QtGui import QPixmap, QImage, QPainter, QFont
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QProgressBar,
    QTextEdit,
    QLabel,
    QPushButton,
    QGraphicsView, 
    QGraphicsScene
)

import matplotlib.pyplot as plt
import torch
import pyaudio
import numpy as np
import torch.nn.functional as nnF
from PIL import Image

from AudioTransformer import AudioTransformer
from Inference_process import vectorize_f
from Inference_process import process_repcycles as get_repcycles; 

WINDOW_X = 1000
WINDOW_Y = 400
FRAMES_PER_BUFFER = 160
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

commands = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def crop(waveform, target_length=16000):
  if len(waveform) < target_length:
    padding = target_length - len(waveform)
    waveform = np.pad(waveform, (0, padding), mode='constant')
  else:
    waveform = waveform[:target_length]
  return waveform

def get_repcycles_wav(waveform, repcycles, vec_size=48):

  # [0,0,0,0,0,(2025.2, 2125.3),(),(),0,0,0,0,0,0]
  #2D Array of the repcycles with size of the columns being the max repcycle size
  out = torch.zeros(len(repcycles), vec_size)

  for i, cycle in enumerate(repcycles):
    if not cycle:
      continue
    repc_wav = vectorize_f(waveform[ math.floor(cycle[0]):math.floor(cycle[1])], vec_size)
    out[i] = torch.tensor(repc_wav) 
  
  return out
  
REPC_VEC_SIZE = 64

MODEL_PATH='N_ATmodel_32SEG_64VEC_E100_8_4_B64_H32_2.pth'
model = AudioTransformer(n_segments=32, repc_vec_size=64, n_blocks=4, hidden_d=32, n_heads=8, out_d=10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True,map_location=torch.device(device)))
print(f"Model loaded from {MODEL_PATH}")
model.eval()  # Set the model to evaluation mode

# Create the Window
class SpeechRecognitionApp(QWidget):
  # Define a signal to safely update the recognized text from the worker thread.
  audio_buffer_signal = Signal(object)

  def __init__(self):
    super().__init__()

    # Configure the main window.
    self.setWindowTitle("Audio Transformer")
    self.setGeometry(100, 100, WINDOW_X, WINDOW_Y)

    # Create the main layout.
    layout = QVBoxLayout(self)

    # Add a progress bar at the top.
    self.progress_bar = QProgressBar(self)
    self.progress_bar.setAlignment(Qt.AlignCenter)
    layout.addWidget(self.progress_bar)

    # Create a horizontal layout for the label and text editor
    outputtext_layout = QHBoxLayout()

    # Add a label to the left of the text editor
    self.text_label = QLabel("Output")
    font = QFont()
    font.setPointSize(20)  # Set font size to 20
    font.setBold(True)  # Set bold font
    self.text_label.setFont(font)
    self.text_label.setFixedSize(100,60)
    outputtext_layout.addWidget(self.text_label)
    
    # Add the text editor next to the label
    self.text_output = QTextEdit(self)
    self.text_output.setReadOnly(True)
    self.text_output.setFixedSize(100, 100)
    self.text_output.setFontPointSize(64)  # Set font size to 64
    self.text_output.setAlignment(Qt.AlignHCenter)
    outputtext_layout.addWidget(self.text_output)

    outputtext_layout.setAlignment(Qt.AlignHCenter)  # Align the text editor to the center of the horizontal layout
    layout.addLayout(outputtext_layout) #Add the horizontal layout to the main layout

    # Add an image for waveform visualization.
    self.image_view = QGraphicsView(self)
    self.image_scene = QGraphicsScene()
    self.image_view.setScene(self.image_scene)
    self.image_view.setRenderHint(QPainter.Antialiasing)
    layout.addWidget(self.image_view)

    # Add a record button at the bottom.
    self.record_button = QPushButton("Record", self)
    self.record_button.clicked.connect(self.start_recording)
    layout.addWidget(self.record_button)

    # Set the main layout.
    self.setLayout(layout)

    # Create a timer for updating the progress bar (every 10 ms).
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update_progress)
    self.progress_value = 0

    # Connect the custom signal to the handler to update the text output.
    self.audio_buffer_signal.connect(self.run_prediction)
  
  def wheelEvent(self, event):
      """
      Handle mouse wheel events to zoom in/out on the image.
      """
      zoom_factor = 1.15  # Adjust zoom sensitivity
      if event.angleDelta().y() > 0:  # Zoom in
        self.image_view.scale(zoom_factor, zoom_factor)
      else:  # Zoom out
        self.image_view.scale(1 / zoom_factor, 1 / zoom_factor)

  def start_recording(self):
    self.record_button.setEnabled(False)
    # Reset the progress bar and clear any previous output.
    self.progress_value = 0
    self.progress_bar.setValue(0)
    self.text_output.clear()

    # Start the audio recording in a separate thread.
    threading.Thread(target=self.record_audio, daemon=True).start()

    # Start the timer to update the progress bar.
    self.timer.start(10)  # 10 ms interval (totaling roughly 1 second).

  def record_audio(self):
    # PyAudio settings.
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1

    # Initialize PyAudio.
    p = pyaudio.PyAudio()

    # Open the stream for recording (default input device).
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    # Calculate the number of chunks needed for 1 second of audio.
    num_chunks = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(num_chunks):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream.
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert the recorded frames to a waveform.
    waveform = np.frombuffer(b''.join(frames), dtype=np.int16)
    waveform = (waveform / np.iinfo(np.int16).max).astype(np.float32)

    # Emit the waveform to update the UI in a thread-safe manner.
    self.audio_buffer_signal.emit(waveform)

  def update_progress(self):
    self.progress_value += 1
    self.progress_bar.setValue(self.progress_value)

    # Stop the timer when the progress reaches 100%.
    if self.progress_value >= 100:
      self.timer.stop()

  def run_prediction(self, waveform):
    repcycles = get_repcycles(waveform)
    input = get_repcycles_wav(waveform, repcycles, vec_size=REPC_VEC_SIZE).to(device) # get the repcycles for the waveform and vectorize them

    # Add a batch dimension to the input tensor
    input = input.unsqueeze(0)  # Shape: (1, sequence_length, feature_size)

    with torch.no_grad():
      output = model(input)
    np_output = (output.cpu()).numpy()[0]
    label_pred = np.argmax(np_output)  # Get the predicted label
    command = commands[label_pred]
    self.text_output.setText(command)

    # Generate an image from the waveform 
    #plt.figure(figsize=(6, 2))
    fig = plt.figure(figsize=(50, 10))  # Adjust the figure size and DPI as needed
    ax = fig.add_subplot(111)
    plt.plot(range(len(waveform)), waveform)
    y_max = max(abs(np.amin(waveform)), abs(np.amax(waveform)))

    if y_max < 0.1: 
      y_max = 0.1
      plt.ylim(-y_max, y_max)

    for repcycle in repcycles:
      #ax.fill_between(t, 1, where=s > 0, facecolor='green', alpha=.5)
      #ax.fill_between(t, -1, where=s < 0, facecolor='red', alpha=.5)
      ax.fill_between(repcycle, -y_max, y_max, facecolor='green', alpha=.25)
    plt.xticks(np.arange(len(waveform), 100))
    plt.grid()

    #plt.plot(waveform, color='blue')
    #plt.title("Waveform Visualization")
    #plt.xlabel("Sample Index")
    #plt.ylabel("Amplitude")
    #plt.tight_layout()

    # Save the image to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()

    # Convert the buffer to a QImage and display it in image_frame
    qimage = QImage.fromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimage)
        # Set the pixmap into the QGraphicsScene
    self.image_scene.clear()  # Clear previous image
    self.image_scene.addPixmap(pixmap)
    self.image_view.fitInView(self.image_scene.sceneRect(), Qt.KeepAspectRatio)
    
    # Re-enable the record button
    self.record_button.setEnabled(True)

    

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = SpeechRecognitionApp()
  window.show()
  sys.exit(app.exec())
