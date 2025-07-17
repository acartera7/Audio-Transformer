import io
import sys
import math
import threading
import time

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
    QGraphicsScene,
    QSpacerItem
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

WINDOW_X = 1400
WINDOW_Y = 800
FRAMES_PER_BUFFER = 160
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

commands = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

times = {}

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
    main_layout = QHBoxLayout()

    ############################################################################ 
    #### LAYOUT 1
    ############################################################################ 
    layout1 = QVBoxLayout()

    # Add a progress bar at the top.
    self.progress_bar = QProgressBar(self)
    self.progress_bar.setAlignment(Qt.AlignCenter)
    layout1.addWidget(self.progress_bar)

    # Create a horizontal layout for the label and text editor
    outputtext_layout = QHBoxLayout()

    # Add a label to the left of the text editor
    text_label = QLabel("Output")
    font = QFont()
    font.setPointSize(15)  # Set font size to 20
    font.setBold(True)  # Set bold font
    text_label.setFont(font)
    text_label.setFixedSize(100,100)
    outputtext_layout.addWidget(text_label)
    
    # Add the text editor next to the label
    self.text_output = QTextEdit(self)
    self.text_output.setReadOnly(True)
    self.text_output.setFixedSize(100, 100)
    self.text_output.setFontPointSize(64)  # Set font size to 64
    self.text_output.setAlignment(Qt.AlignHCenter)
    outputtext_layout.addWidget(self.text_output)

    outputtext_layout.addItem(QSpacerItem(15, 100))  # Add some space between the label and text editor

    timing_label = QLabel("Timing:")
    timing_label.setFont(font)
    timing_label.setFixedSize(100,100)
    outputtext_layout.addWidget(timing_label)

    self.timing_text = QTextEdit(self)
    self.timing_text.setReadOnly(True)
    self.timing_text.setFixedSize(400, 100)
    outputtext_layout.addWidget(self.timing_text)

    outputtext_layout.setAlignment(Qt.AlignHCenter)  # Align the text editor to the center of the horizontal layout
    layout1.addLayout(outputtext_layout) #Add the horizontal layout to the main layout

    changeviewbuttons_layout = QHBoxLayout()

    self.waveformview_button = QPushButton("Waveform", self)
    self.waveformview_button.setEnabled(False)  # Disable the button initially
    self.waveformview_button.clicked.connect(lambda: self.change_view(self.waveformview_button))
    changeviewbuttons_layout.addWidget(self.waveformview_button)

    self.inputtensorview_button = QPushButton("Input Tensor", self)
    self.inputtensorview_button.clicked.connect(lambda: self.change_view(self.inputtensorview_button))
    changeviewbuttons_layout.addWidget(self.inputtensorview_button)

    changeviewbuttons_layout.setAlignment(Qt.AlignHCenter) 
    layout1.addLayout(changeviewbuttons_layout) #Add the horizontal layout to the main layout

    # Add an image for waveform visualizations and other info.
    self.wav_image_scene = QGraphicsScene()
    self.tensor_image_scene = QGraphicsScene()
    self.image_view = QGraphicsView(self)
    self.image_view.setScene(self.wav_image_scene)
    self.image_view.setRenderHint(QPainter.Antialiasing)
    layout1.addWidget(self.image_view)

    # Add a record button at the bottom.
    self.record_button = QPushButton("Record", self)
    self.record_button.setFixedHeight(50)
    self.record_button.clicked.connect(self.start_recording)
    layout1.addWidget(self.record_button)
    
    main_layout.addLayout(layout1)

    ############################################################################ 
    #### LAYOUT 2
    ############################################################################ 
    layout2 = QVBoxLayout()

    rawrepcycle_label = QLabel("Repcycles:")
    rawrepcycle_label.setFont(font)
    rawrepcycle_label.setFixedSize(100,100)
    rawrepcycle_label.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
    #layout2.setAlignment(Qt.AlignHCenter)  
    layout2.addWidget(rawrepcycle_label)

    self.rawrepcycles_text = QTextEdit(self)
    self.rawrepcycles_text.setReadOnly(True)
    self.rawrepcycles_text.setFixedWidth(150)
    layout2.addWidget(self.rawrepcycles_text)

    main_layout.addLayout(layout2)  
    # Set the main layout.
    self.setLayout(main_layout)

    # Create a timer for updating the progress bar (every 10 ms).
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update_progress)
    self.progress_value = 0

    # Connect the custom signal to the prediction function.
    self.audio_buffer_signal.connect(self.run_prediction)

  def change_view(self, button):
    """
    Change the view of the image display based on the button clicked.
    """
    if button == self.waveformview_button:
      self.waveformview_button.setEnabled(False)
      self.inputtensorview_button.setEnabled(True)

      self.image_view.setScene(self.wav_image_scene)
      #self.image_view.fitInView(self.wav_image_scene.sceneRect(), Qt.KeepAspectRatio)
    
    elif button == self.inputtensorview_button:
      self.inputtensorview_button.setEnabled(False)
      self.waveformview_button.setEnabled(True)

      self.image_view.setScene(self.tensor_image_scene)
      #self.image_view.fitInView(self.tensor_image_scene.sceneRect(), Qt.KeepAspectRatio)

  def mousePressEvent(self, event):
    """
    Handle mouse press events to initiate panning.
    """
    if event.button() == Qt.LeftButton:
      # Check if the mouse is over an image view
      if self.image_view.underMouse():
        self.last_mouse_position = event.position()
        self.is_panning = True
        self.grabMouse()  # Grab the mouse to ensure events are captured
      else:
        self.is_panning = False

  def mouseMoveEvent(self, event):
    """
    Handle mouse move events to perform panning when the left mouse button is pressed.
    """
    if self.is_panning:
      delta = event.position() - self.last_mouse_position
      self.last_mouse_position = event.position()

      # Check which image view the mouse is hovering over
      if self.image_view.underMouse():
        # Adjust the view's transformation to pan
        self.image_view.horizontalScrollBar().setValue(
          self.image_view.horizontalScrollBar().value() - delta.x()
        )
        self.image_view.verticalScrollBar().setValue(
          self.image_view.verticalScrollBar().value() - delta.y()
        )
  def mouseReleaseEvent(self, event):
    """
    Handle mouse release events to stop panning.
    """
    if event.button() == Qt.LeftButton:
      self.is_panning = False
      self.releaseMouse()  # Release the mouse when panning stops
      self.is_panning = False

  def wheelEvent(self, event):
    """
    Handle mouse wheel events to zoom in/out on the image view currently being hovered over.
    If the Ctrl key is held, zoom the y-axis instead.
    If the Shift key is held, zoom the x-axis instead.
    Otherwise, zoom both axes.
    """
    zoom_factor = 1.15  # Adjust zoom sensitivity

    # Check which image view the mouse is hovering over
    if self.image_view.underMouse():

    # Check if the Ctrl key is pressed
    #if QApplication.keyboardModifiers() == Qt.ControlModifier:
    #  # Zoom the y-axis only
    #  if event.angleDelta().y() > 0:  # Zoom in
    #    target_view.scale(1, zoom_factor)
    #  else:  # Zoom out
    #    target_view.scale(1, 1 / zoom_factor)
    #elif QApplication.keyboardModifiers() == Qt.ShiftModifier:
    #  # Zoom the y-axis only
    #  if event.angleDelta().y() > 0:  # Zoom in
    #    target_view.scale(zoom_factor, 1)
    #  else:  # Zoom out
    #    target_view.scale(1 / zoom_factor, 1)
    #else:
    # Perform regular zoom on both axes
      if event.angleDelta().y() > 0:  # Zoom in
        self.image_view.scale(zoom_factor, zoom_factor)
      else:  # Zoom out
        self.image_view.scale(1 / zoom_factor, 1 / zoom_factor)

  def generate_input_tensor_image(self, input_tensor):
    flattened = input_tensor.flatten().numpy()
    
    # Plot the signal
    plt.figure(figsize=(60,10))

    #y_max = max(abs(np.amin(flattened)), abs(np.amax(flattened)))
    #if y_max < 0.1: 
    #  y_max = 0.1
    #  plt.ylim(-y_max, y_max)
    plt.ylim(-1, 1)
    #plt.vlines(range(0,len(flattened), self.vec_size), -y_max, y_max, colors='red', linestyles="dashed", linewidth=1)
    plt.xticks(range(0, len(flattened), REPC_VEC_SIZE))
    plt.plot(range(len(flattened)), flattened)
    #plt.xlabel("Index")
    #plt.ylabel("Value")
    #plt.title("Tensor Signal Plot")
    plt.grid()
    plt.gca().tick_params(axis='x', colors='red')  # Set the color of x-axis tick lines to red
    plt.gca().xaxis.grid(True, color='red', linestyle='--', linewidth=1)  # Set the color of x-axis grid lines to red
    # Save the image to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a QImage and display it in image_frame
    qimage = QImage.fromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

  def generate_waveform_image(self, waveform, repcycles):
    # Generate an image from the waveform
    fig = plt.figure(figsize=(60, 10))  # Adjust the figure size and DPI as needed
    ax = fig.add_subplot(111)
    plt.plot(range(len(waveform)), waveform)

    plt.ylim(-1, 1)
    for repcycle in repcycles:
      ax.fill_between(repcycle, -1, 1, facecolor='green', alpha=.25)

    # Set x-axis ticks and labels
    plt.xticks(range(0, len(waveform), 500))  # Divide the x-axis into 10 intervals

    plt.grid()

    # Save the image to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a QImage and display it in image_frame
    qimage = QImage.fromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

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
    CHUNK = 1000
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
    start_time = time.perf_counter() 
    repcycles = get_repcycles(waveform) # find recpycles in the waveform
    end_time = time.perf_counter()
    times['Finding Repcycles'] = end_time - start_time

    start_time = time.perf_counter()
    input = get_repcycles_wav(waveform, repcycles, vec_size=REPC_VEC_SIZE).to(device) # get the repcycles wave data and vectorize them
    end_time = time.perf_counter()
    times['Getting Repcycles Data'] = end_time - start_time

    # Add a batch dimension to the input tensor
    input = input.unsqueeze(0)  # Shape: (1, sequence_length, feature_size)

    with torch.no_grad():
      start_time = time.perf_counter()
      output = model(input)
      end_time = time.perf_counter()
      times['Model Inference'] = end_time - start_time

    np_output = (output.cpu()).numpy()[0]
    label_pred = np.argmax(np_output)  # Get the predicted label
    command = commands[label_pred]
    self.text_output.setText(command)

    formatted_repcycles = "\n".join(f"{i}: ({cycle[0]:.2f}, {cycle[1]:.2f})" if cycle else f"{i}: 0" for i, cycle in enumerate(repcycles))
    self.rawrepcycles_text.setText(formatted_repcycles)

    self.timing_text.clear()
    for key, value in times.items():
      self.timing_text.append(f"{key}: {value:.4f} seconds")

    wav_pixmap = self.generate_waveform_image(waveform, repcycles)
    tensor_pixmap = self.generate_input_tensor_image(input.to('cpu'))  # Get the first (and only) batch
    
    # Set the pixmap into the QGraphicsScene
    self.wav_image_scene.clear()  # Clear previous image
    self.wav_image_scene.addPixmap(wav_pixmap)

    self.tensor_image_scene.clear()  # Clear previous tensor image
    self.tensor_image_scene.addPixmap(tensor_pixmap)

    self.image_view.fitInView(self.wav_image_scene.sceneRect(), Qt.KeepAspectRatio)

    # Re-enable the record button
    self.record_button.setEnabled(True)
    

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = SpeechRecognitionApp()
  window.show()
  sys.exit(app.exec())
