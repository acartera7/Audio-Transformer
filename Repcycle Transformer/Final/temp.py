import sys
import threading
import pyaudio

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QProgressBar,
    QTextEdit,
    QLabel,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer, Signal


class SpeechRecognitionApp(QWidget):
  # Define a signal to safely update the recognized text from the worker thread.
  recognized_signal = Signal(str)

  def __init__(self):
    super().__init__()

    # Configure the main window.
    self.setWindowTitle("Audio Transformer")
    self.setGeometry(100, 100, 400, 300)

    # Create the main layout.
    layout = QVBoxLayout()

    # Add a progress bar at the top.
    self.progress_bar = QProgressBar(self)
    self.progress_bar.setAlignment(Qt.AlignCenter)
    layout.addWidget(self.progress_bar)

    # Add a text output area in the middle.
    self.text_output = QTextEdit(self)
    self.text_output.setPlaceholderText("Output will appear here...")
    self.text_output.setReadOnly(True)
    layout.addWidget(self.text_output)

    # Add an image (or waveform) placeholder.
    self.image_frame = QLabel(self)
    self.image_frame.setStyleSheet("border: 1px solid black;")
    self.image_frame.setFixedHeight(100)
    self.image_frame.setAlignment(Qt.AlignCenter)
    self.image_frame.setText("Waveform display placeholder")
    layout.addWidget(self.image_frame)

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
    self.recognized_signal.connect(self.display_recognition_result)

  def start_recording(self):
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

    # Process the audio frames here.
    # For this example, we simulate a recognition result.
    recognized_text = "Recognized speech text: Hello, world!"

    # Emit the signal to update the UI in a thread-safe manner.
    self.recognized_signal.emit(recognized_text)

  def update_progress(self):
    self.progress_value += 1
    self.progress_bar.setValue(self.progress_value)

    # Stop the timer when the progress reaches 100%.
    if self.progress_value >= 100:
        self.timer.stop()

  def display_recognition_result(self, text):
    self.text_output.setText(text)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = SpeechRecognitionApp()
  window.show()
  sys.exit(app.exec())
