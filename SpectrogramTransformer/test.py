import PySimpleGUI as sg
from pt_helper import process_audio, plot_spectrogram
import torch
from vit import MyViT
import numpy as np
import pyaudio
from PIL import Image

WINDOW_X = 500
WINDOW_Y = 780
FRAMES_PER_BUFFER = 160
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

commands = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# All the stuff inside your window.
layout = [  [sg.Column([[sg.ProgressBar(100, orientation='h', size=(WINDOW_X, 20), key='-PROGRESS-', expand_x=True)]], expand_x=True)],
            [sg.Frame('Output', [[
              sg.Multiline('',  size=(40, 2), key='-TEXTBOX-', expand_x=True,
                                expand_y=True, no_scrollbar=True, disabled=True,
                                justification='center',font=('Helvetica', 128) )]], 
              expand_x=True, size=(WINDOW_X, 200), key='-OUTPUT-')],
            [sg.Image(size=(100,100),key='-IMAGE-')],
            [sg.VPush()],
            [sg.B('Record')] ]
  

# Create the Window
window = sg.Window('Window Title', layout, size=(WINDOW_X,WINDOW_Y), resizable=True)

loaded_model = MyViT((1,200,200), n_patches=25, n_blocks=4, hidden_d=32, n_heads=4, out_d=10).to('cpu')
loaded_model.load_state_dict(torch.load('ASTmodel_E10_200_25_4_4_25_32.pth', weights_only=True))
loaded_model.eval()

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    #print("start recording...")

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
      data = stream.read(FRAMES_PER_BUFFER)
      frames.append(data)
      window['-PROGRESS-'].update_bar(i+1)
    # print("recording stopped")

    stream.stop_stream()
    stream.close()
    
    return np.frombuffer(b''.join(frames), dtype=np.int16)

def predict_mic():
  audio = record_audio()
  spec = process_audio(audio)
  #save_spectrogram(spec,'./output_spectrograms/out.jpg')
  image_data = plot_spectrogram(spec)
  window['-IMAGE-'].update(data=image_data)
  with torch.no_grad():
    prediction = loaded_model(spec.unsqueeze(0))
  label_pred = np.argmax(prediction, axis=1)
  command = commands[label_pred[0]]
  #print("Predicted label:", command)
  return command

# Event Loop to process "events" and get the "values" of the inputs
while True:
    
  event, values = window.read(timeout=10)
  if event in (sg.WINDOW_CLOSED, 'Exit'):
    break

  elif event == 'Record':
    command = predict_mic()
    window['-TEXTBOX-'].update(command)
    window['-PROGRESS-'].update_bar(0)

p.terminate()
#Path.Path.unlink('./output_spectrograms/out.jpg')
window.close()
