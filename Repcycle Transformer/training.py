# Description: This script trains an AST whose input has been modified to take audio insteasd of patches of images 
# Original code is based off a tutorial by Brian Pulfer
# https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Andrei Cartera -- Mar 2025

import time
import datetime
import numpy as np
import CustomSpeechCommands_Repcycle as SpeechCommands
from AudioTransformer import AudioTransformer 
from tqdm.notebook import tqdm, trange
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
NUM_CLASSES = 10

# Hyperparameters
N_SEGMENTS = 32
REPC_VEC_SIZE = 64

EPOCHS = 50
N_HEADS = 8
N_ENCODERS = 4
BATCH_SIZE = 64
HIDDEN_DIM = 32
DROPOUT = 0.15
ACTIVATION="gelu"
LR = 0.0009

today = datetime.date.today()

MODEL_PATH = f'models/({today})New_ATmodel_{N_SEGMENTS}SEG_{REPC_VEC_SIZE}VEC_E{EPOCHS}_{N_HEADS}_{N_ENCODERS}_B{BATCH_SIZE}_H{HIDDEN_DIM}.pth'

print(f"Model path: {MODEL_PATH}")


np.random.seed(0)
torch.manual_seed(0)


print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
  # Loading data

  print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
  model = AudioTransformer(N_SEGMENTS, REPC_VEC_SIZE, N_ENCODERS, HIDDEN_DIM, N_HEADS, NUM_CLASSES).to(device)

  train_set = SpeechCommands.CustomSpeechCommandsDataset_Repcycle("../custom_speech_commands", n_segments=N_SEGMENTS, shuffle=False, vec_size=REPC_VEC_SIZE, divisor=BATCH_SIZE)
  
  train_loader = DataLoader(train_set, shuffle=False, batch_size=BATCH_SIZE)

  # Defining model and training options

  # Training loop
  optimizer = Adam(model.parameters(), lr=LR)
  scheduler = lr_scheduler.LinearLR(optimizer)
  criterion = CrossEntropyLoss()

  model.train()  # Set the model to training mode                                     
  for epoch in trange(EPOCHS, desc="Training"):
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
      x, y = batch
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss = criterion(y_hat, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * x.size(0)
      
    train_loss /= len(train_loader.dataset) 
    scheduler.step(train_loss)
    #torch.cuda.empty_cache()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} loss: {train_loss:.2f}, LR: {current_lr}")
  
  torch.save(model.state_dict(), MODEL_PATH)
  print(f"Model saved as {MODEL_PATH}")

