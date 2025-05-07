# %%
# Visual Transformer based off the original publication An Image is Worth 16x16 Words:
# https://arxiv.org/pdf/2010.11929

# Code is taken from a tutorial by Brian Pulfer
# https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Andrei Cartera -- Oct 2024


import numpy as np
import CustomSpeechCommands as SpeechCommands
from tqdm.notebook import tqdm, trange
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


np.random.seed(0)
torch.manual_seed(0)

print(torch.__version__)

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

EPOCHS = 30

CHW = (1,224,224)
NUM_CLASSES = 10
N_PATCHES = 28
N_HEADS = 8
N_ENCODERS = 4
BATCH_SIZE = 32
HIDDEN_DIM = 16

DROPOUT = 0.05
ACTIVATION="gelu"
LR = 0.001


# %%
def patchify(images, n_patches):
  n, c, h, w = images.shape

  assert h == w, "Patchify method is implemented for square images only"

  patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
  patch_size = h // n_patches

  for idx, image in enumerate(images):
    for i in range(n_patches):
      for j in range(n_patches):
        patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
        patches[idx, i * n_patches + j] = patch.flatten()
  return patches

# %%
def optimized_patchify(images, n_patches):
  n, c, h, w = images.shape
  assert h == w, "Patchify method is implemented for square images only"

  patch_size = h // n_patches
  patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
  patches = patches.contiguous().view(n, c, -1, patch_size, patch_size)
  patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(n, -1, patch_size * patch_size * c)
  
  return patches

# %%
def get_positional_embeddings(sequence_length, d):
  result = torch.ones(sequence_length, d)
  for i in range(sequence_length):
    for j in range(d):
      result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
  return result

# %%
image = torch.rand(1,1,224,224)
patches = optimized_patchify(image, 28)
print(patches.shape)

# %%
class MyMSA(nn.Module):
  def __init__(self, d, n_heads=2):
    super(MyMSA, self).__init__()
    self.d = d
    self.n_heads = n_heads

    assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

    d_head = int(d / n_heads)
    self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
    self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
    self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
    self.d_head = d_head
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, sequences):
    # Sequences has shape (N, seq_length, token_dim)
    # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
    # And come back to    (N, seq_length, item_dim)  (through concatenation)
    result = []
    for sequence in sequences:
      seq_result = []
      for head in range(self.n_heads):
        q_mapping = self.q_mappings[head]
        k_mapping = self.k_mappings[head]
        v_mapping = self.v_mappings[head]

        seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
        q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

        attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
        seq_result.append(attention @ v)
      result.append(torch.hstack(seq_result))
    return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# %%
class MyViTBlock(nn.Module):
  def __init__(self, hidden_d, n_heads, mlp_ratio=4):
    super(MyViTBlock, self).__init__()
    self.hidden_d = hidden_d
    self.n_heads = n_heads

    self.norm1 = nn.LayerNorm(hidden_d)
    self.mhsa = MyMSA(hidden_d, n_heads)
    self.norm2 = nn.LayerNorm(hidden_d)
    self.mlp = nn.Sequential(
      nn.Linear(hidden_d, mlp_ratio * hidden_d),
      nn.GELU(),
      nn.Linear(mlp_ratio * hidden_d, hidden_d)
    )

  def forward(self, x):
    out = x + self.mhsa(self.norm1(x))
    out = out + self.mlp(self.norm2(out))
    return out

# %%
class MyViT(nn.Module):
  def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
    # Super constructor
    super(MyViT, self).__init__()
    
    # Attributes
    self.chw = chw # ( C , H , W )
    self.n_patches = n_patches
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.hidden_d = hidden_d
    
    # Input and patches sizes
    assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

    # 1) Linear mapper
    self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
    
    # 2) Learnable classification token
    self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
    
    # 3) Positional embedding
    self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
    
    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
    
    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_d, out_d),
        nn.Softmax(dim=-1)
    )

  def forward(self, images):
    # Dividing images into patches
    n, c, h, w = images.shape
    patches = optimized_patchify(images, self.n_patches).to(self.positional_embeddings.device)
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    tokens = self.linear_mapper(patches)
    
    # Adding classification token to the tokens
    tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
    
    # Adding positional embedding
    out = tokens + self.positional_embeddings.repeat(n, 1, 1)
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)
        
    # Getting the classification token only
    out = out[:, 0]
    
    return self.mlp(out) # Map to output dimension, output category distribution

# %%
def main():
  # Loading data

  print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
  model = MyViT(CHW, N_PATCHES, N_ENCODERS, HIDDEN_DIM, N_HEADS, NUM_CLASSES).to(device)
  #model.load_state_dict(torch.load('my_model3_2_Transfer.pth',map_location=torch.device('cpu')))
  #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

  train_set = SpeechCommands.CustomSpeechCommandsDataset("../datasets/custom_speech_commands", shuffle=True, divisor=BATCH_SIZE, out_size=(224,224))
  test_set = SpeechCommands.CustomSpeechCommandsDataset("../datasets/custom_speech_commands", subset="testing", shuffle=True,divisor=BATCH_SIZE, out_size=(224,224))
  
  train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

  # Defining model and training options

  # Training loop
  optimizer = Adam(model.parameters(), lr=LR)
  criterion = CrossEntropyLoss()
  #scheduler = lr_scheduler.LinearLR(optimizer)
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
                                             
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
    torch.cuda.empty_cache()

    print(f"Epoch {epoch + 1}/{EPOCHS} loss: {train_loss:.2f}")

  torch.save(model.state_dict(), 'ASTmodel_E30_224_28_8_4_32_16.pth')

  # Test loop
  with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader, desc="Testing"):
      x, y = batch

      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss = criterion(y_hat, y)
      test_loss += loss.detach().cpu().item() / len(test_loader)

      correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
      total += len(x)
      
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")

# %% [markdown]
# ## TRAINING LOOP

# %%
if __name__ == "__main__":  
  main()

# %% [markdown]
# ### TESTING

# %%
if __name__ == "__main__":  
  model = MyViT((1,200,200), n_patches=25, n_blocks=4, hidden_d=32, n_heads=4, out_d=10).to(device)
  model.load_state_dict(torch.load('ASTmodel_E10_200_25_4_4_25_32.pth', weights_only=True))

  test_set = SpeechCommands.CustomSpeechCommandsDataset("../datasets/custom_speech_commands", subset="testing", shuffle=True,divisor=30, out_size=(200,200))
  test_loader = DataLoader(test_set, shuffle=False, batch_size=30)

  criterion = CrossEntropyLoss()

  # Test loop
  with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader, desc="Testing"):
      x, y = batch

      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss = criterion(y_hat, y)
      test_loss += loss.detach().cpu().item() / len(test_loader)

      correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
      total += len(x)
      
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")

# %%
#if __name__ == '__main__':
#  # Current model
#  model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10,).to(device)
#  model.load_state_dict(torch.load('my_model3_2_Transfer.pth',map_location=torch.device('cpu')))
#
#  #imgs = torch.randn(7, 1, 28, 28).to(device) # Dummy images
#  #print(model(imgs).shape) # torch.Size([7, 49, 16])


# %%
#if __name__ == '__main__':
#  train_set = SpeechCommands.CustomSpeechCommandsDataset("../datasets/custom_speech_commands",out_size=(28,28))
#  _, label = train_set[0]
#  print(label)


