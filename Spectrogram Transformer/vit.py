# Visual Transformer based off the original publication An Image is Worth 16x16 Words:
# https://arxiv.org/pdf/2010.11929

# Code is taken from a tutorial by Brian Pulfer
# https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Andrei Cartera -- Oct 2024


import numpy as np

import torch
import torch.nn as nn

EPOCHS = 30

CHW = (1,224,224)
NUM_CLASSES = 10
N_PATCHES = 28
N_HEADS = 8
N_ENCODERS = 4
BATCH_SIZE = 1
HIDDEN_DIM = 16

DROPOUT = 0.05
ACTIVATION="gelu"
LR = 0.001

np.random.seed(0)
torch.manual_seed(0)

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def optimized_patchify(images, n_patches):
  n, c, h, w = images.shape
  assert h == w, "Patchify method is implemented for square images only"

  patch_size = h // n_patches
  patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
  patches = patches.contiguous().view(n, c, -1, patch_size, patch_size)
  patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(n, -1, patch_size * patch_size * c)
  
  return patches

def get_positional_embeddings(sequence_length, d):
  result = torch.ones(sequence_length, d)
  for i in range(sequence_length):
    for j in range(d):
      result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
  return result

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
  
