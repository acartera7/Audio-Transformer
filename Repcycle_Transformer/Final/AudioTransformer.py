# Description: This script trains an AST whose input has been modified to take audio insteasd of patches of images 
# Original code is based off a tutorial by Brian Pulfer
# https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Andrei Cartera -- Mar 2025

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
NUM_CLASSES = 10

# Hyperparameters
N_SEGMENTS = 32
REPC_VEC_SIZE = 80

EPOCHS = 100
N_HEADS = 8
N_ENCODERS = 4
BATCH_SIZE = 64
HIDDEN_DIM = 32
DROPOUT = 0.15
ACTIVATION="gelu"
LR = 0.0009

def get_positional_embeddings(sequence_length, d):
  result = torch.ones(sequence_length, d)
  for i in range(sequence_length):
    for j in range(d):
      result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
  return result

class NewMSA(nn.Module):
  def __init__(self, d, n_heads):
    super(NewMSA, self).__init__()
    self.multihead_attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, dropout=DROPOUT)
  
  def forward(self, x):
    # x: (batch, seq_len, d)
    # Transpose to (seq_len, batch, d) as required by nn.MultiheadAttention
    x_t = x.transpose(0, 1)
    # Compute self-attention with queries, keys and values set to x_t
    attn_output, _ = self.multihead_attn(x_t, x_t, x_t)
    # Transpose back to (batch, seq_len, d)
    return attn_output.transpose(0, 1)
  
class AudioTransformerBlock(nn.Module):
  def __init__(self, hidden_d, n_heads, mlp_ratio=4):
    super(AudioTransformerBlock, self).__init__()
    self.hidden_d = hidden_d
    self.n_heads = n_heads

    self.norm1 = nn.LayerNorm(hidden_d)
    self.mhsa = NewMSA(hidden_d, n_heads)  # Updated MSA module
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
  
class AudioTransformer(nn.Module):
  def __init__(self, n_segments, repc_vec_size , n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
    # Super constructor
    super(AudioTransformer, self).__init__()
    
    # Attributes
    self.n_segments = n_segments
    self.repc_vec_size  = repc_vec_size 
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.hidden_d = hidden_d
    
    # 1) Linear mapper
    self.input_d = repc_vec_size
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
    
    # 2) Learnable classification token
    self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
    
    # 3) Positional Embedding
    self.register_buffer(
      'positional_embeddings',
      get_positional_embeddings(n_segments + 1, hidden_d),
      persistent=False
    )
    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([AudioTransformerBlock(hidden_d, n_heads) for _ in range(n_blocks)])
    
    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_d, out_d),
        nn.Softmax(dim=-1)
    )

  def forward(self, audio):
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    tokens = self.linear_mapper(audio)
    
    # Adding classification token to the tokens
    tokens = torch.cat((self.class_token.expand(audio.shape[0], 1, -1), tokens), dim=1)
    
    # Adding positional embedding
    out = tokens + self.positional_embeddings.repeat(audio.shape[0], 1, 1)
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)
        
    # Getting the classification token only
    out = out[:, 0]
    
    return self.mlp(out) # Map to output dimension, output category distribution
