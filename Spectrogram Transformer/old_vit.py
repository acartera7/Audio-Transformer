from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

RANDOM_SEED = 42
BATCH_SIZE = 25
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 16
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 12
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION="gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS # 768
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 # 196

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class PatchEmbedding(nn.Module):
  def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
    super().__init__()
    self.patcher = nn.Sequential(
      nn.Conv2d(
        in_channels=in_channels,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size,
      ),                  
      nn.Flatten(2))

    self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
    self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)

    x = self.patcher(x).permute(0, 2, 1)
    x = torch.cat([cls_token, x], dim=1)
    x = self.position_embeddings + x 
    x = self.dropout(x)
    return x
  
class ViT(nn.Module):
    def __init__(self, num_patches, num_classes, patch_size, embed_dim, num_encoders, num_heads, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x

#model = ViT(NUM_PATCHES, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODERS, NUM_HEADS, DROPOUT, ACTIVATION, IN_CHANNELS).to(device)
#x = torch.randn(BATCH_SIZE, IN_CHANNELS, 224, 224).to(device)
#print(model(x).shape)
