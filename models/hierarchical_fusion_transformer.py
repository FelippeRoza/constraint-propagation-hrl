import torch
import torch.nn as nn
from einops import rearrange

class HierarchicalFusionTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=1024)
            for _ in range(num_layers)
        ])
        self.constraint_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        c_high = self.constraint_projection(x.mean(dim=1))
        return x, c_high
