# models/constraint_decoder.py
import torch
import torch.nn as nn

class ConstraintDecoder(nn.Module):
    def __init__(self, embed_dim, proprio_dim, attn_dim=64):
        super().__init__()
        # Project high-level constraint to attention dimension
        self.c_high_proj = nn.Linear(embed_dim, attn_dim)
        # Project state to attention dimension
        self.state_proj = nn.Linear(proprio_dim, attn_dim)
        
        # Layer to generate the low-level constraint
        self.low_constraint = nn.Linear(embed_dim, embed_dim)

    def forward(self, c_high, state):
        # Project both inputs into a shared space
        c_high_q = self.c_high_proj(c_high) # Query
        state_k = self.state_proj(state)    # Key

        # Compute attention score (dot product)
        attn = torch.sum(c_high_q * state_k, dim=-1)
        c_low = self.low_constraint(c_high) * attn.unsqueeze(-1)
        return c_low
