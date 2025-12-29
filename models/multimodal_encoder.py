import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalEncoder(nn.Module):
    def __init__(self, proprio_dim, embed_dim=512, freeze_backbones=True):
        super().__init__()
        self.freeze_backbones = freeze_backbones
        self.vision_encoder = AutoModel.from_pretrained("facebook/dinov2-small")  # image embeddings
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")  # text embeddings
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64), nn.ReLU(), nn.Linear(64, 128)
        )

        # Calculate the combined dimension after concatenation
        vision_dim = self.vision_encoder.config.hidden_size # 384
        text_dim = self.text_encoder.config.hidden_size   # 768
        proprio_out_dim = 128
        combined_dim = vision_dim + text_dim + proprio_out_dim # 1280

        # Add a projection layer to map the combined embedding to the desired dimension
        self.projection = nn.Linear(combined_dim, embed_dim)

        if self.freeze_backbones:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbones:
            self.vision_encoder.eval()
            self.text_encoder.eval()

    def forward(self, vision, text, text_attention_mask, proprio):
        v = self.vision_encoder(vision).last_hidden_state.mean(dim=1)
        t = self.text_encoder(text, attention_mask=text_attention_mask).last_hidden_state.mean(dim=1)
        p = self.proprio_encoder(proprio)
        concatenated = torch.cat((v, t, p), dim=-1)
        return self.projection(concatenated)
