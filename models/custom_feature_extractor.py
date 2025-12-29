import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.multimodal_encoder import MultimodalEncoder
from models.hierarchical_fusion_transformer import HierarchicalFusionTransformer
from models.constraint_decoder import ConstraintDecoder

class CustomMultimodalFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multimodal observations (vision, text, proprio)
    that integrates the MultimodalEncoder, HierarchicalFusionTransformer,
    and ConstraintDecoder.
    """
    def __init__(self, observation_space: spaces.Dict, embed_dim: int, proprio_dim: int, tiny_transformer_layers: int, action_dim: int):
        super().__init__(observation_space, features_dim=embed_dim)

        # Initialize models
        self.multimodal_encoder = MultimodalEncoder(proprio_dim=proprio_dim, embed_dim=embed_dim, freeze_backbones=True)
        self.fusion_transformer = HierarchicalFusionTransformer(embed_dim=embed_dim, num_layers=tiny_transformer_layers)
        self.constraint_decoder = ConstraintDecoder(embed_dim=embed_dim, proprio_dim=proprio_dim)
        
        # In the SB3, the feature extractor usually outputs features before the action/value heads.
        # I.e., the actual action_head will be handled by SB3's policy network.
        # self.action_head = nn.Linear(embed_dim, action_dim) # This will be part of the SB3 policy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        # Extract and process each modality
        vision_tensor = observations["vision"].float()
        text_tensor = observations["text"].long()
        text_mask_tensor = observations["text_attention_mask"].long()
        proprio_tensor = observations["proprio"].float()

        # Multimodal Encoding
        multimodal_embedding = self.multimodal_encoder(
            vision=vision_tensor, text=text_tensor, text_attention_mask=text_mask_tensor, proprio=proprio_tensor
        ) # Shape: (B, embed_dim)

        # Hierarchical Fusion Transformer
        # Transformer expects (Sequence, Batch, Embedding). Here, Sequence=1
        fused_features, c_high = self.fusion_transformer(multimodal_embedding.unsqueeze(1))
        # fused_features shape: (B, 1, embed_dim)
        # c_high shape: (B, embed_dim)

        # Constraint Decoder
        c_low = self.constraint_decoder(c_high, proprio_tensor) # Shape: (B, embed_dim)

        # Combine features for the final policy/value heads
        # fused_features.squeeze(1) converts (B, 1, embed_dim) to (B, embed_dim)
        combined_features = fused_features.squeeze(1) + c_low
        
        return combined_features