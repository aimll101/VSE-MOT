import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(embed_dim))
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, cross_features, motrv2_queries):
        w = torch.sigmoid(self.weight_logits).unsqueeze(0).unsqueeze(1)
        fused_features = (1 - w) * cross_features + w * motrv2_queries
        return self.transform(fused_features)
