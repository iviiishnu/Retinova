# src/models/fusion_model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class FusionEffNetTabular(nn.Module):
    def __init__(
        self,
        img_feature_dim=512,
        tabular_input_dim=2,   # [age, gender]
        tabular_hidden_dim=64,
        fusion_hidden_dim=128,
        dropout=0.3,
        pretrained=True,
    ):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.img_proj = nn.Sequential(
            nn.Linear(in_features, img_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.tab_proj = nn.Linear(tabular_hidden_dim, 64)

        fusion_input_dim = img_feature_dim + 64
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),  # CVD logit
        )

    def forward(self, x_img, x_tab):
        img_features = self.backbone(x_img)          # [B, 1536]
        img_features = self.img_proj(img_features)   # [B, img_feature_dim]

        tab_hidden = self.tab_mlp(x_tab)
        tab_features = self.tab_proj(tab_hidden)     # [B, 64]

        fused = torch.cat([img_features, tab_features], dim=1)
        logit = self.fusion_mlp(fused).squeeze(1)    # [B]
        return logit
