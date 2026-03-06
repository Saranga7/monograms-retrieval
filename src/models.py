from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import logging
from transformers import AutoModel
from omegaconf import OmegaConf



logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, cfg):
        super(ProjectionHead, self).__init__()

        if cfg.model.proj_head_complexity == 0:
            self.net = nn.Linear(in_dim, cfg.model.embed_dim)

        elif cfg.model.proj_head_complexity == 1:
            self.net = nn.Sequential(
                    nn.Linear(in_dim, 512),
                    nn.LayerNorm(512),
                    nn.PReLU(),
                    nn.Dropout(0.07),
                    nn.Linear(512, cfg.model.embed_dim)
                )
        elif cfg.model.proj_head_complexity == 2:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, cfg.model.embed_dim)
            )
        elif cfg.model.proj_head_complexity == 3:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, cfg.model.embed_dim)
            )
        else:
            raise ValueError(f"Invalid projection head complexity: {cfg.model.proj_head_complexity}")

    def forward(self, x):
        x = self.net(x)
        return x


class DualEncoder(nn.Module):
    def __init__(self, cfg):
        super(DualEncoder, self).__init__()

        self.cfg = cfg

        if cfg.model.name.startswith("dinov3"):
            if cfg.model.freeze_backbone and not cfg.model.unfreeze_last_layer: # if backbone is completely frozen, share weights to save memory
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = AutoModel.from_pretrained(cfg.model.model_version)
                self.schema_encoder = self.seal_encoder

            else:
                self.schema_encoder = AutoModel.from_pretrained(cfg.model.model_version)
                self.seal_encoder = AutoModel.from_pretrained(cfg.model.model_version)

        elif cfg.model.name == "resnet18":
            if cfg.model.freeze_backbone and not cfg.model.unfreeze_last_layer:
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = models.resnet18(weights="IMAGENET1K_V1")
                self.schema_encoder = self.seal_encoder
            else:
                self.schema_encoder = models.resnet18(weights="IMAGENET1K_V1")
                self.seal_encoder = models.resnet18(weights="IMAGENET1K_V1")

        elif cfg.model.name == "resnet50":
            if cfg.model.freeze_backbone and not cfg.model.unfreeze_last_layer:
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = models.resnet50(weights="IMAGENET1K_V1")
                self.schema_encoder = self.seal_encoder
            else:
                self.schema_encoder = models.resnet50(weights="IMAGENET1K_V1")
                self.seal_encoder = models.resnet50(weights="IMAGENET1K_V1")

        elif cfg.model.name == "efficientnet_b0":
            if cfg.model.freeze_backbone and not cfg.model.unfreeze_last_layer:
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = models.efficientnet_b0(weights="IMAGENET1K_V1")
                self.schema_encoder = self.seal_encoder
            else:
                self.schema_encoder = models.efficientnet_b0(weights="IMAGENET1K_V1")
                self.seal_encoder = models.efficientnet_b0(weights="IMAGENET1K_V1")
        
        else:
            raise ValueError(f"Unsupported model name: {cfg.model.name}")
        
        self.feature_dim, self.schema_encoder, self.seal_encoder = self._strip_backbone()

        if cfg.model.freeze_backbone:
            for param in self.schema_encoder.parameters():
                param.requires_grad = False
            for param in self.seal_encoder.parameters():
                param.requires_grad = False
        
        if cfg.model.unfreeze_last_layer:
            if cfg.model.name.startswith("dinov3"):
                for param in self.schema_encoder.layer[-1].parameters():
                    param.requires_grad = True
                for param in self.seal_encoder.layer[-1].parameters():
                    param.requires_grad = True


        # Projection heads to map to a common embedding space
        self.schema_proj = ProjectionHead(self.feature_dim, cfg)
        self.seal_proj = ProjectionHead(self.feature_dim, cfg)


    def forward(self, schema, seal):
        # Extract features

        if self.cfg.model.name.startswith("dinov3"):
            z_schema = self.schema_encoder(schema).last_hidden_state[:, 0, :]  # CLS token
            z_seal = self.seal_encoder(seal).last_hidden_state[:, 0, :]  # CLS token
        else:
            z_schema = self.schema_encoder(schema)
            z_seal = self.seal_encoder(seal)

        # Project to common embedding space
        z_schema = self.schema_proj(z_schema)
        z_seal = self.seal_proj(z_seal)


        z_schema = F.normalize(z_schema, dim=1)
        z_seal = F.normalize(z_seal, dim=1)

        return z_schema, z_seal
    
    
    def _strip_backbone(self):

        # for dinov3
        if self.cfg.model.name.startswith("dinov3"):
            feat_dim = self.schema_encoder.config.hidden_size
            # self.schema_encoder.head = nn.Identity()
            # self.seal_encoder.head = nn.Identity()

            return feat_dim, self.schema_encoder, self.seal_encoder

        # For ResNet-like
        if hasattr(self.schema_encoder, "fc"):
            feat_dim = self.schema_encoder.fc.in_features
            self.schema_encoder.fc = nn.Identity()
            self.seal_encoder.fc = nn.Identity()

            return feat_dim, self.schema_encoder, self.seal_encoder
        
        # EfficientNet / MobileNet / ConvNeXt
        if hasattr(self.schema_encoder, "classifier"):
            last_layer = self.schema_encoder.classifier[-1]
            feat_dim = last_layer.in_features
            self.schema_encoder.classifier = nn.Identity()
            self.seal_encoder.classifier = nn.Identity()

            return feat_dim, self.schema_encoder, self.seal_encoder
            
        # ViT / Swin
        if hasattr(self.schema_encoder, "head"):
            last_layer = self.schema_encoder.head[-1]
            feat_dim = last_layer.in_features
            self.schema_encoder.head = nn.Identity()
            self.seal_encoder.head = nn.Identity()

            return feat_dim, self.schema_encoder, self.seal_encoder
        
        # Fallback: run dummy input
        logger.info("Falling back to dummy forward pass to find output dim.")

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.schema_encoder(dummy)
            if out.ndim > 2:  # Flatten if needed
                out = torch.flatten(out, 1)
            feat_dim = out.shape[1]

        return feat_dim, self.schema_encoder, self.seal_encoder
    

if __name__ == "__main__":
    # Example usage
    cfg = OmegaConf.create({
    "model": {
        "name": "dinov3",
        "model_version": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "embed_dim": 256,
        "freeze_backbone": True,
        "unfreeze_last_layer": False,
        "proj_head_complexity": 1,
    }
})

    model = DualEncoder(cfg)
    dummy_schema = torch.randn(2, 3, 224, 224)
    dummy_seal = torch.randn(2, 3, 224, 224)
    z_schema, z_seal = model(dummy_schema, dummy_seal)
    print(f"Schema embedding shape: {z_schema.shape}, Seal embedding shape: {z_seal.shape}")

        