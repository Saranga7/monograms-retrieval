from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import logging
from transformers import AutoModel
from omegaconf import OmegaConf

from src.utils import ResidualGatedAttention


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
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, cfg.model.embed_dim)
            )
        elif cfg.model.proj_head_complexity == 3:
            self.net = nn.Sequential(
                ResidualGatedAttention(in_dim, num_heads = 4),
                nn.Dropout(0.3),
                nn.Linear(in_dim, cfg.model.embed_dim),
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
            if cfg.model.share_backbone and not cfg.model.freeze_backbone: # if backbone is completely frozen, share weights to save memory
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = AutoModel.from_pretrained(cfg.model.model_version)
                self.schema_encoder = self.seal_encoder

            else:
                self.schema_encoder = AutoModel.from_pretrained(cfg.model.model_version)
                self.seal_encoder = AutoModel.from_pretrained(cfg.model.model_version)

        elif cfg.model.name == "resnet18":
            if cfg.model.share_backbone and not cfg.model.freeze_backbone: # if backbone is completely frozen, share weights to save memory
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = models.resnet18(weights="IMAGENET1K_V1")
                self.schema_encoder = self.seal_encoder
            else:
                self.schema_encoder = models.resnet18(weights="IMAGENET1K_V1")
                self.seal_encoder = models.resnet18(weights="IMAGENET1K_V1")

        elif cfg.model.name == "resnet50":
            if cfg.model.share_backbone and not cfg.model.freeze_backbone: # if backbone is completely frozen, share weights to save memory
                logger.info("Frozen backbone, hence sharing weights for schema and seal encoders.")
                self.seal_encoder = models.resnet50(weights="IMAGENET1K_V1")
                self.schema_encoder = self.seal_encoder
            else:
                self.schema_encoder = models.resnet50(weights="IMAGENET1K_V1")
                self.seal_encoder = models.resnet50(weights="IMAGENET1K_V1")

        elif cfg.model.name == "efficientnet_b0":
            if cfg.model.share_backbone and not cfg.model.freeze_backbone: # if backbone is completely frozen, share weights to save memory
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
                schema_last = self.schema_encoder.layer[-1]
                seal_last = self.seal_encoder.layer[-1]
            elif cfg.model.name in ["resnet18", "resnet50"]:
                schema_last = self.schema_encoder.layer4
                seal_last = self.seal_encoder.layer4
            elif cfg.model.name == "efficientnet_b0":
                schema_last = self.schema_encoder.features[-1]
                seal_last = self.seal_encoder.features[-1]
            else:
                raise ValueError(f"unfreeze_last_layer not implemented for {cfg.model.name}")

            for param in schema_last.parameters():
                param.requires_grad = True
            for param in seal_last.parameters():
                param.requires_grad = True


        # Projection heads to map to a common embedding space
        self.schema_proj = ProjectionHead(self.feature_dim, cfg)
        self.seal_proj = ProjectionHead(self.feature_dim, cfg)


    def forward(self, schema, seal):
        # Extract features
        z_schema = self.encode_schema(schema)
        z_seal = self.encode_seal(seal)
        return z_schema, z_seal

    def encode_schema(self, schema):
        z = self._encode_backbone(schema, self.schema_encoder)
        z = self.schema_proj(z) # project to common embedding space
        z = F.normalize(z, dim=1) # L2 normalize for cosine similarity
        return z
    
    def encode_seal(self, seal):
        z = self._encode_backbone(seal, self.seal_encoder)
        z = self.seal_proj(z) # project to common embedding space
        z = F.normalize(z, dim=1) # L2 normalize for cosine similarity
        return z


    def extract_tokens(self, x, encoder):
        assert self.cfg.model.name.startswith("dinov3"), "extract_tokens is only implemented for dinov3"
        out = encoder(x)
        h = out.last_hidden_state            # [B, 1+T, D]
        cls_token = h[:, 0, :]
        cls_token = F.normalize(cls_token, dim=-1)  # L2 normalize
        patch_tokens = h[:, 1:, :]
        patch_tokens = F.normalize(patch_tokens, dim=-1)  # L2 normalize
        return {"CLS_token": cls_token, "patch_tokens": patch_tokens}
    
    
    def _encode_backbone(self, x, encoder):
        if self.cfg.model.name.startswith("dinov3"):
            h = encoder(x).last_hidden_state
            cls = h[:, 0, :]
            patch_mean = h[:, 1:, :].mean(dim=1)

            pool_type = self.cfg.model.get("token_pool", "cls")

            if pool_type == "cls":
                return cls
            elif pool_type == "patch_mean":
                return patch_mean
            elif pool_type == "all_mean":
                return h.mean(dim=1)
            elif pool_type == "cls_patch_mean":
                return torch.cat([cls, patch_mean], dim=-1)
            else:
                raise ValueError(f"Unknown token_pool: {pool_type}")
        else:
            return encoder(x)
    
    
    def _strip_backbone(self):

        # for dinov3
        if self.cfg.model.name.startswith("dinov3"):
            feat_dim = self.schema_encoder.config.hidden_size
            if self.cfg.model.token_pool == "cls_patch_mean":
                feat_dim *= 2

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

        