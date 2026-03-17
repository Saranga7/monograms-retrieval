import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CLIPLoss(nn.Module):
    def __init__(
        self,
        init_temperature: float = 0.07,
        max_scale: float = 100.0,
        return_metrics: bool = False,
    ):
        super().__init__()

        self.return_metrics = return_metrics
        self.max_logit_scale = math.log(max_scale)

        # Learnable logit scale, same idea as CLIP
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / init_temperature)
        )

    def forward(self, z_schema: torch.Tensor, z_seal: torch.Tensor):
        """
        z_schema: (B, D)
        z_seal:   (B, D)

        Assumes aligned batch: i-th schema matches i-th seal.
        """

        # Clamp for stability
        logit_scale = self.logit_scale.exp().clamp(max=math.exp(self.max_logit_scale))

        # Similarity matrix
        logits = torch.matmul(z_schema, z_seal.T) * logit_scale

        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric contrastive loss
        loss_sc2se = F.cross_entropy(logits, labels)     # schema -> seal
        loss_se2sc = F.cross_entropy(logits.T, labels)   # seal -> schema
        loss = (loss_sc2se + loss_se2sc) / 2

        if not self.return_metrics:
            return loss, loss # dummy second return for compatibility 

        with torch.no_grad():
            preds_sc2se = logits.argmax(dim=1)
            preds_se2sc = logits.T.argmax(dim=1)

            acc_sc2se = (preds_sc2se == labels).float().mean()
            acc_se2sc = (preds_se2sc == labels).float().mean()
            acc = (acc_sc2se + acc_se2sc) / 2

        return loss, {
            "acc_sc2se": acc_sc2se.item(),
            "acc_se2sc": acc_se2sc.item(),
            "acc_mean": acc.item(),
        }




class ArcFaceCLIPLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        init_temperature: float = 0.07,
        max_scale: float = 100.0,
        return_metrics: bool = False,
    ):
        super().__init__()

        self.margin = margin
        self.return_metrics = return_metrics
        self.max_logit_scale = math.log(max_scale)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / init_temperature)
        )

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, z_schema: torch.Tensor, z_seal: torch.Tensor):
        B = z_schema.size(0)

        cosine = torch.matmul(z_schema, z_seal.T).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * self.cos_m - sine * self.sin_m

        labels = torch.arange(B, device=cosine.device)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine

        logit_scale = self.logit_scale.exp().clamp(max=math.exp(self.max_logit_scale))
        logits = logits * logit_scale

        loss_sc2se = F.cross_entropy(logits, labels)
        loss_se2sc = F.cross_entropy(logits.T, labels)
        loss = (loss_sc2se + loss_se2sc) / 2

        if not self.return_metrics:
            return loss, loss # dummy second return for compatibility with CLIPLoss

        with torch.no_grad():
            preds_sc2se = logits.argmax(dim=1)
            preds_se2sc = logits.T.argmax(dim=1)

            acc_sc2se = (preds_sc2se == labels).float().mean()
            acc_se2sc = (preds_se2sc == labels).float().mean()
            acc = (acc_sc2se + acc_se2sc) / 2

        return loss, {
            "acc_sc2se": acc_sc2se.item(),
            "acc_se2sc": acc_se2sc.item(),
            "acc_mean": acc.item(),
        }


class EmbeddingConsistencyLoss(nn.Module):
    """
    Encourages two embeddings of the same seal
    (raw seal and processed seal) to be close.

    Assumes inputs are already L2-normalized
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, z_raw: torch.Tensor, z_proc: torch.Tensor):

        # cosine similarity per sample
        cos_sim = (z_raw * z_proc).sum(dim = 1)
        loss = 1.0 - cos_sim

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss