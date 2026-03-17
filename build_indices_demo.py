from html import parser
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra import initialize, compose

from src.dataset import MonogramPairDataset
from src.models import DualEncoder
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute embeddings for notebook demo.")
    parser.add_argument("--config_path", type=str, default="configs",
                    help="Path to Hydra config directory.")
    parser.add_argument("--config_name", type=str, default="training",
                    help="Hydra config name without .yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--output", type=str, required=True, default="configs/training.yaml",
                        help="Path to save the precomputed embedding index (.pt).")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"],
                        help="Dataset split to encode.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold to use if split is train/test.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cpu or cuda.")
    return parser.parse_args()


def load_cfg(args):
    """
    Compose Hydra config from configs/training.yaml defaults.
    Optionally allows overrides from CLI later.
    """
    config_path = "configs"
    config_name = "training"

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    cfg.data.fold = args.fold
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers

    return cfg


def build_index(cfg, checkpoint_path, split="all", device="cpu"):
    device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir if split != "all" else None,
        fold=cfg.data.fold if split != "all" else None,
        split=split,
        return_paths=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DualEncoder(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    schema_embeddings = []
    seal_embeddings = []
    schema_paths_all = []
    seal_paths_all = []
    pair_ids = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Encoding")
        for batch in pbar:
            schema = batch["schema"].to(device)
            seal = batch["seal"].to(device)
            schema_paths = batch["schema_path"]
            seal_paths = batch["seal_path"]

            z_schema, z_seal = model(schema, seal)

            schema_embeddings.append(z_schema.cpu())
            seal_embeddings.append(z_seal.cpu())

            schema_paths_all.extend(list(schema_paths))
            seal_paths_all.extend(list(seal_paths))

            pair_ids.extend([
                os.path.splitext(os.path.basename(p))[0]
                for p in schema_paths
            ])

    schema_embeddings = torch.cat(schema_embeddings, dim=0)
    seal_embeddings = torch.cat(seal_embeddings, dim=0)

    return {
        "schema_embeddings": schema_embeddings,   # [N, D]
        "seal_embeddings": seal_embeddings,       # [N, D]
        "schema_paths": schema_paths_all,
        "seal_paths": seal_paths_all,
        "pair_ids": pair_ids,
        "split": split,
        "fold": cfg.data.fold,
        "checkpoint_path": checkpoint_path,
        "model_name": cfg.model.name,
        "model_version": cfg.model.model_version,
        "embed_dim": cfg.model.embed_dim,
    }


def main():
    args = parse_args()
    cfg = load_cfg(args)
    
    index_data = build_index(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index_data, output_path)

    print(f"Saved index to: {output_path}")
    print(f"Num pairs: {len(index_data['pair_ids'])}")
    print(f"Schema embeddings: {index_data['schema_embeddings'].shape}")
    print(f"Seal embeddings: {index_data['seal_embeddings'].shape}")


if __name__ == "__main__":
    main()