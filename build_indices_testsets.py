from html import parser
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra import initialize, compose

from src.dataset import MonogramPairDataset
from src.models import DualEncoder



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
    # parser.add_argument("--fold", type=int, default=0,
    #                     help="Fold to use if split is train/test.")
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
    config_path = args.config_path
    config_name = args.config_name

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers

    return cfg


def build_index(cfg, fold, checkpoint_path, split="all", device="cpu"):
    device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        split_regime=cfg.data.split_regime,
        fold=fold,
        split=split,
        transform = cfg.data.transforms.enable,
        return_paths=True,
        use_processed_seals = cfg.train.use_processed_seals,
        kwargs={
            "use_strong_augmentation": cfg.data.transforms.use_strong_augmentation,
            "use_grayscale": cfg.data.transforms.use_grayscale
        }
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
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
        "fold": fold,
        "checkpoint_path": checkpoint_path,
        "model_name": cfg.model.name,
        "model_version": cfg.model.model_version,
        "embed_dim": cfg.model.embed_dim,
    }


def main():
    args = parse_args()
    cfg = load_cfg(args)

    output_dir = os.path.dirname(args.output)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    all_schema_embeddings = []
    all_seal_embeddings = []
    all_schema_paths = []
    all_seal_paths = []
    all_pair_ids = []
    all_source_folds = []

    for fold in range(5):
        print(f"\n{'=' * 60}")
        print(f"Processing fold {fold}")
        print(f"{'=' * 60}")

        checkpoint_path = os.path.join(args.checkpoint, 
                                       "checkpoints", 
                                       f"fold_{fold}", 
                                       "best_model.pth") 
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        

        index_data = build_index(
            cfg=cfg,
            fold = fold,
            checkpoint_path=checkpoint_path,
            split=args.split,
            device=args.device,
        )

        n = len(index_data["pair_ids"])
        print(f"Fold {fold} - Num pairs: {n}")

        all_schema_embeddings.append(index_data["schema_embeddings"])
        all_seal_embeddings.append(index_data["seal_embeddings"])
        all_schema_paths.extend(index_data["schema_paths"])
        all_seal_paths.extend(index_data["seal_paths"])
        all_pair_ids.extend(index_data["pair_ids"])
        all_source_folds.extend([fold] * n)

    
    combined_index_data = {
        "schema_embeddings": torch.cat(all_schema_embeddings, dim=0),   # [N, D]
        "seal_embeddings": torch.cat(all_seal_embeddings, dim=0),       # [N, D]
        "schema_paths": all_schema_paths,
        "seal_paths": all_seal_paths,
        "pair_ids": all_pair_ids,
        "source_folds": all_source_folds,
        "split": args.split,
        "num_folds_combined": 5,
        "checkpoint_dir": args.checkpoint,
        "model_name": cfg.model.name,
        "model_version": cfg.model.model_version,
        "embed_dim": cfg.model.embed_dim,
    }


    torch.save(combined_index_data, args.output)

    print(f"Saved index to: {args.output}")
    print(f"Num pairs: {len(combined_index_data['pair_ids'])}")
    print(f"Schema embeddings: {combined_index_data['schema_embeddings'].shape}")
    print(f"Seal embeddings: {combined_index_data['seal_embeddings'].shape}")


if __name__ == "__main__":
    main()