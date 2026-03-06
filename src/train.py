import math
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from tqdm import tqdm
import logging
import datetime

from src.dataset import MonogramPairDataset
from src.models import DualEncoder
from src.losses import CLIPLoss, ArcFaceCLIPLoss
from src.evaluation import evaluate_retrieval_accuracy
from src.visualization import visualize


logger = logging.getLogger(__name__)



def train(cfg):
    
    start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.wandb.use_wandb_logging:
        wandb.init(
            project = cfg.wandb.project_name, 
            name = cfg.wandb.name,
            config = dict(cfg),
            )
    
    logger.info(f"Using Device: {device}")
    logger.info(f"Fold: {cfg.data.fold}")
    

    # Dataset
    train_dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        fold = cfg.data.fold,
        split = "train",
    )

    test_dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        fold = cfg.data.fold,
        split = "test",
    )

    if cfg.data.batch_size == "full":
        batch_size = len(train_dataset)
        logger.info(f"Using full batch size: {batch_size}")
    else:
        batch_size = cfg.data.batch_size

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = cfg.data.shuffle, 
        num_workers = cfg.data.num_workers,
        pin_memory = True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = cfg.data.num_workers,
        pin_memory = True
        )


    # Model
    model = DualEncoder(cfg).to(device)

    # Loss function
    if cfg.loss.name.lower().startswith("arcface"):
        logger.info("Using ArcFaceCLIPLoss")
        criterion = ArcFaceCLIPLoss(
            margin = cfg.loss.margin,
            init_temperature = cfg.loss.temperature,
            max_scale = cfg.loss.max_scale,
            return_metrics = False
        ).to(device)
    else:
        logger.info("Using CLIP Loss")
        criterion = CLIPLoss(
            init_temperature = cfg.loss.temperature,
            max_scale = cfg.loss.max_scale,
            return_metrics = False
        ).to(device)
    best_test_loss = float('inf')


    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    pbar = tqdm(range(cfg.train.epochs), desc = f"Training Fold {cfg.data.fold}")
    for epoch in pbar:
        # -------------------------
        # TRAINING
        # -------------------------
        model.train()
        train_loss = 0

        for schema, seal in train_loader:
            schema, seal = schema.to(device), seal.to(device)
            z_schema, z_seal = model(schema, seal)

            loss, _ = criterion(z_schema, z_seal)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for schema, seal in test_loader:
                schema, seal = schema.to(device), seal.to(device)
                z_schema, z_seal = model(schema, seal)
                loss, _ = criterion(z_schema, z_seal)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        # -------------------------
        # LOGGING
        # -------------------------
        if cfg.wandb.use_wandb_logging:
            wandb.log({
                "train_loss": train_loss, 
                "test_loss": test_loss, 
                "epoch": epoch,
                "logit_scale": criterion.logit_scale.exp().item()
            })

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

        # -------------------------
        # CHECKPOINTING
        # -------------------------
        if epoch % cfg.train.log_interval == 0:
            checkpoint_path = cfg.train.checkpoint_dir
            os.makedirs(checkpoint_path, exist_ok=True)

            save_dir = os.path.join(
                checkpoint_path,
                f"fold_{cfg.data.fold}"
            )

            os.makedirs(save_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                save_dir,
                f"epoch_{epoch}.pth"
            )
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            # torch.save({
            #         "model_state_dict": model.state_dict(),
            #         "criterion_state_dict": criterion.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "epoch": epoch,
            #     }, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")
    

        # save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss

            checkpoint_path = cfg.train.checkpoint_dir
            os.makedirs(checkpoint_path, exist_ok=True)
        
            save_dir = os.path.join(
                checkpoint_path,
                f"fold_{cfg.data.fold}"
            )
            os.makedirs(save_dir, exist_ok=True)

            best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"Best model saved at epoch {epoch} with test_loss={best_test_loss:.4f}")

    logger.info("Training complete.\n\n-----------------------------\n")
    # Testing retrieval performance of the best model
    logger.info("Loading best model for evaluation...")
    del model
    torch.cuda.empty_cache()
    
    model = DualEncoder(cfg).to(device)
    state_dict = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    logger.info("Evaluating best model on train set...")
    metrics_train = evaluate_retrieval_accuracy(model, train_loader, device)

    logger.info("Seal -> Schema")
    for k, v in metrics_train["seal2schema"].items():
        logger.info(f"{k}: {v:.4f}")
    
    logger.info("Schema -> Seal")
    for k, v in metrics_train["schema2seal"].items():
        logger.info(f"{k}: {v:.4f}")

    if cfg.wandb.use_wandb_logging:
        wandb.log({
            "train_R@1_se2sc": metrics_train["seal2schema"]["R@1"],
            "train_R@5_se2sc": metrics_train["seal2schema"]["R@5"],
            "train_R@10_se2sc": metrics_train["seal2schema"]["R@10"],
            "train_MRR_se2sc": metrics_train["seal2schema"]["MRR"],
            "train_MedianRank_se2sc": metrics_train["seal2schema"]["MedianRank"],
        })

        wandb.log({
            "train_R@1_sc2se": metrics_train["schema2seal"]["R@1"],
            "train_R@5_sc2se": metrics_train["schema2seal"]["R@5"],
            "train_R@10_sc2se": metrics_train["schema2seal"]["R@10"],
            "train_MRR_sc2se": metrics_train["schema2seal"]["MRR"],
            "train_MedianRank_sc2se": metrics_train["schema2seal"]["MedianRank"],
        })

    

    logger.info("-----------------------------")

    logger.info("Evaluating best model on test set...")
    metrics_test = evaluate_retrieval_accuracy(model, test_loader, device)  
    logger.info("Seal -> Schema")
    for k, v in metrics_test["seal2schema"].items():
        logger.info(f"{k}: {v:.4f}")

    logger.info("Schema -> Seal")
    for k, v in metrics_test["schema2seal"].items():
        logger.info(f"{k}: {v:.4f}") 

    if cfg.wandb.use_wandb_logging:
        wandb.log({
            "test_R@1_se2sc": metrics_test["seal2schema"]["R@1"],
            "test_R@5_se2sc": metrics_test["seal2schema"]["R@5"],
            "test_R@10_se2sc": metrics_test["seal2schema"]["R@10"],
            "test_MRR_se2sc": metrics_test["seal2schema"]["MRR"],
            "test_MedianRank_se2sc": metrics_test["seal2schema"]["MedianRank"],
        })

        wandb.log({
            "test_R@1_sc2se": metrics_test["schema2seal"]["R@1"],
            "test_R@5_sc2se": metrics_test["schema2seal"]["R@5"],
            "test_R@10_sc2se": metrics_test["schema2seal"]["R@10"],
            "test_MRR_sc2se": metrics_test["schema2seal"]["MRR"],
            "test_MedianRank_sc2se": metrics_test["schema2seal"]["MedianRank"],
        })
            
        # wandb.finish()

    # Visualization
    if cfg.visualize_after_train:
        logger.info("Generating embedding visualizations...")
        vis_dir = os.path.join(
            cfg.train.checkpoint_dir,
            f"fold_{cfg.data.fold}",
            "visualizations"
        )

        visualize(
            cfg,
            checkpoint_path=best_checkpoint_path,
            split="test",
            output_dir=vis_dir
        )

        # Optional W&B logging
        try:
            wandb.log({
                "tsne_plot": wandb.Image(os.path.join(vis_dir, "tsne_plot.png")),
                "umap_plot": wandb.Image(os.path.join(vis_dir, "umap_plot.png")),
            })
        except Exception as e:
            logger.warning(f"Failed to log visualization images to W&B: {e}")

    if cfg.wandb.use_wandb_logging:
        wandb.finish()
    
    # Finishing
    end = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    total_time = datetime.datetime.strptime(end, "%Y-%m-%d_%H-%M-%S") - datetime.datetime.strptime(start, "%Y-%m-%d_%H-%M-%S")
    total_seconds = total_time.total_seconds()
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60    
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

