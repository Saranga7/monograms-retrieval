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
from src.losses import CLIPLoss, ArcFaceCLIPLoss, EmbeddingConsistencyLoss
from src.evaluation import evaluate_retrieval_accuracy
from src.visualization import visualize


logger = logging.getLogger(__name__)



def train(cfg):
    
    start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.wandb.use_wandb_logging:
        wandb.init(
            project = cfg.wandb.project_name, 
            name = f"{cfg.model.name}_fulldataset",
            config = dict(cfg),
            )
    
    logger.info(f"Using Device: {device}")
    logger.info(f"Fold: {cfg.data.fold}")
    

    # Dataset
    train_dataset = MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        fold = None,  # Use all folds for training
        split = "all",
        transform = cfg.data.transforms.enable,
        return_paths = False,
        use_processed_seals = cfg.train.use_processed_seals,
        kwargs={
            "use_strong_augmentation": cfg.data.transforms.use_strong_augmentation,
            "use_grayscale": cfg.data.transforms.use_grayscale
        }
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

        consistency_criterion = EmbeddingConsistencyLoss().to(device)
        lambda_aux = cfg.loss.lambda_aux
        # lambda_proc = cfg.loss.lambda_proc

    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    best_train_loss = float('inf')
    pbar = tqdm(range(cfg.train.epochs), desc = f"Training Fold {cfg.data.fold}")
    for epoch in pbar:
        # -------------------------
        # TRAINING
        # -------------------------
        model.train()
        train_loss = 0
        train_loss_aux = 0
        train_loss_clip = 0
        # train_loss_clip_proc = 0

        for batch in train_loader:
            if cfg.train.use_processed_seals:
                schema, seal, seal_proc = batch["schema"].to(device), batch["seal"].to(device), batch["seal_proc"].to(device)
            else:
                schema, seal = batch["schema"].to(device), batch["seal"].to(device)

            z_schema, z_seal = model(schema, seal)

            clip_loss, _ = criterion(z_schema, z_seal)
            train_loss_clip += clip_loss.item()

            if cfg.train.use_processed_seals:
                z_seal_proc = model.encode_seal(seal_proc)
                loss_aux = consistency_criterion(z_seal, z_seal_proc)
                train_loss_aux += loss_aux.item()

                loss = clip_loss + lambda_aux * loss_aux

                # clip_loss_proc, _ = criterion(z_schema, z_seal_proc) 
                # train_loss_clip_proc += clip_loss_proc.item()

                # loss = clip_loss + lambda_aux * loss_aux + lambda_proc * clip_loss_proc
            else:
                loss = clip_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        if cfg.train.use_processed_seals:
            train_loss_aux /= len(train_loader)
            train_loss_clip /= len(train_loader)
            # train_loss_clip_proc /= len(train_loader)

        if cfg.train.use_processed_seals:

            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} \
                        (clip={train_loss_clip:.4f}, aux={train_loss_aux:.4f})")
       
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        # -------------------------
        # LOGGING
        # -------------------------
        if cfg.wandb.use_wandb_logging:
            wandb.log({
                "epoch": epoch,
                "logit_scale": criterion.logit_scale.exp().item()
            })

            if cfg.train.use_processed_seals:
                wandb.log({
                    "train_loss_clip": train_loss_clip,
                    "train_loss_aux": train_loss_aux,
                    # "train_loss_clip_proc": train_loss_clip_proc,
                    "total_train_loss": train_loss,
                    
                })

                
            else:
                wandb.log({
                    "train_loss": train_loss, 
                })


        

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
        if train_loss < best_train_loss:
            best_train_loss = train_loss

            checkpoint_path = cfg.train.checkpoint_dir
            os.makedirs(checkpoint_path, exist_ok=True)
        
            save_dir = os.path.join(
                checkpoint_path,
                f"fold_{cfg.data.fold}"
            )
            os.makedirs(save_dir, exist_ok=True)

            best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"Best model saved at epoch {epoch} with train_loss={best_train_loss:.4f}")

    logger.info("Training complete.\n\n-----------------------------\n")
    # Testing retrieval performance of the best model
    logger.info("Loading best model for evaluation...")
    del model
    torch.cuda.empty_cache()
    
    model = DualEncoder(cfg).to(device)
    state_dict = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


    logger.info("Evaluating best model on full dataset...")
    metrics_test = evaluate_retrieval_accuracy(model, 
                                               train_loader, 
                                               device,
                                               top_k = None,
                                               rerank_mode = None) 

    logger.info("Seal -> Schema")
    for k, v in metrics_test["seal2schema"].items():
        logger.info(f"Baseline {k}: {v:.4f}")

    logger.info("Schema -> Seal")
    for k, v in metrics_test["schema2seal"].items():
        logger.info(f"Baseline {k}: {v:.4f}") 

    if cfg.wandb.use_wandb_logging:
        wandb.log({
            "baseline/R@1_se2sc": metrics_test["seal2schema"]["R@1"],
            "baseline/R@5_se2sc": metrics_test["seal2schema"]["R@5"],
            "baseline/R@10_se2sc": metrics_test["seal2schema"]["R@10"],
            "baseline/MRR_se2sc": metrics_test["seal2schema"]["MRR"],
            "baseline/MedianRank_se2sc": metrics_test["seal2schema"]["MedianRank"],
        })

        wandb.log({
            "baseline/R@1_sc2se": metrics_test["schema2seal"]["R@1"],
            "baseline/R@5_sc2se": metrics_test["schema2seal"]["R@5"],
            "baseline/R@10_sc2se": metrics_test["schema2seal"]["R@10"],
            "baseline/MRR_sc2se": metrics_test["schema2seal"]["MRR"],
            "baseline/MedianRank_sc2se": metrics_test["schema2seal"]["MedianRank"],
        })

    if cfg.test.reranking.enable:
        for topk in [10, 20, 50]:
            logger.info(f"\nApplying patch token-level reranking with top_k={topk}...")
            reranked_metrics = evaluate_retrieval_accuracy(model, 
                                                        test_loader, 
                                                        device, 
                                                        top_k = topk,
                                                        rerank_mode = cfg.test.reranking.mode,
                                                        alpha = cfg.test.reranking.alpha,
                                                        normalize_mode = cfg.test.reranking.normalize_mode
                                                        )

            logger.info("Seal -> Schema")
            for k, v in reranked_metrics["seal2schema"].items():
                logger.info(f"Reranked_top{topk} {k}: {v:.4f}")

            logger.info("Schema -> Seal")
            for k, v in reranked_metrics["schema2seal"].items():
                logger.info(f"Reranked_top{topk} {k}: {v:.4f}")

            if cfg.wandb.use_wandb_logging:
                wandb.log({
                    f"reranked_top{topk}/R@1_se2sc": reranked_metrics["seal2schema"]["R@1"],
                    f"reranked_top{topk}/R@5_se2sc": reranked_metrics["seal2schema"]["R@5"],
                    f"reranked_top{topk}/R@10_se2sc": reranked_metrics["seal2schema"]["R@10"],
                    f"reranked_top{topk}/MRR_se2sc": reranked_metrics["seal2schema"]["MRR"],
                    f"reranked_top{topk}/MedianRank_se2sc": reranked_metrics["seal2schema"]["MedianRank"],
                })

                wandb.log({
                    f"reranked_top{topk}/R@1_sc2se": reranked_metrics["schema2seal"]["R@1"],
                    f"reranked_top{topk}/R@5_sc2se": reranked_metrics["schema2seal"]["R@5"],
                    f"reranked_top{topk}/R@10_sc2se": reranked_metrics["schema2seal"]["R@10"],
                    f"reranked_top{topk}/MRR_sc2se": reranked_metrics["schema2seal"]["MRR"],
                    f"reranked_top{topk}/MedianRank_sc2se": reranked_metrics["schema2seal"]["MedianRank"],
                })

    # Visualization
    if cfg.visualize_after_train:
        logger.info("Generating embedding visualizations...")
        vis_dir = os.path.join(
            cfg.train.checkpoint_dir,
            "visualizations"
        )

        visualize(
            cfg,
            checkpoint_path=best_checkpoint_path,
            split="all",
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

