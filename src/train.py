import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from tqdm import tqdm
import logging
import time
from omegaconf import OmegaConf

from src.dataset import MonogramPairDataset
from src.models import DualEncoder
from src.losses import CLIPLoss, ArcFaceCLIPLoss, EmbeddingConsistencyLoss
from src.evaluation import evaluate_retrieval_accuracy, evaluate_retrieval_with_fixed_gallery
from src.visualization import visualize
from src.utils import _log_metrics_to_wandb, _log_metrics_to_console


logger = logging.getLogger(__name__)


def _build_dataset(cfg, split_name):
    return MonogramPairDataset(
        data_dir=cfg.data.data_dir,
        splits_dir=cfg.data.splits_dir,
        split_regime=cfg.data.split_regime,
        fold=cfg.data.fold,
        split=split_name,
        transform=cfg.data.transforms.enable,
        return_paths=False,
        use_processed_seals=cfg.train.use_processed_seals,
        kwargs={
            "use_strong_augmentation": cfg.data.transforms.use_strong_augmentation,
            "use_grayscale": cfg.data.transforms.use_grayscale,
        },
    )


def _build_loader(cfg, dataset, batch_size, shuffle = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )



def _visualize_split(cfg, split_name, best_checkpoint_path):
    if not cfg.visualize_after_train:
        return

    logger.info(f"Generating embedding visualizations for {split_name}...")
    if cfg.data.split_regime == "stratified":
        vis_dir = os.path.join(
            cfg.train.checkpoint_dir,
            f"fold_{cfg.data.fold}",
            "visualizations"
        )
    else:
        vis_dir = os.path.join(
            cfg.train.checkpoint_dir,
            "visualizations"
        )

    visualize(
        cfg,
        split=split_name,
        output_dir=vis_dir,
        checkpoint_path=best_checkpoint_path,
    )

    if cfg.wandb.use_wandb_logging:
        try:
            wandb.log({
                f"{split_name}_plots/tsne_plot": wandb.Image(os.path.join(vis_dir, "tsne_plot.png")),
                f"{split_name}_plots/umap_plot": wandb.Image(os.path.join(vis_dir, "umap_plot.png")),
            })
        except Exception as e:
            logger.warning(f"Failed to log visualization images to W&B: {e}")


def _evaluate_stratified_regime(cfg, model, device, batch_size, best_checkpoint_path):
    logger.info("Evaluating best model on stratified test split...")

    test_dataset = _build_dataset(cfg, "test")
    test_loader = _build_loader(cfg, test_dataset, batch_size)

    metrics = evaluate_retrieval_accuracy(
        model=model,
        dataloader=test_loader,
        device=device,
        top_k=None,
        rerank_mode=None,
    )

    _log_metrics_to_console("[test]", metrics)

    if cfg.wandb.use_wandb_logging:
        _log_metrics_to_wandb("test", metrics)

    if cfg.test.reranking.enable:
        for topk in [10, 20, 50]:
            logger.info(f"Applying patch token-level reranking with top_k={topk} on test...")
            reranked_metrics = evaluate_retrieval_accuracy(
                model=model,
                dataloader=test_loader,
                device=device,
                top_k=topk,
                rerank_mode=cfg.test.reranking.mode,
                alpha=cfg.test.reranking.alpha,
                normalize_mode=cfg.test.reranking.normalize_mode,
            )

            _log_metrics_to_console("[test]", reranked_metrics, topk=topk)

            if cfg.wandb.use_wandb_logging:
                _log_metrics_to_wandb(f"reranked_top{topk}/test/", reranked_metrics)

    _visualize_split(cfg, "test", best_checkpoint_path)



def _evaluate_generalization_regime(cfg, model, device, batch_size, best_checkpoint_path):
    gallery_split = "test_gallery"

    if "strict" in cfg.data.split_regime:
        query_splits = ["test_medium_q2", "test_hard_q3"]
    else:
        query_splits = ["test_easy_q0q1", "test_medium_q2", "test_hard_q3"]

    gallery_dataset = _build_dataset(cfg, gallery_split)
    gallery_loader = _build_loader(cfg, gallery_dataset, batch_size)

    logger.info("Evaluating overall held-out gallery: test_gallery -> test_gallery")
    overall_metrics = evaluate_retrieval_with_fixed_gallery(
        model=model,
        query_loader=gallery_loader,
        gallery_loader=gallery_loader,
        device=device,
        top_k=None,
        rerank_mode="none",
    )

    _log_metrics_to_console("[test_gallery -> test_gallery]", overall_metrics)

    if cfg.wandb.use_wandb_logging:
        _log_metrics_to_wandb("test_gallery", overall_metrics)

    for eval_split in query_splits:
        logger.info(f"Evaluating queries from {eval_split} against gallery {gallery_split}")

        query_dataset = _build_dataset(cfg, eval_split)
        query_loader = _build_loader(cfg, query_dataset, batch_size)

        metrics = evaluate_retrieval_with_fixed_gallery(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            top_k=None,
            rerank_mode="none",
        )

        _log_metrics_to_console(f"[{eval_split} -> {gallery_split}]", metrics)

        if cfg.wandb.use_wandb_logging:
            _log_metrics_to_wandb(f"{eval_split}_vs_{gallery_split}", metrics)

        if cfg.test.reranking.enable:
            for topk in [10, 20, 50]:
                logger.info(f"Applying patch token-level reranking with top_k={topk} on {eval_split} vs {gallery_split}...")
                reranked_metrics = evaluate_retrieval_with_fixed_gallery(
                    model=model,
                    query_loader=query_loader,
                    gallery_loader=gallery_loader,
                    device=device,
                    top_k=topk,
                    rerank_mode=cfg.test.reranking.mode,
                    alpha=cfg.test.reranking.alpha,
                    normalize_mode=cfg.test.reranking.normalize_mode,
                )

                _log_metrics_to_console(f"[{eval_split} -> {gallery_split}]", reranked_metrics, topk=topk)

                if cfg.wandb.use_wandb_logging:
                    _log_metrics_to_wandb(
                        f"reranked_top{topk}/{eval_split}_vs_{gallery_split}",
                        reranked_metrics,
                    )

        _visualize_split(cfg, eval_split, best_checkpoint_path)

    # optional: also visualize the shared gallery once
    _visualize_split(cfg, "test_gallery", best_checkpoint_path)





def train(cfg):
    
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.wandb.use_wandb_logging:
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=cfg.wandb.tags + [cfg.data.split_regime],
            config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    logger.info(f"Using Device: {device}")
    logger.info(f"Fold: {cfg.data.fold}")
    
    # Dataset
    train_dataset = _build_dataset(cfg, split_name="train")
    val_dataset = _build_dataset(cfg, split_name="val")


    if cfg.data.batch_size == "full":
        batch_size = len(train_dataset)
        logger.info(f"Using full batch size: {batch_size}")
    else:
        batch_size = cfg.data.batch_size

    # Dataloaders
    train_loader = _build_loader(cfg, train_dataset, batch_size, shuffle = True)
    val_loader = _build_loader(cfg, val_dataset, batch_size)
        

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

    if cfg.train.use_processed_seals:
        consistency_criterion = EmbeddingConsistencyLoss().to(device)
        lambda_aux = cfg.loss.lambda_aux
        # lambda_proc = cfg.loss.lambda_proc

    
    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    best_val_loss = float('inf')
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

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0
        val_loss_aux = 0
        val_loss_clip = 0
        # val_loss_clip_proc = 0
        with torch.no_grad():
            for batch in val_loader:
                
                if cfg.train.use_processed_seals:
                    schema, seal, seal_proc = batch["schema"].to(device), batch["seal"].to(device), batch["seal_proc"].to(device)
                else:
                    schema, seal = batch["schema"].to(device), batch["seal"].to(device)

                z_schema, z_seal = model(schema, seal)
                clip_loss, _ = criterion(z_schema, z_seal)
                val_loss_clip += clip_loss.item()

                if cfg.train.use_processed_seals:
                    z_seal_proc = model.encode_seal(seal_proc)
                    loss_aux = consistency_criterion(z_seal, z_seal_proc)
                    val_loss_aux += loss_aux.item()

                    loss = clip_loss + lambda_aux * loss_aux

                    # clip_loss_proc, _ = criterion(z_schema, z_seal_proc)
                    # val_loss_clip_proc += clip_loss_proc.item()

                    # loss = clip_loss + lambda_aux * loss_aux + lambda_proc * clip_loss_proc
                else:
                    loss = clip_loss

                val_loss += loss.item()
        val_loss /= len(val_loader)
        if cfg.train.use_processed_seals:
            val_loss_aux /= len(val_loader)
            val_loss_clip /= len(val_loader)
            # val_loss_clip_proc /= len(val_loader)

            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} \
                        (clip={train_loss_clip:.4f}, aux={train_loss_aux:.4f}) \nval_loss={val_loss:.4f} \
                        (clip={val_loss_clip:.4f}, aux={val_loss_aux:.4f})")
       
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
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
                    "train/loss_clip": train_loss_clip,
                    "train/loss_aux": train_loss_aux,
                    # "train/loss_clip_proc": train_loss_clip_proc,
                    "train/total_train_loss": train_loss,
                    
                    "val/loss_clip": val_loss_clip,
                    "val/loss_aux": val_loss_aux,
                    # "val/loss_clip_proc": val_loss_clip_proc,
                    "val/total_val_loss": val_loss
                })

                
            else:
                wandb.log({
                    "train/loss": train_loss, 
                    "val/loss": val_loss, 
                })


        

        # -------------------------
        # CHECKPOINTING
        # -------------------------
        if epoch % cfg.train.log_interval == 0:
            checkpoint_path = cfg.train.checkpoint_dir
            os.makedirs(checkpoint_path, exist_ok=True)

            if cfg.data.split_regime == "stratified":
                save_dir = os.path.join(
                    checkpoint_path,
                    f"fold_{cfg.data.fold}"
                )
            else:
                save_dir = os.path.join(
                    checkpoint_path,
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = cfg.train.checkpoint_dir
            os.makedirs(checkpoint_path, exist_ok=True)
        
            if cfg.data.split_regime == "stratified":
                save_dir = os.path.join(
                    checkpoint_path,
                    f"fold_{cfg.data.fold}"
                )
            else:
                save_dir = os.path.join(
                    checkpoint_path,
                )
            os.makedirs(save_dir, exist_ok=True)

            best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"Best model saved at epoch {epoch} with val_loss={best_val_loss:.4f}")

    logger.info("Training complete.\n\n-----------------------------\n")

    





    # Testing retrieval performance of the best model
    logger.info("Loading best model for evaluation...")
    del model
    torch.cuda.empty_cache()
    
    model = DualEncoder(cfg).to(device)
    state_dict = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    
    if cfg.data.split_regime == "stratified":
        _evaluate_stratified_regime(
            cfg=cfg,
            model=model,
            device=device,
            batch_size=batch_size,
            best_checkpoint_path=best_checkpoint_path,
        )
    elif cfg.data.split_regime.startswith("generalization"):
        _evaluate_generalization_regime(
            cfg=cfg,
            model=model,
            device=device,
            batch_size=batch_size,
            best_checkpoint_path=best_checkpoint_path,
        )
    else:
        raise ValueError(f"Unsupported split_regime: {cfg.data.split_regime}")


    
    if cfg.wandb.use_wandb_logging:
        wandb.finish()
    
    # Finishing
    end = time.time()
    total_seconds = end - start
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60    
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

