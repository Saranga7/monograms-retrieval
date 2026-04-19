import os 
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import logging
import pandas as pd
import torchvision.transforms.functional as F


logger = logging.getLogger(__name__)

class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img

        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top

        return F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


class MonogramPairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        splits_dir = None,
        split_regime = None, #"stratified" or "generalization" or "generalization_strict"
        fold = None,
        split = "train",   # "train", "test", or "all" when no split file is used
        transform = True,
        return_paths = False,
        use_processed_seals = False,
        kwargs = None
    ):
        
        if kwargs is None:
            kwargs = {}
        
        if split == "all":
            if splits_dir is not None or split_regime is not None or fold is not None:
                logger.warning("split='all' ignores splits_dir, split_regime, and fold.")
        elif isinstance(split, str):
            if splits_dir is None or split_regime is None:
                raise ValueError("For any named split, both splits_dir and split_regime must be provided.")

            if split_regime == "stratified":
                if fold is None:
                    raise ValueError("For split_regime='stratified', fold must be provided.")
            elif split_regime == "generalization":
                if fold is not None:
                    logger.warning("fold is ignored for split_regime='generalization'.")
            elif split_regime == "generalization_strict":
                if fold is not None:
                    logger.warning("fold is ignored for split_regime='generalization_strict'.")
            else:
                raise ValueError(f"Unsupported split_regime: {split_regime}")
        else:
            raise ValueError(f"Unsupported split: {split}")
        
        self.data_dir = data_dir
        self.split_name = split
        self.mode = "train" if split == "train" else "eval"
        self.transform = transform
        self.return_paths = return_paths
        self.use_processed_seals = use_processed_seals

        self.schema_dir = os.path.join(data_dir, "schemas")
        self.seal_dir = os.path.join(data_dir, "seals")
        self.seal_proc_dir = os.path.join(data_dir, "seals_proc_grad")   # change this if using different preprocessing

        self.seal_paths = []
        self.schema_paths = []
        self.seal_proc_paths = []

        valid_exts = (".jpg", ".png", ".jpeg")

        # if kwargs is None:
        #     kwargs = {} 
        
        # else:
        if kwargs.get("use_strong_augmentation", False):
            logger.info("Using strong augmentations for training.")
            self.use_strong_augmentation = True
            
        else:
            logger.info("Using default augmentations for training.")
            self.use_strong_augmentation = False

        if kwargs.get("use_grayscale", False):
            # converting to grayscale to avoid spurious color correlations, but keeping 3 channels for compatibility with pretrained models
            logger.info("Using grayscale augmentations for training.")
            self.use_grayscale = True
            self.normalize = T.Normalize(mean=[0.5] * 3, std = [0.5] * 3)
            self.color_transform = T.Grayscale(3)
        else:
            logger.info("Using RGB augmentations for training.")
            self.use_grayscale = False
            self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
            self.color_transform = lambda x: x

        if self.use_strong_augmentation:
            self.schema_T, self.seal_T = self._get_transforms_strong()
        else:
            self.schema_T, self.seal_T = self._get_transforms()

        # -------------------------------------------------
        # 1️⃣ Collect all schemas
        # -------------------------------------------------
        schema_files = [
            f for f in os.listdir(self.schema_dir)
            if f.lower().endswith(valid_exts)
        ]

        self.schema_map = {
            os.path.splitext(f)[0]: os.path.join(self.schema_dir, f)
            for f in schema_files
        }

        # -------------------------------------------------
        # 2️⃣ If fold is provided → read split file
        # -------------------------------------------------
        allowed_stems = None

        if split != "all":
            if split_regime == "stratified":
                split_file = os.path.join(
                    splits_dir,
                    split_regime,
                    f"fold_{fold}",
                    f"{split}.csv"
                )
            elif split_regime in ["generalization", "generalization_strict"]:
                split_file = os.path.join(
                    splits_dir,
                    split_regime,
                    f"{split}.csv"
                )
            else:
                raise ValueError(f"Unsupported split_regime: {split_regime}")

            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")

            split_df = pd.read_csv(split_file)

            if "monogram_id" not in split_df.columns:
                raise ValueError(f"'monogram_id' column not found in {split_file}")

            allowed_stems = set(split_df["monogram_id"].astype(str).str.strip())

        # -------------------------------------------------
        # 3️⃣ Match seals to schemas
        # -------------------------------------------------
        for f in sorted(os.listdir(self.seal_dir)):
            if f.lower().endswith(valid_exts):
                stem = os.path.splitext(f)[0]

                if stem in self.schema_map:

                    # If using folds → filter
                    if allowed_stems is not None:
                        if stem not in allowed_stems:
                            continue

                    self.seal_paths.append(os.path.join(self.seal_dir, f))
                    self.schema_paths.append(self.schema_map[stem])
                    self.seal_proc_paths.append(os.path.join(self.seal_proc_dir, f))
                else:
                    logger.warning(f"No matching schema for seal '{f}'")

        logger.info(f"[Fold {fold} | {split}] Loaded {len(self.seal_paths)} pairs")
     
    
    def __len__(self):
        return len(self.seal_paths)
    
    def __getitem__(self, idx):
        seal_path = self.seal_paths[idx]
        schema_path = self.schema_paths[idx]
        pair_id = os.path.splitext(os.path.basename(schema_path))[0]

        schema = Image.open(schema_path).convert('RGB')
        seal = Image.open(seal_path).convert('RGB')

        if self.transform:
            schema = self.schema_T(schema)
            seal = self.seal_T(seal)

        if self.use_processed_seals:
            seal_proc_path = self.seal_proc_paths[idx]
            seal_proc = Image.open(seal_proc_path).convert('RGB')

            if self.transform:
                seal_proc = self.schema_T(seal_proc)
        
            return {"schema" : schema, 
                    "seal": seal, 
                    "seal_proc" : seal_proc,
                    "pair_id": pair_id}
    
        if self.return_paths:
            return {"schema" : schema, 
                    "seal": seal, 
                    "schema_path": schema_path, 
                    "seal_path": seal_path,
                    "pair_id": pair_id}
        
        return {"schema" : schema, 
                "seal": seal,
                "pair_id": pair_id}

        
            


    def _get_transforms(self):
        """
        Returns transforms depending on the mode (train/val/test)
        """
        if self.mode == 'train':
        #     schema_transform = T.Compose([
        #     self.color_transform,
        #     T.Resize((224, 224)),
        #     # T.RandomResizedCrop((224, 224), scale=(0.95, 1.0)),
        #     T.ToTensor(),
        #     self.normalize
        # ])

            schema_transform = T.Compose([
                self.color_transform,
                PadToSquare(fill=0),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=3,
                        translate=(0.02, 0.02),
                        scale=(0.98, 1.02),
                        shear=2,
                        fill=0
                    )
                ], p=0.5),
                T.Resize((224, 224)),
                T.ToTensor(),
                self.normalize
            ])

            #  seal_transform = T.Compose([
            #     self.color_transform,
            #     T.RandomResizedCrop((224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            #     T.RandomRotation(5),
            #     T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            #     T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            #     T.ToTensor(),
            #     self.normalize
            # ])

            seal_transform = T.Compose([
                self.color_transform,
                T.Resize((224, 224)),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=5,
                        translate=(0.03, 0.03),
                        scale=(0.95, 1.05),
                        shear=3,
                        fill=0
                    )
                ], p=0.7),
                T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
                T.ToTensor(),
                self.normalize
            ])
             
        else:   # test / val/ all / inference-time transforms
            schema_transform = T.Compose([
                self.color_transform,
                PadToSquare(fill=0),
                T.Resize((224, 224)),
                T.ToTensor(),
                self.normalize
            ])
            seal_transform = T.Compose([
                self.color_transform,
                T.Resize((224, 224)),
                # T.CenterCrop(224),
                T.ToTensor(),
                self.normalize
            ])

        return schema_transform, seal_transform
    

    def _get_transforms_strong(self):
        """
        Returns stronger augmentations for training
        """
        if self.mode == "train":
            schema_transform = T.Compose([
                self.color_transform,
                T.RandomResizedCrop(224, scale=(0.92, 1.0), ratio=(0.97, 1.03)),
                T.ToTensor(),
                self.normalize,
            ])

            seal_transform = T.Compose([
                self.color_transform,
                T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.93, 1.07)),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=4,
                        translate=(0.03, 0.03),
                        scale=(0.95, 1.05),
                        shear=3
                    )
                ], p=0.7),
                T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.4)], p=0.8),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.4),
                T.RandomApply([T.RandomPerspective(distortion_scale=0.12, p=1.0)], p = 0.2),
                T.ToTensor(),
                T.RandomErasing(p = 0.2, scale=(0.01, 0.05), ratio=(0.5, 2.0), value="random"),
                self.normalize,
            ])
        else:
            schema_transform = T.Compose([
                self.color_transform,
                T.Resize((224, 224)),
                T.ToTensor(),
                self.normalize,
            ])

            seal_transform = T.Compose([
                self.color_transform,
                T.Resize((224, 224)),
                # T.CenterCrop(224),
                T.ToTensor(),
                self.normalize,
            ])

        return schema_transform, seal_transform



if __name__ == "__main__":
    # dataset = MonogramPairDataset(data_dir='/scratch/mahantas/datasets/MonogramSchema_Seal_pairs')
    # print(f"Dataset size: {len(dataset)}")
    # schema, seal = dataset[0]
    # print(f"Schema shape: {schema.shape}, Seal shape: {seal.shape}")
    # viz_image_pairs(dataset, num_pairs = 5)

    from utils import viz_image_pairs

    for fold in range(5):

        train_dataset = MonogramPairDataset(
            data_dir='/scratch/mahantas/datasets/MonogramSchema_Seal_pairs',
            splits_dir= '/scratch/mahantas/cross_modal_retrieval/splits/stratified_5fold',
            fold = fold,
            split = "train",
            transform = True,
            return_paths = False,
            use_processed_seals = False,
            kwargs={
                "use_strong_augmentation": False,
                "use_grayscale": True
            }
        )

        print(f"Train Dataset size: {len(train_dataset)}")

        viz_image_pairs(train_dataset, name = f"padded_fold_{fold}_train", num_pairs = 5)

      