import os 
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import logging
logging.basicConfig(level=logging.INFO)

# from utils import viz_image_pairs

logger = logging.getLogger(__name__)


class MonogramPairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        splits_dir = None,
        fold = None,
        split = "train",   # "train", "test", or "all" when no split file is used
        transform = True,
        return_paths = False,
        use_processed_seals = False,
        kwargs = None
    ):
        
        if split == "all":
            if splits_dir is not None or fold is not None:
                logger.warning("split='all' ignores splits_dir and fold; loading all matched pairs.")
        elif split in {"train", "test"}:
            if (splits_dir is None) != (fold is None):
                raise ValueError("For split='train' or 'test', both splits_dir and fold must be provided together.")
        else:
            raise ValueError(f"Unsupported split: {split}. Use 'train', 'test', or 'all'.")
        
        self.data_dir = data_dir
        self.mode = split
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

        if split != "all" and splits_dir is not None and fold is not None:
            split_file = os.path.join(
                splits_dir,
                f"fold_{fold}",
                f"{split}.txt"
            )

            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with open(split_file, "r") as f:
                schema_paths = [line.strip() for line in f.readlines()]

            # extract stems from schema paths
            allowed_stems = {
                os.path.splitext(os.path.basename(p))[0]
                for p in schema_paths
            }

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
        
            return {"schema" : schema, "seal": seal, "seal_proc" : seal_proc}
    
        if self.return_paths:
            return {"schema" : schema, "seal": seal, "schema_path": schema_path, "seal_path": seal_path}
        
        return {"schema" : schema, "seal": seal}

        
            


    def _get_transforms(self):
        """
        Returns transforms depending on the mode (train/val/test)
        """

        if self.mode == 'train':
            schema_transform = T.Compose([
                self.color_transform,
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            seal_transform = T.Compose([
                self.color_transform,
                T.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
                T.RandomRotation(5),
                T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.4),
                T.ToTensor(),
                self.normalize
            ])
        else:   # test / all / inference-time transforms
            schema_transform = T.Compose([
                self.color_transform,
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

    for fold in range(5):

        train_dataset = MonogramPairDataset(
            data_dir='/scratch/mahantas/datasets/MonogramSchema_Seal_pairs',
            splits_dir= '/scratch/mahantas/cross_modal_retrieval/splits',
            fold = fold,
            split = "train",
            kwargs={
            "use_strong_augmentation": True
        }
        )

        print(f"Train Dataset size: {len(train_dataset)}")

        # viz_image_pairs(train_dataset, name = f"strong_fold_{fold}_train", num_pairs = 5)

        test_dataset = MonogramPairDataset(
            data_dir='/scratch/mahantas/datasets/MonogramSchema_Seal_pairs',
            splits_dir= '/scratch/mahantas/cross_modal_retrieval/splits',
            fold = fold,
            split = "test",
            kwargs={
            "use_strong_augmentation": True
        }
        )

        print(f"Test Dataset size: {len(test_dataset)}")

        # viz_image_pairs(test_dataset, name = f"strong_fold_{fold}_test", num_pairs = 5)
