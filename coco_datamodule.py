from typing import Any, Dict, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from torchvision import transforms
from lightning import LightningDataModule


class COCODataModule(LightningDataModule):
    """`LightningDataModule` for the COCO dataset.

    COCO (Common Objects in Context) dataset contains images with object detection, segmentation, 
    and keypoint annotations. This module loads the dataset, applies transformations, 
    and provides dataloaders for training, validation, and testing.

    Official dataset link: https://cocodataset.org/
    """

    def __init__(
        self,
        data_dir: str = "data/coco/",
        train_ann_file: str = "annotations/instances_train2017.json",
        val_ann_file: str = "annotations/instances_val2017.json",
        test_ann_file: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the COCODataModule.

        :param data_dir: Directory where COCO images and annotations are stored.
        :param train_ann_file: JSON annotation file for training.
        :param val_ann_file: JSON annotation file for validation.
        :param test_ann_file: JSON annotation file for testing (optional).
        :param batch_size: Number of samples per batch.
        :param num_workers: Number of worker threads for data loading.
        :param pin_memory: Whether to pin memory (recommended for GPU).
        """
        super().__init__()

        self.save_hyperparameters()

        # Define transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize((512, 512)),  # Resize all images to 512x512
                transforms.ToTensor(),  # Convert to Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
            ]
        )

        self.data_train: Optional[CocoDetection] = None
        self.data_val: Optional[CocoDetection] = None
        self.data_test: Optional[CocoDetection] = None

    def prepare_data(self) -> None:
        """Prepare COCO dataset (download if needed)."""
        # COCO dataset does not provide an automatic download via torchvision, so it must be manually downloaded.
        print("Ensure COCO dataset is downloaded and placed in", self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the dataset and apply transformations."""
        if stage in (None, "fit"):
            self.data_train = CocoDetection(
                root=f"{self.hparams.data_dir}/train2017",
                annFile=f"{self.hparams.data_dir}/{self.hparams.train_ann_file}",
                transform=self.transforms,
            )
            self.data_val = CocoDetection(
                root=f"{self.hparams.data_dir}/val2017",
                annFile=f"{self.hparams.data_dir}/{self.hparams.val_ann_file}",
                transform=self.transforms,
            )

        if stage in (None, "test") and self.hparams.test_ann_file:
            self.data_test = CocoDetection(
                root=f"{self.hparams.data_dir}/test2017",
                annFile=f"{self.hparams.data_dir}/{self.hparams.test_ann_file}",
                transform=self.transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader (if available)."""
        if self.data_test is not None:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
        return None

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up resources after training/testing."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Save state dictionary (not needed for this DataModule)."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary (not needed for this DataModule)."""
        pass


if __name__ == "__main__":
    coco_dm = COCODataModule()
    coco_dm.prepare_data()
    coco_dm.setup("fit")

    train_loader = coco_dm.train_dataloader()
    for images, targets in train_loader:
        print("Batch size:", len(images))
        break
