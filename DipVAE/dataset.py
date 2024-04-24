import os
import torch
from torch import Tensor
from typing import List, Optional, Sequence, Union
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_train=True):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        # print(self.images)
        # split = int(len(self.all_images) * 1 )# Reserve last 5 images for validation
        # if is_train:
        #     self.images = self.all_images[:split]
        # else:
        #     self.images = self.all_images[split:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # print(image[0])
        return image, image  # Using image as both input and target for simplicity


# VAEDataset class with the CustomDataset integration
class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # print(self.patch_size)
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        # Setup train and validation datasets using CustomDataset
        self.train_dataset = CustomDataset(
            os.path.join(self.data_dir, "train"), transform=transform, is_train=True
        )

        self.val_dataset = CustomDataset(
            os.path.join(self.data_dir, "val"), transform=transform, is_train=True
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()  # Reuse validation dataloader for testing
