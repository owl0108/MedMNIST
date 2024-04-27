from typing import List

import numpy as np
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import random_split, DataLoader, Dataset
import medmnist.datasets as medclasses # import all the datset classes
from medmnist.info import INFO, HOMEPAGE

# Note - you must have torchvision installed for this example
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class MedMNISTDataModule(L.LightningDataModule):
    def __init__(self, tasks: List[str], batch_size, iter_mode, data_dir: str = "./", resize=True, **kwargs):
        super().__init__()
        def medmnist_transform(resize=True):
            """Resize images of size 28x28 to 224x224
            Args:
                resize (bool): Defaults to True.
            Returns:
                torchvision.transforms.Compose: A sequence of image transformations.
            """
            if resize:
                data_transform = transforms.Compose(
                    [transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])])
            else:
                data_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])])
            return data_transform
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tasks = tasks
        self.iter_mode = iter_mode
        self.transform = medmnist_transform(resize)

    def setup(self, stage: str, download=False):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds_dict = {task: getattr(medclasses, INFO[task]['python_class'])(
                root=self.data_dir, split='train', transform=self.transform, download=download, as_rgb=True) for task in self.tasks}
            self.val_ds_dict = {task: getattr(medclasses, INFO[task]['python_class'])(
                root=self.data_dir, split='val', transform=self.transform, download=download, as_rgb=True) for task in self.tasks}
        # Assign test datasets for use in dataloader(s)
        if stage == "test":
            self.test_ds_dict = {task: getattr(medclasses, INFO[task]['python_class'])(
                root=self.data_dir, split='test', transform=self.transform, download=download, as_rgb=True) for task in self.tasks}
            
    def train_dataloader(self):
        # combined_loader returns batch: dict, batch_idx, dataloader_idx
        dl_list = {task: DataLoader(self.train_ds_dict[task], batch_size=self.batch_size, shuffle=True) for task in self.tasks}
        return CombinedLoader(dl_list, mode=self.iter_mode)

    def val_dataloader(self):
        dl_list = {task: DataLoader(self.val_ds_dict[task], batch_size=self.batch_size, shuffle=False) for task in self.tasks}
        return CombinedLoader(dl_list, mode=self.iter_mode)

    def test_dataloader(self):
        dl_list = {task: DataLoader(self.test_ds_dict[task], batch_size=self.batch_size, shuffle=False) for task in self.tasks}
        return CombinedLoader(dl_list, mode=self.iter_mode)