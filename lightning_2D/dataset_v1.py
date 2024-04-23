import os
from typing import List, Callable, Optional

import PIL
import numpy as np
import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
from medmnist import INFO

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class MedMNIST(Dataset):
    # flag = ... # will be set later

    def __init__(
        self,
        split: str,
        tasks: Optional[List[str]] =None,
        transform: Optional[Callable] =None,
        target_transform: Optional[Callable] =None,
        as_rgb: Optional[bool] =False,
        data_root: Optional[str] ='./',
        size: Optional[int] =None,
        mmap_mode: Optional[str]=None,
    ):
        """
        Args:

            split (string): 'train', 'val' or 'test', required
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default: False.
            as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
            size (int, optional): The size of the returned images. If None, use MNIST-like 28. Default: None.
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.
            root (string, optional): Root directory of dataset. Default: `~/.medmnist`.

        """

        # Here, `size_flag` is blank for 28 images, and `_size` for larger images, e.g., "_64".
        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            assert size in self.available_sizes
            self.size = size
            self.size_flag = f"_{size}"

        self.INFO = INFO

        if data_root is not None and os.path.exists(data_root):
            self.data_root = data_root
        else:
            raise RuntimeError(
                "Failed to setup the default `data_root` directory. "
                + "Please specify and create the `data_root` directory manually."
            )

        self.tasks = tasks

        npz_files = {}
        for flag in self.tasks:
            if not os.path.exists(
                os.path.join(self.root, f"{flag}{self.size_flag}.npz")
            ):
                raise RuntimeError(
                    f"Dataset not found for {flag}"
                    )
            npz_files[flag] = np.load(
                os.path.join(self.root, f"{flag}{self.size_flag}.npz"),
                mmap_mode=mmap_mode,
            )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        # load data
        all_imgs = []
        all_labels = []
        all_flags = []
        if self.split in ["train", "val", "test"]:
            for flag in self.tasks:
                n_sample =  npz_files[flag][f"{self.split}_images"].shape[0]
                assert n_sample == npz_files[flag][f"{self.split}_labels"].shape[0]
                all_imgs.extend(npz_files[flag][f"{self.split}_images"].tolist()) # list of (H, W) or (H, W, 3)
                all_labels.extend(npz_files[flag][f"{self.split}_labels"].tolist()) # list of labels
                all_flags.extend([flag]*n_sample)
            self.imgs = all_imgs
            self.labels = all_labels
            self.img_flags = all_flags
        else:
            raise ValueError
    
    def to_rgb(self, img_arr):
        """Convert images to RGB format

        Args:
            img_arr(np.array): shape (N, H, W) where N is number of images

        Returns:
            np.array: shape (N, H, W, 3)
        """
        img_arr = np.repeat(img_arr[..., np.newaxis], 3, -1)
        return img_arr

    def __len__(self):
        sum = sum([self.INFO[flag]["n_samples"][self.split] for flag in self.tasks])
        assert sum == len(self.imgs)
        return len(self.imgs)

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img_flag = self.img_flags[index]
        img = PIL.Image.fromarray(img)
        if self.as_rgb:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, img_flag

class MedMNISTDataModule(L.LightningDataModule):
    def __init__(self, tasks: List[str], batch_size, data_dir: str = "./", resize=True, ):
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
        self.transform = medmnist_transform(resize)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = MedMNIST(split='train', tasks=self.tasks,
                                          data__root=self.data_dir, transform=self.transform,
                                          as_rgb=True)
            self.val_ds = MedMNIST(split='val', tasks=self.tasks,
                                        data_root=self.data_dir, transform=self.transform,
                                        as_rgb=True)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = MedMNIST(split='test', tasks=self.tasks,
                                        data_root=self.data_dir, transform=self.transform,
                                        as_rgb=True)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)