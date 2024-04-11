
from torch.utils.data import DataLoader, Dataset
from typing import List
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from LibMTL.utils import get_root_dir

import medmnist
from info import INFO

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
    
def get_medmnist_dataloaders(task_name: List[str], batch_size: int, root_path: str,
                             resize: bool = True, download: bool = False, as_rgb: bool = True):
    data_loader = {}
    #iter_data_loader = {}
    for i, task in enumerate(task_name):
        data_loader[task] = {}
        #iter_data_loader[task] = {}
        info = INFO[task]
        for mode in ['train', 'val', 'test']:
            shuffle = True if mode == 'train' else False
            #NOTE: drop_last is set to True to get ROC AUC score without error
            # drop_last = True if mode == 'train' else False # ignore the incomplete batch when training
            drop_last = True
            DataClass = getattr(medmnist, info['python_class']) # get pre-defined dataset class

            #TODO: split=mode after debugging
            # construct a dataset
            img_dataset = DataClass(split=mode, root=root_path,
                                    transform=medmnist_transform(resize), download=download, as_rgb=as_rgb)
            

            # dictionary
            #NOTE: using num_workers>0 doesn't work with the cluster somehow ...
            data_loader[task][mode] = DataLoader(img_dataset, 
                                                    num_workers=0, 
                                                    pin_memory=True, 
                                                    batch_size=batch_size, 
                                                    shuffle=shuffle,
                                                    drop_last=drop_last)
            
            #TODO: delete the following line if not necessary                                  
            #iter_data_loader[task][mode] = iter(data_loader[task][mode])
    # return data_loader, iter_data_loader
    return data_loader        


class office_Dataset(Dataset):
    def __init__(self, dataset, root_path, task, mode):
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ])
        f = open(os.path.join(get_root_dir(), 'examples/office', 'data_txt/{}/{}_{}.txt'.format(dataset, task, mode)), 'r')
        self.img_list = f.readlines()
        f.close()
        self.root_path = root_path
        
    def __getitem__(self, i):
        img_path = self.img_list[i][:-1].split(' ')[0]
        y = int(self.img_list[i][:-1].split(' ')[1])
        img = Image.open(os.path.join(self.root_path, img_path)).convert('RGB')
        return self.transform(img), y
        
    def __len__(self):
        return len(self.img_list)
    
def office_dataloader(dataset, batchsize, root_path):
    if dataset == 'office-31':
        tasks = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-home':
        tasks = ['Art', 'Clipart', 'Product', 'Real_World']
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        data_loader[d] = {}
        iter_data_loader[d] = {}
        for mode in ['train', 'val', 'test']:
            shuffle = True if mode == 'train' else False
            drop_last = True if mode == 'train' else False
            txt_dataset = office_Dataset(dataset, root_path, d, mode)
#             print(d, mode, len(txt_dataset))
            data_loader[d][mode] = DataLoader(txt_dataset, 
                                              num_workers=2,
                                              pin_memory=True, 
                                              batch_size=batchsize, 
                                              shuffle=shuffle,
                                              drop_last=drop_last)
            iter_data_loader[d][mode] = iter(data_loader[d][mode])
    return data_loader, iter_data_loader
