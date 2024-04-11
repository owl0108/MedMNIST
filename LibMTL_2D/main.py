import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from time import strftime

from create_dataset import office_dataloader, get_medmnist_dataloaders

from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.metrics import AccMetric

from medmnist import INFO

from config import LibMTL_args, prepare_args
from loss import CELoss, BCELoss
from metric import MedMnistMetric
from trainer import Trainer

SCRATCH_PATH = '/scratch/izar/ishii'

def parse_args(parser):
    parser.add_argument('--dataset', default='medmnist-2d', type=str, help='default is 2d datasets from MedMNIST')
    #TODO: configure args
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default=SCRATCH_PATH, type=str, help='dataset path')
    parser.add_argument('--resize', default=True, type=bool, help='resize images to 224x224')
    parser.add_argument('--download', default=False, type=bool, help='download dataset')
    parser.add_argument('--run_name', default=strftime("%m-%d-%Y_%H:%M:%S"), type=str, help='run_name for wandb')
    return parser.parse_args()

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    MNIST_INFO = INFO # dictionary from MedMNIST package containing various dataset-specific info
    if params.dataset == 'medmnist-2d':
        task_name = ['pathmnist', 'octmnist', 'pneumoniamnist', 'chestmnist', 'dermamnist', 'retinamnist',
                      'breastmnist', 'bloodmnist', 'organsmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
        # task_name = ['octmnist', 'breastmnist']
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))
    
    # define tasks
    task_dict = {}
    for task in task_name:
        task_type = MNIST_INFO[task]['task']
        if task_type == "multi-label, binary-class":
            loss_fn = BCELoss()
        else:
            loss_fn = CELoss()
        
        task_dict[task] = {'metrics': ['AUC', 'ACC'],
                       'metrics_fn': MedMnistMetric(task_type),
                       'loss_fn': loss_fn,
                       'weight': [1, 1], # 1 if the metric should be maximized, 0 if it should be minimized
                       'class_num': len(MNIST_INFO[task]['label']),
                       'task_type': task_type
                       }
    
    # prepare dataloaders
    data_loader = get_medmnist_dataloaders(task_name=task_name, batch_size=params.bs, root_path=params.dataset_path,
                                           resize=params.resize, download=params.download, as_rgb=True)
    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}
    
    # define encoder and decoders
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            # hidden_dim = 512
            self.resnet_network = resnet18(pretrained=False)
            
            # Use raw output from resnet18
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # self.hidden_layer_list = [nn.Linear(512, hidden_dim),
            #                           nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            # self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            # initialization
            # self.hidden_layer[0].weight.data.normal_(0, 0.005)
            # self.hidden_layer[0].bias.data.fill_(0.1)
            
        def forward(self, inputs):
            out = self.resnet_network(inputs) # out: 512 planes, 7x7?
            out = torch.flatten(self.avgpool(out), 1) # get one value from each plane
            # out = self.hidden_layer(out)
            return out

    # TODO: need to change in in_channels to 512 * block.expansion for resnet50>
    decoders = nn.ModuleDict({task: nn.Linear(512, task_dict[task]['class_num']) for task in list(task_dict.keys())})
    
    model = Trainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=Encoder, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          wandb_name=params.run_name,
                          **kwargs)
    if params.mode == 'train':
        model.train(train_dataloaders=train_dataloaders, 
                        val_dataloaders=val_dataloaders,
                        test_dataloaders=test_dataloaders, 
                        epochs=params.epochs)
    elif params.mode == 'test':
        model.test(test_dataloaders)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)