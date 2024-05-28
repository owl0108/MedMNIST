from typing import List

import lightning as L
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


from resnet import resnet18
from convnext_model.convnext import convnext_tiny
from medmnist.info import INFO
from selector import DSelect_k

from utils import getACC, getAUC
from model import LinearModelHead, Encoder

class GeneralistModel(L.LightningModule):
    def __init__(self, tasks: List[str], lr, weighting: str='EW', head=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.INFO = INFO
        self.tasks = tasks
        self.task_id_dict = {task: i + 1 for i, task in enumerate(tasks)} # id start from 1
        self.lr = lr
        self.head_type = head
        self.head = None
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.batch_size = kwargs['batch_size']
        pretrained = kwargs['pretrained']
        encoder_type = kwargs['encoder_type']
        class_num_dict = {task: len(INFO[task]['label'].keys()) for task in tasks}
        self.encoder = Encoder(pretrained, encoder_type) # encoder_type has to be passed to DSelect_k, so don't explicitly specify in __init__

        self.embed_dim = 512 # maybe changed for GPU with more memory
        self.decoder = nn.ModuleDict({task: nn.Linear(self.embed_dim, class_num_dict[task]) for task in tasks})
        
        if weighting == 'EW': # equal weighting
            self.weighting = torch.mean
        else:
            raise NotImplementedError("Weighting strategy is not implemented")
        
        self.training_step_outputs = {task: [] for task in tasks}
        self.validation_step_outputs = {task: [] for task in tasks}
        self.test_step_outputs = {task: [] for task in tasks}
        self.kwargs = kwargs

        # head initialization
        if self.head_type == 'MultiheadAttention':
            print("Head is MultiheadAttention ...")
            self.head = MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
            # initialize random tokens
            task_num = len(tasks)
            batch_size = kwargs['batch_size']
            self.register_buffer('rand_tokens', torch.randn(size=[batch_size, task_num, self.embed_dim]))
        elif self.head_type is None:
            print("Only task-specific linear layer is used ...")
            self.head = None
        elif self.head_type not in [None, 'DSelect_k', "DSelect_k_LinearHead"]:
            raise NotImplementedError(f"Head type {self.head_type} is not implemented")
        
    def configure_model(self):
        # in order to give the correct self.device to DSelect_k
        # the model is moved to device after starting training
        if self.head_type == "DSelect_k_LinearHead":
            print("Head is DSelect_k_LinearHead ...")
            # linear heads as experts
            self.head = DSelect_k(task_name=self.tasks, encoder_class=LinearModelHead,
                                  decoders=self.decoder, device=self.device,
                                  multi_input=False, rep_grad=False, img_size=self.embed_dim, num_nonzeros=2,
                                  kgamma=1.0, **self.kwargs)
        if self.head_type == 'DSelect_k':
            print("Head is DSelect_k (for backbone) ...")
            # resnet or convnext as experts
            self.head = DSelect_k(task_name=self.tasks, encoder_class=type(self.encoder),
                                  decoders=self.decoder, device=self.device,
                                  multi_input=False, rep_grad=False, img_size=[3, 224, 224], num_nonzeros=2,
                                  kgamma=1.0, **self.kwargs)
        
    def forward(self, inputs, task):
        """Forward path

        Args:
            inputs: input data
            task: dataset name

        Returns:
            pred, head_ouptut (Tensor, Optional[Tensor]): prediction and head output (one-hot)
        """
        head_output = None
        if self.head_type is None:
            out = self.encoder(inputs)
            pred = self.decoder[task](out)
        elif self.head_type == 'DSelect_k':
            pred, head_output = self.head(inputs, task)
        elif self.head_type == 'DSelect_k_LinearHead':
            out = self.encoder(inputs)
            pred, head_output = self.head(out, task)
        elif self.head_type == 'MultiheadAttention':
            out = self.encoder(inputs) # (batch, embed_dim)
            # add seq_len dimension
            out = out.unsqueeze(1) # (batch, seq_len, embed_dim)
            if out.shape[0] < self.batch_size:
                out = torch.cat([out, self.rand_tokens[:out.shape[0], ...]], dim=1)
            else:   
                out = torch.concat([out, self.rand_tokens], dim=1)
            out= self.head(out, out, out)[0]
            task_id = self.task_id_dict[task]
            out = out[:, task_id, ...] #[batch_size, task_num + 1, embed_dim] only select the task_id-th part
            # divide separate parts into separate parts of decoder
            pred = self.decoder[task](out)
        return pred, head_output # optionally return the second var

    def _on_shared_step(self, batch, mode):
        head_outputs = []
        losses = []
        loss_dict = {}
        for (task, batch_for_a_single_task) in batch.items():
            #NOTE: what happens if none is detected
            if batch_for_a_single_task is None:
                loss_dict.pop(f"{mode}_loss_"+task, None) # remove the key if it exists
                continue
            else:
                inputs, target = batch_for_a_single_task
                output, head_output = self.forward(inputs, task) # head_output is not None when DSelect_k is used
                if mode == 'train':
                    self.training_step_outputs[task].append((output.detach(), target.detach()))
                elif mode == 'val':
                    self.validation_step_outputs[task].append((output.detach(), target.detach()))
                elif mode == 'test':
                    self.test_step_outputs[task].append((output.detach(), target.detach()))
                else:
                    raise ValueError(f"mode {mode} is not supported")
                
                task_type = INFO[task]['task']
                # compute task-wise loss
                if task_type == "multi-label, binary-class":
                    loss_fn = self.bcewithlogitsloss
                    target = target.float()
                else:
                    loss_fn = self.ce_loss
                    target = torch.squeeze(target, 1) # reshape target for CrossEnropyLoss
                loss = loss_fn(output, target)

                # compute regularization loss from DSeleckt_k
                if type(self.head) == DSelect_k:
                    head_loss_fn = self.head.entropy_reg_loss
                    head_loss = head_loss_fn(head_output)
                    loss = loss + head_loss
                losses.append(loss)
                loss_dict[f"{mode}_loss_"+task] = loss
        losses = torch.stack(losses) # to Tensor
        return self.weighting(losses), loss_dict

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        weighted_loss, loss_dict = self._on_shared_step(batch, 'train')
        log_dict = {'train_loss': weighted_loss}
        self.log('train_loss', weighted_loss)
        log_dict.update(loss_dict)
        self.log_dict(log_dict, on_step=True)
        # .zero_grad(), .backward(), .step() will be called automatically
        return weighted_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        weighted_loss, loss_dict = self._on_shared_step(batch, 'val')
        log_dict = {'val_loss': weighted_loss}
        log_dict.update(loss_dict)
        self.log_dict(log_dict)
        return weighted_loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        weighted_loss, loss_dict = self._on_shared_step(batch, 'test')
        log_dict = {'test_loss': weighted_loss}
        log_dict.update(loss_dict)
        self.log_dict(log_dict)
        return weighted_loss
    
    def _on_shared_epoch_end(self, mode):
        """Calculate the ACC and AUC.

        Args:
            mode (str): train, val, or test.
        """
        accs = []
        aucs = []
        if mode == 'train':
            step_outputs = self.training_step_outputs
        elif mode == 'val':
            step_outputs = self.validation_step_outputs
        elif mode == 'test':
            step_outputs = self.test_step_outputs
        for task in self.tasks:
            preds, gts = zip(*(step_outputs[task]))
            all_preds = torch.cat(preds)
            all_gts = torch.cat(gts)
            all_gts = all_gts.squeeze()
            task_type = INFO[task]['task']
            acc = getACC(all_preds, all_gts, task_type, self.device, threshold=0.5)
            auc = getAUC(all_preds, all_gts, task_type, self.device)
            accs.append(acc)
            aucs.append(auc)
            self.log_dict({f"{mode}_ACC_"+task: acc, f"{mode}_AUC_"+task: auc}, on_epoch=True)
        
        avg_acc = torch.stack(accs).mean()
        avg_auc = torch.stack(aucs).mean()
        self.log(f"{mode}_ACC", avg_acc, on_epoch=True)
        self.log(f"{mode}_AUC", avg_auc, on_epoch=True)
        step_outputs.clear() # clear the outputs

    def on_train_epoch_end(self):
        self._on_shared_epoch_end('train')
        self.training_step_outputs = {task: [] for task in self.tasks}

    def on_validation_epoch_end(self):
        self._on_shared_epoch_end('val')
        self.validation_step_outputs = {task: [] for task in self.tasks}
    
    def on_test_epoch_end(self):
        self._on_shared_epoch_end('test')
        self.task_step_outputs = {task: [] for task in self.tasks}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #TODO: configure scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": lr_scheduler,
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None
        }
        optim_config = {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optim_config