from typing import List

import lightning as L
import torch
import torch.nn as nn
from resnet import resnet18
from convnext_model.convnext import convnext_tiny
from medmnist.info import INFO
from medmnist.selector import DSelect_k

from utils import getACC, getAUC

class Encoder(nn.Module):
        def __init__(self, encoder_type='resnet18'):
            super(Encoder, self).__init__()
            # hidden_dim = 512
            if encoder_type == 'resnet18':
                self.network = resnet18(pretrained=False) # already global avg pooled
            elif encoder_type == 'convnext_tiny':
                self.network = convnext_tiny(pretrained=False,num_classes=512) # alrady global avg pooled
            # Use raw output from resnet18
            #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            
        def forward(self, inputs):
            out = self.network(inputs) # out: 512 planes, 7x7?
            out = torch.flatten(out, start_dim=1)
            #out = torch.flatten(self.avgpool(out), 1) # get one value from each plane
            # out = self.hidden_layer(out)
            return out

class LinearModelHead(nn.Module):
    def __init__(self, input_dim):
        super(LinearModelHead, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        return self.fc(x)


class GeneralistModel(L.LightningModule):
    def __init__(self, tasks: List[str], lr=0.001, weighting: str='EW', selector=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.INFO = INFO
        self.tasks = tasks
        self.lr = lr
        self.encoder = Encoder(kwargs['encoder_type'])
        self.selector_type = selector
        self.selector = None
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        class_num_dict = {task: len(INFO[task]['label'].keys()) for task in tasks}
        # self.encoder = Encoder()
        self.decoder = nn.ModuleDict({task: nn.Linear(512, class_num_dict[task]) for task in tasks})    
        if weighting == 'EW': # equal weighting
            self.weighting = torch.mean
        else:
            raise NotImplementedError("Weighting strategy is not implemented")
        
        self.training_step_outputs = {task: [] for task in tasks}
        self.validation_step_outputs = {task: [] for task in tasks}
        self.test_step_outputs = {task: [] for task in tasks}
        self.kwargs = kwargs
        
    def configure_model(self):
        # in order to give the correct self.device to DSelect_k
        # the model is moved to device after starting training
        if self.selector_type == 'DSelect_k':
            self.selector = DSelect_k(task_name=self.tasks, encoder_class=Encoder,
                                  decoders=self.decoder, device=self.device,
                                  multi_input=False, rep_grad=False, img_size=[3, 224, 224], num_nonzeros=2,
                                  kgamma=1.0, **self.kwargs)
        elif self.selector_type is None:
            self.selector = None
        else:
            raise NotImplementedError(f"Selector type {self.selector_type} is not implemented")
        
    def forward(self, inputs, task):
        """Forward path

        Args:
            inputs: input data
            task: dataset name

        Returns:
            pred, selector_ouptut (Tensor, Optional[Tensor]): prediction and selector output (one-hot)
        """
        # selector_output = None
        # repr = self.encoder(inputs)
        
        # else:
        #     pred, selector_output = self.selector(repr, task)
        if self.selector is None:
            out = self.encoder(inputs)
            pred = self.decoder[task](out)
            selector_output = None
        else:
            pred, selector_output = self.selector(inputs, task)
        return pred, selector_output # optionally return the second var

    def _on_shared_step(self, batch, mode):
        selector_outputs = []
        losses = []
        loss_dict = {}
        for (task, batch_for_a_single_task) in batch.items():
            #NOTE: what happens if none is detected
            if batch_for_a_single_task is None:
                loss_dict.pop(f"{mode}_loss_"+task, None) # remove the key if it exists
                continue
            else:
                inputs, target = batch_for_a_single_task
                output, selector_output = self.forward(inputs, task) # selector_output is None if selector is None
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

                # compute regularization loss from selector
                if self.selector is not None:
                    selector_loss_fn = self.selector.entropy_reg_loss
                    selector_loss = selector_loss_fn(selector_output)
                    # print("loss: ", loss)
                    # print("selector loss: ", selector_loss)
                    loss = loss + selector_loss
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