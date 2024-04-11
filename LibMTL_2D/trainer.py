from LibMTL import Trainer

import traceback

import numpy as np
import os
import torch
import wandb
from tqdm import trange

class Trainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, save_path, load_path, wandb_name, **kwargs):
        '''
        One gpu training by default. The model is moved to GPU by initialization.
        '''
        #TODO: add project title as param arg
        wandb.init(project='LibMTL_2D', config=kwargs, name=wandb_name)
        super().__init__(task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, save_path, load_path, **kwargs)
    
    def process_gts(self, gts, task_name=None):
        #TODO: complete implementation
        r'''The processing of ground truth labels for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``gts`` with type :class:`torch.Tensor` is the ground truth of this task.
        - otherwise, ``task_name`` is invalid and ``gts`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        # implemented only for multi-input case
        if not self.multi_input:
            return gts
        else:
             # Recover task_type here
            task_type = self.task_dict[task_name]['task_type']
            if task_type == 'multi-label, binary-class':
                gts = gts.to(torch.float32)
            else:
                gts = torch.squeeze(gts, 1).long()
            return gts
        
    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            # catching StopIteration exception
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def _run_single_epoch(self, epoch, total_num_batches, loader, mode):
        """Run a single epoch.

        Args:
            epoch (int): nth epoch
            total_num_batches (int): Maximum number of batches among all the task datasets
            loader (dict): Dictionary of dataloader

        Raises:
            NotImplementedError: only self.multi_input = True is implemented
        
        Return:
            epoch_preds (dict): Dictionary containing predictions for each task
            epoch_gts (dict): Dictionary containing ground truth labels for each task
        """
        #TODO: calculate AUC and ACC only once per epoch using following lines
        # train_batch: how many batches in total
        epoch_preds = dict()
        epoch_gts = dict()
        for task in self.task_name:
            task_type = self.task_dict[task]['task_type']
            class_num = self.task_dict[task]['class_num']
            epoch_preds[task] = np.zeros([total_num_batches * self.bs, class_num]) # 2d (dataset_size, class_num)
            if task_type == 'multi-label, binary-class':
                epoch_gts[task] = np.zeros([total_num_batches * self.bs, class_num])
            else:
                epoch_gts[task] = np.zeros([total_num_batches * self.bs]) # 1d (dataset_size)

        for batch_index in trange(total_num_batches):
            if not self.multi_input:
                raise NotImplementedError()
                
            else:
                losses = torch.zeros(self.task_num).to(self.device)
                # calculate loss for all tasks for a particular batch_num
                for tn, task in enumerate(self.task_name):
                    # _process_data retrieves the batch and moves it to device
                    input, gt = self._process_data(loader[task])
                    pred = self.model(input, task) # returns dict containing pred for each task
                    pred = self.process_preds(pred[task], task)
                    # reshape ground truth label to be compatible with loss function
                    gt = self.process_gts(gt, task)
                    # populate numpy arrays
                    epoch_preds[task][batch_index * self.bs: (batch_index + 1) * self.bs] = pred.cpu().detach().numpy()
                    epoch_gts[task][batch_index * self.bs: (batch_index + 1) * self.bs] = gt.cpu().detach().numpy()
                    # _compute_loss returns a loss for one batch (= a 1x1 Tensor)
                    losses[tn] = self._compute_loss(pred, gt, task)
                    # logging
                    if  mode == "train":
                        #print("val logging in progress ...")
                        if batch_index % ((total_num_batches-1) // 4) == 0:
                            wandb.log({
                                f'{task}_train_loss': losses[tn].item(),
                                'epoch': epoch,
                                'batch': batch_index
                                })
                    # self.meter.update(train_pred, train_gt, task)
                    # raise AssertionError()

            if mode == "train":
                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
        return epoch_preds, epoch_gts


    def validation(self, val_dataloaders, epoch=None, mode='val', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        val_loader, val_batch = self._prepare_dataloaders(val_dataloaders)
        val_batch = max(val_batch)
        
        self.model.eval()
        self.meter.record_time('begin')
        print("validation epoch starting")
        with torch.no_grad():
            epoch_preds, epoch_gts = self._run_single_epoch(epoch, val_batch, val_loader, mode='val')
        self.meter.record_time('end')
        # calls score_fun
        [self.meter.update(epoch_preds[task], epoch_gts[task], task) for task in self.task_name]
        self.meter.get_score()
        log_dict = {'epoch': epoch, f'{mode}_loss': self.meter.loss_item[-1]}
        log_dict.update({f"{mode}_AUC_score_{task}" : self.meter.results[task][0] for task in self.task_name})
        log_dict.update({f"{mode}_ACC_score_{task}" : self.meter.results[task][1] for task in self.task_name})
        wandb.log(log_dict)
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement
    

    def train(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False):
        '''The training process of multi-task learning.

        Args
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        print("Starting the training ...")
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        if isinstance(train_dataloaders, dict): self.bs = next(iter(train_dataloaders.values())).batch_size
        elif isinstance(train_dataloaders, torch.utils.data.DataLoader): self.bs = train_dataloaders.batch_size
        else: raise ValueError("train_dataloaders should be a dict or torch.utils.data.DataLoader")

        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in trange(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            epoch_preds, epoch_gts = self._run_single_epoch(epoch, train_batch, train_loader, mode='train')
            # after one epoch
            self.meter.record_time('end')
            # calculate average loss and score for an epoch
            [self.meter.update(epoch_preds[task], epoch_gts[task], task) for task in self.task_name]
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                # use custom defined validation method for logging purpose
                val_improvement = self.validation(val_dataloaders, epoch, mode='val', return_improvement=True)
            # why there is testing here???
            # self.test(test_dataloaders, epoch, mode='test')

            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                model_save_path = os.path.join(self.save_path, 'best.pt')
                torch.save(self.model.state_dict(), model_save_path)
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.test(test_dataloaders, epoch, mode='test', return_improvement=False)
        self.meter.display_best_result()
        wandb.save(model_save_path) # save the best model
        if return_weight:
            return self.batch_weight

    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        self.validation(test_dataloaders, epoch, mode='test', return_improvement=return_improvement)