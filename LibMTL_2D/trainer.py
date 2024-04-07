from LibMTL import Trainer

import numpy as np
import os
import torch
import wandb
from tqdm import trange

class Trainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, save_path, load_path, **kwargs):
        '''
        One gpu training by default. The model is moved to GPU by initialization.
        '''
        #TODO: add project title as param arg
        #wandb.init(project='LibMTL_2D', config=kwargs)
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
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label

    def validation(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    # for tn, task in enumerate(self.task_name):
                    #     wandb.log({
                    #         f'{task}_{mode}_loss': test_losses[tn].item(),
                    #         'epoch': epoch,
                    #         'batch': batch_index
                    #     })
                    self.meter.update(test_preds, test_gts)
            else:
                test_losses = torch.zeros(self.task_num).to(self.device)
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        # process gt to be compatible with loss function
                        test_gt = self.process_gts(test_gt, task)
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_losses[tn] = self._compute_loss(test_pred, test_gt, task)
                        # wandb.log({
                        #     f'{task}_{mode}_loss': test_losses[tn].item(),
                        #     'epoch': epoch,
                        #     'batch': batch_index
                        #     })
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        # calls score_fun
        self.meter.get_score()
        # wandb.log({
        #     'epoch': epoch,
        #     'time': self.meter.end_time,
        #     f'{mode}_loss': self.meter.loss_item[-1],
        #     f'{mode}_score': self.meter.results
        # })
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
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in trange(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            #TODO: calculate AUC only once per epoch using following lines
            epoch_preds = torch.zeros([train_batch * self.bs, self.task_num]) # 2D
            epoch_gts = torch.zeros(train_batch * self.bs) # 1D
            for batch_index in trange(train_batch):
                if not self.multi_input:
                    # _process_data retrieves the batch and moves it to device
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)

                    # _compute_loss returns a loss for all tasks, 1 x task_num
                    train_losses = self._compute_loss(train_preds, train_gts)
                    #TODO: should I compute this every batch? or concat preds and do it at the end
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    # calculate loss for all tasks for a particular batch_num
                    for tn, task in enumerate(self.task_name):
                        # _process_data retrieves the batch and moves it to device
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        # reshape ground truth label to be compatible with loss function
                        train_gt = self.process_gts(train_gt, task)
                        # _compute_loss returns a loss for one batch (= a 1x1 Tensor)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        # logging
                        # if batch_index == train_batch-1 // 4:
                        #     wandb.log({
                        #         f'{task}_train_loss': train_losses[tn].item(),
                        #         'epoch': epoch,
                        #         'batch': batch_index
                        #         })
                        self.meter.update(train_pred, train_gt, task)

                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            # after one epoch
            self.meter.record_time('end')
            # calculate average loss and score for an epoch
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                # use custom defined validation method for logging purpose
                val_improvement = self.validation(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight