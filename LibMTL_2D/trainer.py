from LibMTL import Trainer

import numpy as np
import os
import torch
import wandb

class Trainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, save_path, load_path, **kwargs):
        #TODO: add project title as param arg
        wandb.init(project='LibMTL_2D', config=kwargs)
        super().__init__(task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, save_path, load_path, **kwargs)

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
        self.meter.get_score()
        wandb.log({
            'epoch': epoch,
            'time': self.meter.end_time,
            f'{mode}_loss': self.meter.loss_item[-1],
            f'{mode}_score': self.meter.results[-1]
        })
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

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)

                    # _compute_loss returns a loss for all tasks, 1 x tn
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        if batch_index == (len(train_batch)-1) // 4:
                            wandb.log({
                                f'{task}_train_loss': train_losses[tn].item(),
                                'epoch': epoch,
                                'batch': batch_index
                            })

                    # calculate loss for all tasks for a particular batch
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        # _compute_loss returns a loss for one batch (= a 1x1 Tensor)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        if batch_index == (len(train_batch)-1) // 4:
                            wandb.log({
                                f'{task}_train_loss': train_losses[tn].item(),
                                'epoch': epoch,
                                'batch': batch_index
                                })
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