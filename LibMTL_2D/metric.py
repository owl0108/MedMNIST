import numpy as np
from LibMTL.metrics import AbsMetric
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import Tensor

def getAUC(y_true: Tensor, y_score: Tensor, task: str):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true: Tensor, y_score: Tensor, task: str, threshold: float=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret



class MedMnistMetric(AbsMetric):
    r"""Implementation of AUC and Accuracy.
    """
    def __init__(self, task_type):
        super().__init__()
        self.task_type = task_type
        self.batch_count = 0
        
    def update_fun(self, pred: Tensor, gt: Tensor):
        '''Called at the end of every batch
        '''
        # batch-wise auc and acc
        auc = getAUC(gt, pred, self.task_type)
        acc = getACC(gt, pred, self.task_type)
        self.record = [auc, acc]
        
    def score_fun(self):
        '''Called at the end of epoch
        '''
        # modify here 
        # divide by how many batches were there
        # auc_sum = sum([rec[0] for rec in self.record])
        # acc_sum = sum([rec[1] for rec in self.record])
        
        #return [auc_sum/self.batch_count, acc_sum/self.batch_count]
        return self.record

