from torchmetrics import AUROC, Accuracy
import torch
from torch import Tensor

def getAUC(y_preds: Tensor, y_gts: Tensor, task_type: str, device):
    """AUC metric

    Args:
        y_preds: the predicted score of each class, shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        y_gts: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        task_type: the task of current dataset
    """
    if task_type == "multi-label, binary-class":
        metric = AUROC(task='multilabel', num_labels=y_preds.shape[1],
                       average='macro')
    elif task_type == "binary-class":
        if y_preds.dim() == 2:
            y_preds = y_preds[:, -1]
        else:
            assert y_preds.dim() == 1
        metric = AUROC(task='binary')
    else:
        metric = AUROC(task='multiclass', num_classes=y_preds.shape[1])
    
    metric.to(device)
    return metric(y_preds, y_gts )

def getACC(y_preds: Tensor, y_gts: Tensor, task_type: str, device, threshold: float = 0.5):
    """Accuracy metric

    Args:
        y_preds: the predicted score of each class, shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        y_gts: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        task_type: the task of current dataset
        threshold: the threshold for multilabel and binary-class tasks
    """
    if task_type == "multi-label, binary-class":
        metric = Accuracy(task='multilabel', average='macro', threshold=threshold, num_labels=y_preds.shape[1])
    elif task_type == "binary-class":
        if y_preds.dim() == 2:
            y_preds = y_preds[:, -1]
        else:
            assert y_preds.dim() == 1
        y_preds = torch.where(y_preds > threshold, 1, 0)
        metric = Accuracy(task='binary', threshold=threshold)
    else: # multiclass
        num_class = y_preds.shape[-1]
        y_preds = y_preds.argmax(dim=1)
        metric = Accuracy(task='multiclass', average='macro', num_classes=num_class)
    
    metric.to(device)
    return metric(y_preds, y_gts)