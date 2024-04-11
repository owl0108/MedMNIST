from LibMTL.loss import AbsLoss
from torch import nn

class CELoss(AbsLoss):
    """The cross-entropy loss function. Softmax is already combined into it.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)

class BCELoss(AbsLoss):
    """The binary cross-entropy loss function. Sigmoid is combined into it.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)


