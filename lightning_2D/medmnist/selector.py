import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsArchitecture(nn.Module):
    r"""An abstract class for MTL architectures.

    Args:
        task_name (list): A list of strings for all tasks.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        device (torch.device): The device where model and data will be allocated. 
        kwargs (dict): A dictionary of hyperparameters of architectures.
     
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device=None, **kwargs):
        super(AbsArchitecture, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder_class = encoder_class
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs
        
        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}
    
    def forward(self, inputs, task_name=None):
        r"""

        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        #TODO: bug???
        s_rep = self.encoder(inputs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out
    
    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad(set_to_none=False)
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep

class MMoE(AbsArchitecture):
    r"""Multi-gate Mixture-of-Experts (MMoE).
    
    This method is proposed in `Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018) <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ \
    and implemented by us.

    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
        num_experts (int): The number of experts shared for all tasks. Each expert is an encoder network.

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.img_size = self.kwargs['img_size']
        self.input_size = np.array(self.img_size, dtype=int).prod()
        self.num_experts = self.kwargs['num_experts']
        # NOTE: when encoder is resnet, the following line is not correct
        # self.experts_shared = nn.ModuleList([encoder_class(self.img_size) for _ in range(self.num_experts)])
        self.experts_shared = nn.ModuleList([encoder_class(encoder_type=kwargs['encoder_type']) for _ in range(self.num_experts)])
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, self.num_experts),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})
        
    def forward(self, inputs, task_name=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1)) 
            gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = self.decoders[task](gate_rep)
        return out
    
    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad(set_to_none=False)

class EntropyRegLoss(nn.Module):
    def __init__(self, power_of_2):
        super(EntropyRegLoss, self).__init__()
        self._power_of_2 = power_of_2
    
    def forward(self, inputs):
        loss = -(inputs*torch.log(inputs+1e-6)).sum() * 1e-6
        if not self._power_of_2:
            num_batch = inputs.shape[0]
            num_non_zero_experts = inputs.shape[1]
            # loss += (1/inputs.sum(-1)).sum()
            loss += (1/inputs.sum(-1).clamp(min=1e-2)).sum()
            loss = loss.div(num_batch*num_non_zero_experts) - 1 # regularization term cannot be more than 1
        return loss

class DSelect_k(MMoE):
    r"""DSelect-k.
    
    This method is proposed in `DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning (NeurIPS 2021) <https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html>`_ \
    and implemented by modifying from the `official TensorFlow implementation <https://github.com/google-research/google-research/tree/master/dselect_k_moe>`_. 

    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
        num_experts (int): The number of experts shared by all the tasks. Each expert is an encoder network.
        num_nonzeros (int): The number of selected experts.
        kgamma (float, default=1.0): A scaling parameter for the smooth-step function.

    """
    def __init__(self, task_name, encoder_class, decoders, device=None, multi_input=True, rep_grad=False, **kwargs):
        """Initialize DSelect_k
        Args:
            task_name (list): List of task names.
            encoder_class (nn.Module): Class that works as an expert
            decoders (nn.Module): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
            device: The device where model and data will be allocated.
            multi_input : defaults to True
            rep_grad: defaults to False
            img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
            num_experts (int): The number of experts shared by all the tasks. Each expert is an encoder network.
            num_nonzeros (int): The number of selected experts.
            kgamma (float, default=1.0): A scaling parameter for the smooth-step function.
        """
        super(DSelect_k, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self._num_nonzeros = self.kwargs['num_nonzeros']
        self._gamma = self.kwargs['kgamma']
        
        self._num_binary = math.ceil(math.log2(self.num_experts))
        self._power_of_2 = (self.num_experts == 2 ** self._num_binary)
        
        self._z_logits = nn.ModuleDict({task: nn.Linear(self.input_size, 
                                                        self._num_nonzeros*self._num_binary) for task in self.task_name})
        self._w_logits = nn.ModuleDict({task: nn.Linear(self.input_size, self._num_nonzeros) for task in self.task_name})
        self.entropy_reg_loss = EntropyRegLoss(self._power_of_2)
        
        # initialization
        for param in self._z_logits.parameters():
            param.data.uniform_(-self._gamma/100, self._gamma/100)
        for param in self._w_logits.parameters():
            param.data.uniform_(-0.05, 0.05)
        
        binary_matrix = np.array([list(np.binary_repr(val, width=self._num_binary)) \
                                  for val in range(self.num_experts)]).astype(bool)
        
        self._binary_codes = torch.from_numpy(binary_matrix).to(self.device).unsqueeze(0)
        # self._binary_codes = torch.from_numpy(binary_matrix).unsqueeze(0)  
        self.gate_specific = None
        
    def _smooth_step_fun(self, t, gamma=1.0):
        return torch.where(t<=-gamma/2, torch.zeros_like(t, device=t.device),
                   torch.where(t>=gamma/2, torch.ones_like(t, device=t.device),
                         (-2/(gamma**3))*(t**3) + (3/(2*gamma))*t + 1/2))
    
    def _entropy_reg_loss(self, inputs):
        loss = -(inputs*torch.log(inputs+1e-6)).sum() * 1e-6
        if not self._power_of_2:
            loss += (1/inputs.sum(-1)).sum()
        return loss
        #loss.backward(retain_graph=True)
    
    def set_device(self, device):
        self.device = device
    
    def forward(self, inputs, task):
        """Forward path for a particular task.

        Args:
            inputs: Input to selector
            task: A specific dataset name.

        Returns:
            _description_
        """
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])

        sample_logits = self._z_logits[task](torch.flatten(inputs, start_dim=1))
        sample_logits = sample_logits.reshape(-1, self._num_nonzeros, 1, self._num_binary)
        smooth_step_activations = self._smooth_step_fun(sample_logits)

        # TODO: fix infinity error here
        selector_output = torch.where(self._binary_codes.unsqueeze(0), smooth_step_activations, 
                                        1 - smooth_step_activations).prod(3)
        selector_weights = F.softmax(self._w_logits[task](torch.flatten(inputs, start_dim=1)), dim=1)
        expert_weights = torch.einsum('ij, ij... -> i...', selector_weights, selector_output)
        gate_rep = torch.einsum('ij, ji... -> i...', expert_weights, experts_shared_rep)
        gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
        pred = self.decoders[task](gate_rep)
        
        # if self.training:
        #     # backward
        #     self._entropy_reg_loss(selector_outputs)
        #NOTE: what is the shape of selector_outputs ???
        return pred, selector_output