import itertools

import torch
import torch.nn as nn
from resnet import resnet18
from convnext_model.convnext import convnext_tiny

class Encoder(nn.Module):
        def __init__(self, pretrained, encoder_type='resnet18'):
            super(Encoder, self).__init__()
            # hidden_dim = 512
            if encoder_type == 'resnet18':
                if pretrained:
                     print("Loading pretrained resnet18 weight ...")
                self.network = resnet18(pretrained) # already global avg pooled
            elif encoder_type == 'convnext_tiny':
                if pretrained:
                     print("Loading pretrained convnext_tiny weight ...")
                     self.network = nn.Sequential(
                          convnext_tiny(pretrained=pretrained, in_22k=True, num_classes=21841),
                          nn.Linear(21841, 512)
                     )
                else:
                     self.network = convnext_tiny(pretrained=pretrained, num_classes=512) # alrady global avg pooled

            
        def forward(self, inputs):
            out = self.network(inputs) # out: 512 planes, 7x7?
            out = torch.flatten(out, start_dim=1)
            return out

class LinearModelHead(nn.Module):
    def __init__(self, input_dim, layers=1):
        super(LinearModelHead, self).__init__()
        linear_layers = [[nn.Linear(input_dim, input_dim),
                          nn.ReLU()] for i in range(layers)]
        linear_layers = list(itertools.chain(*linear_layers))
        self.fc = nn.Sequential(*linear_layers)
    
    def forward(self, x):
        return self.fc(x)