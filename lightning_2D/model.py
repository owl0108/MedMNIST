import torch
import torch.nn as nn
from resnet import resnet18
from convnext_model.convnext import convnext_tiny

class Encoder(nn.Module):
        def __init__(self, encoder_type='resnet18'):
            super(Encoder, self).__init__()
            # hidden_dim = 512
            if encoder_type == 'resnet18':
                self.network = resnet18(pretrained=False) # already global avg pooled
            elif encoder_type == 'convnext_tiny':
                self.network = convnext_tiny(pretrained=False, num_classes=512) # alrady global avg pooled

            
        def forward(self, inputs):
            out = self.network(inputs) # out: 512 planes, 7x7?
            out = torch.flatten(out, start_dim=1)
            return out

class LinearModelHead(nn.Module):
    def __init__(self, input_dim):
        super(LinearModelHead, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        return self.fc(x)